"""An in-process stand-in for the parts of the `modal` package ModalSandbox uses.

Volumes are folders; a sandbox is a folder whose mount paths are symlinks to
its volumes' folders, and exec runs the real launcher (bmb_eval.modal_entry) as
a local subprocess, unprivileged and NOT isolated (like LocalSandbox). Every
created sandbox is recorded, so tests can check which volumes each stage got.

Knobs: fail_creates (that many Sandbox.create calls raise a Modal error),
lose_execs (that many execs find their sandbox gone, with no exit code).
"""
from __future__ import annotations

import contextlib
import enum
import itertools
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from bmb_eval.config import PKG


class FileEntryType(enum.IntEnum):     # as modal's: str() of a member is its number
    UNSPECIFIED = 0
    FILE = 1
    DIRECTORY = 2


class Error(Exception):
    pass


class NotFoundError(Error):
    pass


class ResourceExhaustedError(Error):
    pass


class FakeModal:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.created: list[dict] = []
        self.sandboxes: list[FakeSandbox] = []
        self.fail_creates = 0
        self.no_capacity = 0       # that many Sandbox.create calls find no GPU free (Modal's message)
        self.lose_execs = 0
        self.rate_limit = 0        # that many volume listings/reads raise ResourceExhaustedError
        self.api_calls = 0         # volume listings/reads made
        self._ids = itertools.count()
        self._lock = threading.Lock()
        fm = self
        self.exception = SimpleNamespace(Error=Error, NotFoundError=NotFoundError,
                                         ResourceExhaustedError=ResourceExhaustedError)
        self.App = SimpleNamespace(lookup=lambda name, create_if_missing=False, environment_name=None:
                                   SimpleNamespace(app_id=f"ap-{environment_name}-{name}", name=name))
        self.Image = SimpleNamespace(
            from_id=lambda image_id, client=None: SimpleNamespace(object_id=image_id),
            from_dockerfile=lambda path, context_dir=None, ignore=None, **k: FakeImage(path, context_dir, ignore))
        self.Volume = SimpleNamespace(
            from_name=lambda name, environment_name=None, create_if_missing=False:
                FakeVolume(fm, name, environment_name, create_if_missing),
            objects=SimpleNamespace(delete=self._delete_volume, list=self._list_volumes))
        self.Sandbox = SimpleNamespace(create=self._create, list=self._list_sandboxes)

    # volumes
    def volume_dir(self, env: str | None, name: str) -> Path:
        return self.root / "volumes" / (env or "main") / name

    def _delete_volume(self, name, environment_name=None, allow_missing=False, client=None):
        d = self.volume_dir(environment_name, name)
        if not d.exists() and not allow_missing:
            raise NotFoundError(name)
        shutil.rmtree(d, ignore_errors=True)

    def _list_volumes(self, environment_name="", **k):
        d = self.root / "volumes" / (environment_name or "main")
        return [SimpleNamespace(name=p.name) for p in sorted(d.iterdir())] if d.exists() else []

    # sandboxes
    def _create(self, *cmd, app=None, image=None, volumes=None, gpu=None, cpu=None, memory=None,
                timeout=None, block_network=False, tags=None, **kw):
        with self._lock:
            if self.no_capacity:
                self.no_capacity -= 1
                raise ResourceExhaustedError("Insufficient capacity to create sandbox.")
            if self.fail_creates:
                self.fail_creates -= 1
                raise Error("fake: no capacity")
            n = next(self._ids)
        sb = FakeSandbox(self, n, dict(tags or {}))
        for path, v in (volumes or {}).items():
            target = Path(sb.fake_root + path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not v.dir.exists():
                raise NotFoundError(v.name)
            target.symlink_to(v.dir)
        self.created.append({"cmd": cmd, "gpu": gpu, "cpu": cpu, "memory": memory, "timeout": timeout,
                             "block_network": block_network, "tags": dict(tags or {}),
                             "volumes": {p: (v.name, v.ro) for p, v in (volumes or {}).items()},
                             "image": getattr(image, "object_id", None), "sandbox": sb})
        self.sandboxes.append(sb)
        return sb

    def _list_sandboxes(self, app_id=None, tags=None, client=None):
        return [s for s in self.sandboxes if s.code is None
                and all(s.tags.get(k) == v for k, v in (tags or {}).items())]

    def stage_of(self, rec: dict) -> str | None:
        """Which entry a recorded sandbox ran (from its exec), or "stage" for the stager."""
        return rec["sandbox"].entry


class FakeImage:
    def __init__(self, path, context_dir, ignore):
        self.path, self.context_dir, self.ignore = Path(path), context_dir, ignore
        self.object_id = None

    def build(self, app):
        self.object_id = "im-fake0001"
        return self


class FakeVolume:
    def __init__(self, fm: FakeModal, name: str, env: str | None, create: bool, ro: bool = False):
        self.fm, self.name, self.env, self.ro = fm, name, env, ro
        self.dir = fm.volume_dir(env, name)
        if create:
            self.dir.mkdir(parents=True, exist_ok=True)

    def read_only(self):
        return FakeVolume(self.fm, self.name, self.env, False, ro=True)

    def hydrate(self):
        if not self.dir.exists():
            raise NotFoundError(self.name)
        return self

    def _need(self):
        with self.fm._lock:
            self.fm.api_calls += 1
            if self.fm.rate_limit:
                self.fm.rate_limit -= 1
                raise ResourceExhaustedError("fake: rate limit exceeded")
        if not self.dir.exists():
            raise NotFoundError(self.name)

    def listdir(self, path, recursive=False):
        self._need()
        base = self.dir / path.strip("/")
        it = base.rglob("*") if recursive else base.iterdir()
        return [SimpleNamespace(path=p.relative_to(self.dir).as_posix(), size=p.stat().st_size,
                                type=FileEntryType.FILE if p.is_file() else FileEntryType.DIRECTORY)
                for p in sorted(it)]

    def read_file(self, path):
        self._need()
        p = self.dir / path.strip("/")
        if not p.is_file():
            raise NotFoundError(path)
        yield p.read_bytes()

    def remove_file(self, path, recursive=False):
        p = self.dir / path.strip("/")
        shutil.rmtree(p) if p.is_dir() else p.unlink()

    @contextlib.contextmanager
    def batch_upload(self, force=False):
        self._need()
        vol = self

        class Up:
            def put_file(self, src, remote):
                dst = vol.dir / remote.strip("/")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(src, (str, os.PathLike)):
                    shutil.copyfile(src, dst)
                else:
                    dst.write_bytes(src.read())
        yield Up()


class FakeFS:
    def __init__(self, sb):
        self.sb = sb

    def _alive(self):
        if self.sb.code is not None:
            raise NotFoundError("sandbox has shut down")

    def make_directory(self, path, create_parents=True):
        self._alive()
        Path(path).mkdir(parents=True, exist_ok=True)

    def write_bytes(self, data, path):
        self._alive()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(bytes(data))

    def read_bytes(self, path):
        self._alive()
        return Path(path).read_bytes()

    def stat(self, path):
        self._alive()
        return SimpleNamespace(size=Path(path).stat().st_size)


class FakeProcess:
    def __init__(self, popen):
        self.p = popen
        self._out = None
        self.returncode = None
        sb_out = self

        class Stream:
            def read(self_inner):
                sb_out._finish()
                return sb_out._out
        self.stdout = Stream()

    def _finish(self):
        if self._out is None:
            self._out, _ = self.p.communicate()
            self.returncode = self.p.returncode

    def wait(self):
        self._finish()
        return self.returncode


class FakeSandbox:
    def __init__(self, fm: FakeModal, n: int, tags: dict):
        self.fm, self.tags = fm, tags
        self.fake_root = str(fm.root / "sandboxes" / str(n))
        Path(self.fake_root, "tmp").mkdir(parents=True)
        self.code = None
        self.entry = None
        self.object_id = f"sb-{n}"
        self.filesystem = FakeFS(self)
        self._procs = []

    def exec(self, *args, text=True, **kw):
        if self.code is not None:
            raise NotFoundError("sandbox has shut down")
        args = list(args)
        if args[:3] == ["python", "-m", "bmb_eval.modal_entry"]:
            import json
            self.entry = json.loads(Path(args[3]).read_text())["entry"]
        elif args[:3] == ["python", "-m", "bmb_eval.modal_stage"]:
            self.entry = "stage"
        with self.fm._lock:
            if self.fm.lose_execs and self.entry != "stage":
                self.fm.lose_execs -= 1
                raise NotFoundError("fake: sandbox not found; it has already shut down")
        if args[0] == "python":
            args[0] = sys.executable
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(PKG.parent), "HOME": self.fake_root + "/tmp"}
        with open(Path(self.fake_root, "launcher.err"), "ab") as err:   # for debugging tests
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=err, env=env, cwd=self.fake_root)
        self._procs.append(p)
        return FakeProcess(p)

    def poll(self):
        return self.code

    def terminate(self, wait=False):
        for p in self._procs:
            if p.poll() is None:
                p.kill()
        if self.code is None:
            self.code = 137

    def wait(self, raise_on_termination=True):
        for p in self._procs:
            p.wait()
        return self.code

    def tunnels(self):
        return {}


def install(monkeypatch, root: Path) -> FakeModal:
    """Point bmb_eval's Modal backend at a FakeModal."""
    from bmb_eval import modal_backend
    fm = FakeModal(root)
    monkeypatch.setattr(modal_backend, "import_modal", lambda settings=None: fm)
    return fm


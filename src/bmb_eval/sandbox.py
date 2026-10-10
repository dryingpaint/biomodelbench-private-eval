"""Sandbox backends.

ContainerSandbox is the production backend, driving either docker or podman
(same flags): no network, read-only root, all capabilities dropped, no new
privileges, memory/cpu/pid/file limits, no container logging, and private
files bind-mounted one by one (never their parent directory), read-only.
The container runs as the operator's uid so it can read exactly the files
mounted into it without the operator loosening any file permissions.

Nothing in a sandbox can write to the host. Data moves only through pipes:
the controller writes stdin, and reads stdout up to a byte cap (killing the
container if it is exceeded) and the tail of stderr for the provider-only
run log.

Set sandbox.runtime: runsc in config.yaml to add gVisor's kernel isolation
on top (docker or podman); it is the recommended production setting.

LocalSandbox runs the same entrypoints as plain subprocesses. It is NOT
isolated and exists for tests and for agents dry-running bundles on
synthetic data.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import PKG
from .contract import Limits

LOG_CAP = 1_000_000  # bytes of stderr kept (tail) for the internal run log


@dataclass
class Mount:
    env: str           # env var the entrypoint reads the path from
    host: Path
    container: str


@dataclass
class Result:
    exit_code: int
    timed_out: bool
    stdout: bytes
    stdout_overflow: bool
    stderr: str
    seconds: float


ENGINES = ("docker", "podman")


def detect_engine() -> str | None:
    """First container engine whose daemon/service actually answers."""
    for engine in ("podman", "docker"):
        if shutil.which(engine) and subprocess.run(
                [engine, "info"], capture_output=True).returncode == 0:
            return engine
    return None


def container_cmd(engine: str, image: str, runtime: str | None, *, name: str, entry: str,
                  mounts: list[Mount], env: dict[str, str], limits: Limits) -> list[str]:
    uid = f"{os.getuid()}:{os.getgid()}"
    cmd = [
        engine, "run", "--rm", "-i", "--name", name,
        "--label", "bmb-eval=1",
        "--network", "none",
        "--log-driver", "none",          # stdout/stderr never land in /var/lib/docker
        "--read-only",
        "--tmpfs", f"/tmp:rw,exec,nosuid,nodev,size={limits.tmpfs_mb}m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", str(limits.pids),
        "--memory", limits.memory, "--memory-swap", limits.memory,
        "--cpus", str(limits.cpus),
        "--ulimit", "core=0", "--ulimit", "nofile=4096:4096",
        "--user", uid,
        "--ipc", "private",
        "--hostname", "sandbox",
        "-e", "HOME=/tmp", "-e", "TMPDIR=/tmp", "-e", "PYTHONHASHSEED=0",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
    ]
    if engine == "podman":
        cmd += ["--userns", "keep-id"]  # rootless: container uid == operator uid
    if runtime:
        cmd += ["--runtime", runtime]
    for m in mounts:
        cmd += ["-v", f"{m.host.resolve()}:{m.container}:ro", "-e", f"{m.env}={m.container}"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [image, "python", "-m", f"bmb_eval.sandbox_entry.{entry}"]
    return cmd


class ContainerSandbox:
    def __init__(self, engine: str, image: str, runtime: str | None = None):
        self.engine = engine
        self.image = image
        self.runtime = runtime

    def run(self, *, name: str, entry: str, mounts: list[Mount], env: dict[str, str],
            limits: Limits, timeout_s: int, stdout_cap: int, stdin: bytes = b"") -> Result:
        cmd = container_cmd(self.engine, self.image, self.runtime, name=name, entry=entry,
                            mounts=mounts, env=env, limits=limits)
        # Killing the CLI client does not stop the container; kill it by name.
        return _run(cmd, timeout_s, stdout_cap, stdin, on_kill=lambda: subprocess.run(
            [self.engine, "kill", name], capture_output=True))

    def kill_all(self) -> int:
        ids = subprocess.run([self.engine, "ps", "-q", "--filter", "label=bmb-eval=1"],
                             capture_output=True, text=True).stdout.split()
        if ids:
            subprocess.run([self.engine, "kill", *ids], capture_output=True)
        return len(ids)

    def image_id(self) -> str:
        try:
            r = subprocess.run([self.engine, "image", "inspect", "-f", "{{.Id}}", self.image],
                               capture_output=True, text=True)
        except FileNotFoundError:
            return "missing"
        return r.stdout.strip() or "missing"


class LocalSandbox:
    """Unisolated. Tests and dry runs only."""

    def run(self, *, name: str, entry: str, mounts: list[Mount], env: dict[str, str],
            limits: Limits, timeout_s: int, stdout_cap: int, stdin: bytes = b"") -> Result:
        e = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "TMPDIR")}
        e.update(env)
        e["PYTHONHASHSEED"] = "0"
        e["PYTHONPATH"] = str(PKG.parent)
        for m in mounts:
            e[m.env] = str(m.host.resolve())
        cmd = [sys.executable, "-m", f"bmb_eval.sandbox_entry.{entry}"]
        return _run(cmd, timeout_s, stdout_cap, stdin, env=e)

    def kill_all(self) -> int:
        return 0

    def image_id(self) -> str:
        return "local-unisolated"


def _run(cmd: list[str], timeout_s: int, stdout_cap: int, stdin: bytes,
         env: dict | None = None, on_kill=None) -> Result:
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    out = bytearray()
    err = bytearray()
    overflow = threading.Event()

    def kill() -> None:
        if on_kill:
            on_kill()
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    def read_stdout() -> None:
        while chunk := proc.stdout.read(1 << 16):
            if len(out) + len(chunk) > stdout_cap:
                overflow.set()
                kill()
                break
            out.extend(chunk)
        proc.stdout.close()

    def read_stderr() -> None:
        # Keep draining so the child never blocks, but hold only the tail.
        while chunk := proc.stderr.read(1 << 16):
            err.extend(chunk)
            if len(err) > 2 * LOG_CAP:
                del err[:-LOG_CAP]
        proc.stderr.close()

    def write_stdin() -> None:
        try:
            proc.stdin.write(stdin)
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    threads = [threading.Thread(target=t, daemon=True)
               for t in (read_stdout, read_stderr, write_stdin)]
    for t in threads:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill()
        proc.wait()
    for t in threads:
        t.join(timeout=10)
    return Result(
        exit_code=proc.returncode,
        timed_out=timed_out and not overflow.is_set(),
        stdout=bytes(out),
        stdout_overflow=overflow.is_set(),
        stderr=bytes(err[-LOG_CAP:]).decode(errors="replace"),
        seconds=round(time.time() - t0, 2),
    )


def make(cfg) -> ContainerSandbox | LocalSandbox:
    if cfg.backend in ENGINES:
        return ContainerSandbox(cfg.backend, cfg.image, cfg.runtime)
    return LocalSandbox()

"""Sandbox backends.

ContainerSandbox is the production backend, driving either docker or podman
(same flags; GPUs via the NVIDIA container toolkit): no network, read-only root, all capabilities dropped, no new
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

BwrapSandbox is for the confidential-computing node, which runs as a single
container (Confidential Space) where container engines can't run. Each stage
runs under bubblewrap as an unprivileged user: no network, its own minimal
filesystem (system files read-only plus exactly the mounted files), private
/tmp, process and file limits. Mounted files are made readable to the
sandbox user only for the stage that mounts them.

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

# What goes into the sandbox image; deploy/gcp/hub-deploy.sh stamps the image it
# publishes with the same digest (label bmb.sandbox), so a node can tell when its
# image is older than its code: the validator inside would then fail every run.
# The lock pins every package the image installs, so the stamp names the image's
# whole content, not just our code (tests/test_release.py keeps the lists in step).
SANDBOX_FILES = ("sandbox/Dockerfile", "sandbox/requirements.txt", "sandbox/requirements.lock",
                 "src/bmb_eval/__init__.py",
                 "src/bmb_eval/contract.py", "src/bmb_eval/errors.py",
                 "src/bmb_eval/sandbox_entry/__init__.py", "src/bmb_eval/sandbox_entry/predict.py",
                 "src/bmb_eval/sandbox_entry/validate.py", "src/bmb_eval/sandbox_entry/score.py")


def sandbox_stamp(root: Path) -> str | None:
    """Digest of the sandbox's code, as `cat $SANDBOX_FILES | sha256sum | cut -c1-16`."""
    import hashlib
    h = hashlib.sha256()
    for f in SANDBOX_FILES:
        if not (root / f).is_file():
            return None
        h.update((root / f).read_bytes())
    return h.hexdigest()[:16]


def stale_image_message(cfg) -> str | None:
    """Why this node's container image can't run its code, or None."""
    if cfg.backend not in ENGINES:
        return None
    root = next((d for d in Path(__file__).resolve().parents if (d / "sandbox" / "Dockerfile").is_file()), None)
    want = sandbox_stamp(root) if root else None
    if want is None:
        return None
    have = ContainerSandbox(cfg.backend, cfg.image, cfg.runtime).image_label("bmb.sandbox")
    if have == want:
        return None
    if have is None and (why := engine_access_problem(cfg.backend)):
        return why
    return (f"the sandbox image ({have or 'unstamped'}) is older than this node's code ({want}); "
            "run `bmb-eval build-image` (or re-run the install command) to update it")


ENGINE_QUERY_TIMEOUT_S = 60


def engine_query(cmd: list[str], timeout_s: int = ENGINE_QUERY_TIMEOUT_S):
    """Run a quick container-engine query (info, version); None if it can't run or doesn't
    answer in time. Seen in CI: rootless `podman info` sometimes never returns while it sets
    itself up, which hung connect (and the test suite) with no time limit."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return None


def engine_access_problem(engine: str) -> str | None:
    """Why this process can't use the container engine, when that's the reason it sees no
    image (a user added to the docker group after their session started); None otherwise."""
    r = engine_query([engine, "info"])
    if r is None:
        return None
    if r.returncode != 0 and "permission denied" in (r.stderr + r.stdout).lower():
        return (f"this process can't use {engine} (permission denied on its socket). If you were just "
                f"added to the {engine} group, re-run the install command (it restarts the background "
                "service with the new group), or log out and back in")
    return None


def detect_engine() -> str | None:
    """First container engine whose daemon/service actually answers."""
    for engine in ("podman", "docker"):
        if shutil.which(engine) and (r := engine_query([engine, "info"])) and r.returncode == 0:
            return engine
    return None


def container_cmd(engine: str, image: str, runtime: str | None, *, name: str, entry: str,
                  mounts: list[Mount], env: dict[str, str], limits: Limits,
                  gpus: int = 0) -> list[str]:
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
        "--cpus", str(min(limits.cpus, os.cpu_count() or 1)),
        "--ulimit", "core=0", "--ulimit", "nofile=4096:4096",
        "--user", uid,
        "--ipc", "private", "--shm-size", f"{limits.shm_mb}m",
        "--hostname", "sandbox",
        "-e", "HOME=/tmp", "-e", "TMPDIR=/tmp", "-e", "PYTHONHASHSEED=0",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
    ]
    # Math libraries start one thread per host CPU, not per allowed CPU: on a big
    # machine that alone can exceed the pid limit (OpenBLAS then crashes).
    threads = str(max(1, int(min(limits.cpus, os.cpu_count() or 1))))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        cmd += ["-e", f"{v}={threads}"]
    if engine == "podman":
        cmd += ["--userns", "keep-id"]  # rootless: container uid == operator uid
    if runtime:
        cmd += ["--runtime", runtime]
    if gpus:
        # Podman uses CDI device names; Docker the NVIDIA runtime hook.
        cmd += (["--device", "nvidia.com/gpu=all"] if engine == "podman" else ["--gpus", "all"])
    # SELinux (RHEL, Rocky, Alma, Fedora): mounted files need the container label or the
    # sandbox can't read them. "z" (shared) rather than "Z": inputs can be hard links to the
    # provider's own files, which other programs of theirs may still need to read.
    opts = "ro,z" if Path("/sys/fs/selinux/enforce").exists() else "ro"
    for m in mounts:
        cmd += ["-v", f"{m.host.resolve()}:{m.container}:{opts}", "-e", f"{m.env}={m.container}"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [image, "python", "-m", f"bmb_eval.sandbox_entry.{entry}"]
    return cmd


def gpus_here() -> list[int]:
    """Memory (GB) of each NVIDIA GPU on this machine; [] if none."""
    if not shutil.which("nvidia-smi"):
        return []
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return [int(x) // 1024 for x in r.stdout.split()] if r.returncode == 0 else []


class ContainerSandbox:
    def __init__(self, engine: str, image: str, runtime: str | None = None):
        self.engine = engine
        self.image = image
        self.runtime = runtime

    def run(self, *, name: str, entry: str, mounts: list[Mount], env: dict[str, str],
            limits: Limits, timeout_s: int, stdout_cap: int, stdin: bytes = b"",
            gpus: int = 0) -> Result:
        cmd = container_cmd(self.engine, self.image, self.runtime, name=name, entry=entry,
                            mounts=mounts, env=env, limits=limits, gpus=gpus)
        # Killing the CLI client does not stop the container; kill it by name.
        return _run(cmd, timeout_s, stdout_cap, stdin, on_kill=lambda: subprocess.run(
            [self.engine, "kill", name], capture_output=True))

    def kill_all(self) -> int:
        ids = subprocess.run([self.engine, "ps", "-q", "--filter", "label=bmb-eval=1"],
                             capture_output=True, text=True).stdout.split()
        if ids:
            subprocess.run([self.engine, "kill", *ids], capture_output=True)
        return len(ids)

    def image_label(self, key: str) -> str | None:
        try:
            r = subprocess.run([self.engine, "image", "inspect", "-f",
                                "{{index .Config.Labels \"%s\"}}" % key, self.image],
                               capture_output=True, text=True)
        except FileNotFoundError:
            return None
        v = r.stdout.strip()
        return v if r.returncode == 0 and v and v != "<no value>" else None

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
            limits: Limits, timeout_s: int, stdout_cap: int, stdin: bytes = b"",
            gpus: int = 0) -> Result:
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


SANDBOX_UID = 65534    # "nobody": owns nothing on the node, so it can read only what a stage grants

# Kernel interfaces a model never needs and container escapes often use. The
# sealed node's sandbox shares the kernel (bubblewrap; gVisor can't reach the GPU
# inside Confidential Space), so these fail with EPERM there. clone3 gets ENOSYS
# so libc falls back to clone, whose CLONE_NEWUSER flag is refused.
BLOCKED_SYSCALLS = (
    "unshare", "setns", "mount", "umount2", "pivot_root", "chroot", "move_mount", "open_tree",
    "fsopen", "fsconfig", "fsmount", "fspick", "mount_setattr", "ptrace", "process_vm_readv",
    "process_vm_writev", "kcmp", "bpf", "perf_event_open", "userfaultfd", "io_uring_setup",
    "io_uring_enter", "io_uring_register", "keyctl", "add_key", "request_key", "kexec_load",
    "kexec_file_load", "init_module", "finit_module", "delete_module", "reboot", "swapon",
    "swapoff", "acct", "quotactl", "open_by_handle_at", "name_to_handle_at", "syslog",
    "fanotify_init", "lookup_dcookie", "vhangup", "iopl", "ioperm")
CLONE_NEWUSER = 0x10000000


def seccomp_filter(path: Path) -> Path:
    """Write the sandbox's seccomp BPF program (default allow, BLOCKED_SYSCALLS denied)."""
    import errno
    import pyseccomp as sc
    f = sc.SyscallFilter(sc.ALLOW)
    for name in BLOCKED_SYSCALLS:
        try:
            f.add_rule(sc.ERRNO(errno.EPERM), name)
        except Exception:
            pass                       # not a syscall on this architecture
    f.add_rule(sc.ERRNO(errno.ENOSYS), "clone3")
    f.add_rule(sc.ERRNO(errno.EPERM), "clone", sc.Arg(0, sc.MASKED_EQ, CLONE_NEWUSER, CLONE_NEWUSER))
    with open(path, "wb") as out:
        f.export_bpf(out)
    return path


def _bytes(size: str) -> int:
    n, unit = size.strip().lower()[:-1], size.strip().lower()[-1]
    return int(float(n) * {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[unit]) if unit in "kmg" else int(size)


class BwrapSandbox:
    """bubblewrap stages for the confidential node (see module docstring)."""

    SYSTEM = ["/usr", "/etc", "/opt/bmb"]       # read-only; /usr holds python and the GPU driver

    def __init__(self, image_id: str = "bwrap", seccomp: Path | None = None):
        self._image_id = image_id
        self.seccomp = seccomp          # BPF file from seccomp_filter(); None = no filter

    def cmd(self, *, entry: str, mounts: list[Mount], env: dict[str, str], limits: Limits,
            gpus: int = 0, seccomp_fd: int | None = None) -> list[str]:
        cmd = ["bwrap", "--unshare-user", "--unshare-net", "--unshare-ipc", "--unshare-uts",
               "--unshare-cgroup-try", "--die-with-parent", "--new-session",
               "--uid", str(SANDBOX_UID), "--gid", str(SANDBOX_UID), "--hostname", "sandbox"]
        for d in self.SYSTEM:
            cmd += ["--ro-bind-try", d, d]
        for link, target in (("/bin", "usr/bin"), ("/sbin", "usr/sbin"), ("/lib", "usr/lib"),
                             ("/lib64", "usr/lib64")):
            cmd += ["--symlink", target, link]
        cmd += ["--ro-bind", "/proc", "/proc", "--dev", "/dev",
                "--size", str(limits.tmpfs_mb << 20), "--tmpfs", "/tmp",
                "--size", str(limits.shm_mb << 20), "--tmpfs", "/dev/shm"]
        if gpus:
            for d in sorted(Path("/dev").glob("nvidia*")):
                cmd += ["--dev-bind", str(d), str(d)]
        for m in mounts:
            cmd += ["--ro-bind", str(m.host.resolve()), m.container]
        e = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp",
             "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "/opt/bmb",
             "PIP_NO_INDEX": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             # No cgroups here: size math-library thread pools (and their buffers) to the limit.
             **{v: str(max(1, int(limits.cpus))) for v in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
             **({"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64"} if gpus else {}),
             **{m.env: m.container for m in mounts}, **env}
        cmd += ["--remount-ro", "/"]       # the sandbox's own root: only /tmp and /dev/shm are writable
        if seccomp_fd is not None:
            cmd += ["--seccomp", str(seccomp_fd)]   # the BPF program, an fd passed by run()
        cmd += ["--clearenv", *[x for k, v in e.items() for x in ("--setenv", k, v)],
                "--chdir", "/tmp", "python3", "-m", f"bmb_eval.sandbox_entry.{entry}"]
        return cmd

    def run(self, *, name: str, entry: str, mounts: list[Mount], env: dict[str, str],
            limits: Limits, timeout_s: int, stdout_cap: int, stdin: bytes = b"",
            gpus: int = 0) -> Result:
        import resource

        def limit() -> None:          # in the child, before exec: become the sandbox user
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write("1000")       # under memory pressure, the kernel kills this, not the node
            os.setgroups([])
            os.setgid(SANDBOX_UID)
            os.setuid(SANDBOX_UID)
            resource.setrlimit(resource.RLIMIT_NPROC, (limits.pids, limits.pids))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
            if not gpus:              # CUDA maps huge regions; cap memory on CPU stages only
                mem = _bytes(limits.memory)   # data, not address space: libraries reserve plenty
                resource.setrlimit(resource.RLIMIT_DATA, (mem, mem))

        # GPU stages: the device files too (root-only inside the container otherwise).
        granted = _grant([m.host for m in mounts]
                         + (sorted(Path("/dev").glob("nvidia*")) if gpus else []))
        bpf = open(self.seccomp, "rb") if self.seccomp else None
        fd = bpf.fileno() if bpf else None
        done = threading.Event()
        if gpus:   # no rlimit can cap memory without breaking CUDA: watch it instead
            threading.Thread(target=self._watch_memory, args=(_bytes(limits.memory), done),
                             daemon=True).start()
        try:
            return _run(self.cmd(entry=entry, mounts=mounts, env=env, limits=limits, gpus=gpus,
                                 seccomp_fd=fd),
                        timeout_s, stdout_cap, stdin, env={}, on_kill=self.kill_all,
                        preexec=limit, pass_fds=(fd,) if bpf else ())
        finally:
            done.set()
            self.kill_all()           # nothing a stage started outlives it
            _revoke(granted)
            if bpf:
                bpf.close()

    def _watch_memory(self, cap: int, done: threading.Event) -> None:
        """Kill the stage (exit -9: predict_resource_limit) once the sandbox user's
        anonymous memory passes the limit. GPU memory and shared libraries are
        file-backed mappings, so they don't count."""
        while not done.wait(0.05):
            if sandbox_anon_bytes() > cap:
                self.kill_all()
                return

    def kill_all(self) -> int:
        return int(subprocess.run(["pkill", "-KILL", "-u", str(SANDBOX_UID)],
                                  capture_output=True).returncode == 0)

    def image_id(self) -> str:
        return self._image_id


def sandbox_anon_bytes(proc: Path = Path("/proc"), uid: int = SANDBOX_UID) -> int:
    """Resident anonymous memory (RssAnon) of every process the sandbox user runs."""
    total = 0
    for status in proc.glob("[0-9]*/status"):
        try:
            fields = dict(line.split(":", 1) for line in status.read_text().splitlines() if ":" in line)
        except OSError:           # the process exited meanwhile
            continue
        if fields.get("Uid", "").split()[:1] == [str(uid)] and "RssAnon" in fields:
            total += int(fields["RssAnon"].split()[0]) << 10
    return total


def _grant(paths: list[Path]) -> list[tuple[Path, int, int]]:
    """Make exactly these files (and folders' contents) readable to the sandbox
    user; returns what to restore. Everything else stays root-only."""
    out = []
    for top in paths:
        for p in [top, *(top.rglob("*") if top.is_dir() else [])]:
            st = p.lstat()
            if not p.is_symlink():
                out.append((p, st.st_uid, st.st_gid))
                os.chown(p, SANDBOX_UID, SANDBOX_UID)
    return out


def _revoke(granted: list[tuple[Path, int, int]]) -> None:
    for p, uid, gid in granted:
        try:
            os.chown(p, uid, gid)
        except FileNotFoundError:
            pass


def _run(cmd: list[str], timeout_s: int, stdout_cap: int, stdin: bytes,
         env: dict | None = None, on_kill=None, preexec=None, pass_fds=()) -> Result:
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env, preexec_fn=preexec, pass_fds=pass_fds)
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
            if overflow.is_set():
                continue                 # past the cap: discard, but keep the pipe flowing
            if len(out) + len(chunk) > stdout_cap:
                overflow.set()
                # Kill in the background and keep reading. With Docker the output passes
                # through the client and the daemon, and `docker kill` returns only once
                # the container is gone, which waits for its output to drain. Killing
                # here, or no longer reading, deadlocks (seen: hung for 13 hours).
                threading.Thread(target=kill, daemon=True).start()
                continue
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


def make(cfg) -> ContainerSandbox | BwrapSandbox | LocalSandbox:
    if cfg.backend in ENGINES:
        return ContainerSandbox(cfg.backend, cfg.image, cfg.runtime)
    if cfg.backend == "bwrap":
        bpf = cfg.state_dir / "sandbox.seccomp.bpf"
        if not bpf.exists():
            seccomp_filter(bpf)
        return BwrapSandbox(cfg.image, seccomp=bpf)
    return LocalSandbox()

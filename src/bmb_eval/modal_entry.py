"""Trusted launcher inside a Modal sandbox (the Modal backend's stand-in for the
container engine's flags). Runs as root, then runs one stage entrypoint
(bmb_eval.sandbox_entry.<entry>) as uid 65534:

- no capabilities, no new privileges, its own session; uid 65534 owns nothing
  in the image, so the root filesystem is read-only to it except /tmp;
- process (pids), core-file and open-file limits; the memory limit as a data
  limit on CPU stages and, on GPU stages (where a data limit breaks CUDA), as a
  watchdog on the sandbox user's anonymous memory (exit -9);
- the time limit (killed, timed_out);
- an environment of exactly what the stage needs (no Modal variables);
- stdout read here, byte-capped (killed past the cap, overflow), and written to
  a root-only file the controller reads after checking its size. Modal's stream
  from a sandbox to the controller carries a few MB/s, so stdout itself carries
  only the small status line below.

Usage: python -m bmb_eval.modal_entry <spec.json>   (written by ModalSandbox)
Prints one JSON line: {"exit_code", "timed_out", "overflow", "bytes"}.
Writes <run_dir>/out.bin (the stage's stdout) and <run_dir>/err.log (stderr tail).
"""
from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import threading
from pathlib import Path

UID = 65534
LOG_CAP = 1_000_000
PR_SET_NO_NEW_PRIVS = 38
# From the image's own environment: where bmb_eval is, and what the GPU stack needs.
PASS_ENV = ("PYTHONPATH", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "CUDA_VISIBLE_DEVICES",
            "LANG")


def _no_new_privs() -> None:
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS)")


def _child_setup(spec: dict):
    root = os.getuid() == 0
    mem = int(spec["memory_bytes"])

    def setup() -> None:                 # in the child, before exec
        os.setsid()
        try:
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write("1000")
        except OSError:
            pass
        if root:                         # always, on Modal (tests run it unprivileged)
            os.setgroups([])
            os.setgid(UID)
            os.setuid(UID)               # drops every capability
            # Counted per uid: uid 65534 runs only this stage, so this is its pid limit.
            resource.setrlimit(resource.RLIMIT_NPROC, (spec["pids"], spec["pids"]))
        if sys.platform.startswith("linux"):
            _no_new_privs()
        limits = [(resource.RLIMIT_CORE, 0), (resource.RLIMIT_NOFILE, 4096)]
        if not spec["gpus"]:             # CUDA maps huge regions; cap memory on CPU stages only
            limits.append((resource.RLIMIT_DATA, mem))
        for which, n in limits:
            try:
                resource.setrlimit(which, (n, n))
            except (ValueError, OSError):
                if root:                 # on Modal every limit must hold
                    raise
    return setup


def anon_bytes(uid: int) -> int:
    """Resident anonymous memory of every process this uid runs (as sandbox.sandbox_anon_bytes)."""
    total = 0
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = dict(line.split(":", 1) for line in status.read_text().splitlines() if ":" in line)
        except OSError:
            continue
        if fields.get("Uid", "").split()[:1] == [str(uid)] and "RssAnon" in fields:
            total += int(fields["RssAnon"].split()[0]) << 10
    return total


def kill_stage(proc: subprocess.Popen) -> None:
    """Kill everything the stage started: as uid 65534, kill(-1) reaches exactly its
    processes (not this launcher, not the sandbox's own)."""
    if os.getuid() == 0:
        pid = os.fork()
        if pid == 0:
            try:
                os.setgid(UID)
                os.setuid(UID)
                os.kill(-1, signal.SIGKILL)
            finally:
                os._exit(0)
        os.waitpid(pid, 0)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass


def mirror_inputs(m: dict) -> None:
    """Make the inputs folder the model sees at m["dst"]: the task's inputs volume
    (m["src"]) plus large shared files (the genome) stored once, each in its own
    volume under m["shared"]/<sha[:16]>/blob, at the paths .bmb_links.json gives."""
    src, dst, shared = Path(m["src"]), Path(m["dst"]), Path(m["shared"])
    links = json.loads((src / ".bmb_links.json").read_text())
    real = {str(Path(r).parent) for r in links} - {"."}
    real |= {str(p) for r in list(real) for p in Path(r).parents if str(p) != "."}

    def fill(rel: str) -> None:
        here = src / rel if rel else src
        if here.is_dir():
            for name in sorted(os.listdir(here)):
                if not rel and name.startswith(".bmb"):
                    continue
                r = f"{rel}/{name}" if rel else name
                if r in real:
                    (dst / r).mkdir(parents=True, exist_ok=True)
                    fill(r)
                elif not (dst / r).exists():
                    (dst / r).symlink_to(src / r)
        for r, sha in links.items():
            if str(Path(r).parent) == (rel or "."):
                (dst / r).parent.mkdir(parents=True, exist_ok=True)
                (dst / r).symlink_to(shared / sha[:16] / "blob")

    for r in sorted(real, key=len):
        (dst / r).mkdir(parents=True, exist_ok=True)
    fill("")


def main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    run = Path(spec["run_dir"])
    if spec.get("mirror"):
        mirror_inputs(spec["mirror"])
    env = {k: os.environ[k] for k in PASS_ENV if k in os.environ}
    env.update(spec["env"])
    stdin_path = run / "in.bin"
    out_path, err_path = run / "out.bin", run / "err.log"
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    out = os.fdopen(fd, "wb")
    stdin = open(stdin_path, "rb") if spec.get("stdin") else subprocess.DEVNULL
    proc = subprocess.Popen([sys.executable, "-m", f"bmb_eval.sandbox_entry.{spec['entry']}"],
                            stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                            cwd=spec.get("cwd", "/tmp"), preexec_fn=_child_setup(spec), close_fds=True)
    if stdin is not subprocess.DEVNULL:
        stdin.close()
    cap = int(spec["stdout_cap"])
    n, overflow, err = [0], threading.Event(), bytearray()
    killed = threading.Event()

    def kill() -> None:
        if not killed.is_set():
            killed.set()
            kill_stage(proc)

    def read_stdout() -> None:
        while chunk := proc.stdout.read(1 << 16):
            if overflow.is_set():
                continue
            if n[0] + len(chunk) > cap:
                overflow.set()
                kill()
                continue
            out.write(chunk)
            n[0] += len(chunk)
        out.close()

    def read_stderr() -> None:
        while chunk := proc.stderr.read(1 << 16):
            err.extend(chunk)
            if len(err) > 2 * LOG_CAP:
                del err[:-LOG_CAP]

    readers = [threading.Thread(target=read_stdout, daemon=True),
               threading.Thread(target=read_stderr, daemon=True)]
    for t in readers:
        t.start()
    done = threading.Event()
    if spec["gpus"]:
        def watch() -> None:
            while not done.wait(0.05):
                if anon_bytes(UID if os.getuid() == 0 else os.getuid()) > int(spec["memory_bytes"]):
                    kill()
                    return
        threading.Thread(target=watch, daemon=True).start()
    timed_out = False
    try:
        proc.wait(timeout=spec["timeout_s"])
    except subprocess.TimeoutExpired:
        timed_out = True
        kill()
        proc.wait()
    done.set()
    if os.getuid() == 0:
        kill_stage(proc)             # nothing the stage started outlives it
    for t in readers:
        t.join(timeout=30)
    err_path.write_bytes(bytes(err[-LOG_CAP:]))
    os.chmod(err_path, 0o600)
    print(json.dumps({"exit_code": proc.returncode, "timed_out": timed_out and not overflow.is_set(),
                      "overflow": overflow.is_set(), "bytes": n[0]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

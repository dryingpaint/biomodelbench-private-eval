"""Adversarial bundles for `bmb-eval selftest`.

Each attack bundle tries one thing a hostile submission would. If the attack
works, the bundle raises "ESCAPED: ..." (which shows up as
predict_runtime_error); if the sandbox contains it, the bundle falls back to
a valid constant prediction. Selftest compares each run's exported payload
with the expected one, so it checks the live sandbox, image and runtime
rather than trusting the configuration.
"""
from __future__ import annotations

import io
import tarfile
import textwrap
from dataclasses import dataclass, field

COMMON = textwrap.dedent('''\
    import json, os
    import pandas as pd

    C = json.loads(open(os.environ["BMB_CONTRACT_FILE"]).read())

    def _valid(f):
        if f["type"] == "vector":
            return [_valid({**f, "type": "float"})] * f["length"]
        if f["type"] == "list":
            return {i["name"]: [] for i in f["items"]}
        if f.get("allowed"):
            return f["allowed"][0]
        if f["type"] == "string":
            return "x"
        if f["type"] == "bool":
            return False
        lo = f.get("min", 0)
        hi = f.get("max", lo + 1)
        v = (lo + hi) / 2
        return int(v) if f["type"] == "int" else float(v)

    def baseline(df):
        return pd.DataFrame({f["name"]: [_valid(f) for _ in range(len(df))] for f in C["outputs"]})

    def _inputs_path():
        return os.environ.get("BMB_INPUTS") or os.path.join(os.environ["BMB_INPUTS_DIR"], "samples.parquet")
    ''')


@dataclass
class Attack:
    name: str
    what: str
    predict: str                       # body appended after COMMON
    expect: str                        # "ok" or an error code
    container_only: bool = False       # meaningless (or unsafe) on the local backend
    limits: dict = field(default_factory=dict)
    gpu_only: bool = False             # run only for tasks that use GPUs
    backends: tuple = ()               # run only on these sandbox backends (empty = all)
    also: tuple = ()                   # other outcomes that are equally safe


ATTACKS = [
    Attack("baseline", "valid constant predictions are scored", "predict = baseline\n", "ok"),

    Attack("network", "open outbound TCP and DNS", textwrap.dedent('''\
        import socket
        def load_model():
            for host, port in [("1.1.1.1", 443), ("8.8.8.8", 53), ("example.com", 80)]:
                try:
                    socket.create_connection((host, port), timeout=3).close()
                except OSError:
                    continue
                raise RuntimeError(f"ESCAPED: connected to {host}:{port}")
        predict = baseline
        '''), "ok", container_only=True),

    Attack("find_answers", "look for answers, other mounted files, engine sockets",
           textwrap.dedent('''\
        import sys, time
        # What the sandbox is allowed to have mounted; anything else is an escape.
        ALLOWED = ("/bundle", "/data/inputs", "/contract.json", "/tmp", "/dev", "/proc", "/sys",
                   "/etc/hosts", "/etc/hostname", "/etc/resolv.conf", "/run/.containerenv",
                   "/etc/passwd", "/etc/group", "/run/secrets",
                   "/usr/local/nvidia")   # the GPU driver (read-only), on GPU machines

        def _gpu_driver_mount(target, opts):
            # Podman's GPU spec mounts the host's driver files at their system paths: read-only
            # NVIDIA/CUDA libraries, tools and firmware. Anything else there is still an escape.
            # Modal's GPU containers mount NVIDIA's Vulkan/EGL vendor files the same way.
            name = os.path.basename(target)
            return ("ro" in opts.split(",")
                    and target.startswith(("/usr/lib/", "/usr/lib64/", "/lib/", "/lib64/", "/usr/bin/",
                                           "/etc/OpenCL/vendors/", "/etc/vulkan/icd.d/",
                                           "/usr/share/glvnd/egl_vendor.d/",
                                           "/usr/share/egl/egl_external_platform.d/"))
                    and (name.startswith(("libcuda", "libnvidia", "nvidia")) or "/firmware/nvidia/" in target
                         or (name.endswith(".json") and "nvidia" in name)))

        def _modal_volumes(inputs_dir):
            # Modal: every volume of the sandbox is under /__modal/volumes, reached through a
            # symlink at its mount path. Each must be one this stage was given: the bundle,
            # the inputs, or a shared file linked from the inputs.
            root = "/__modal/volumes"
            if not os.path.isdir(root):
                return []
            given = {os.path.realpath(os.environ["BMB_BUNDLE"]), inputs_dir}
            for d in (os.environ.get("BMB_INPUTS_DIR"), os.path.dirname(_inputs_path())):
                for base, dirs, files in os.walk(d or "/nonexistent"):
                    for n in dirs + files:
                        p = os.path.join(base, n)
                        if os.path.islink(p):
                            given.add(os.path.realpath(p))
                    if base.count(os.sep) > 8:
                        break
            hits = []
            for v in os.listdir(root):
                path = os.path.join(root, v)
                if not any(g == path or g.startswith(path + os.sep) for g in given):
                    hits.append(f"volume:{v}")
            return hits

        def load_model():
            hits = [k for k in os.environ if "ANSWER" in k.upper() or "TOKEN" in k.upper()]
            for sock in ["/var/run/docker.sock", "/run/docker.sock", "/run/podman/podman.sock"]:
                if os.path.exists(sock):
                    hits.append(sock)
            # 1. Mounts: the precise answer to "what host files can this code reach?"
            #    Memory-only and kernel filesystems are private to the container.
            private = {"tmpfs", "proc", "sysfs", "devpts", "mqueue", "cgroup", "cgroup2",
                       "devtmpfs", "shm", "overlay"}
            with open("/proc/self/mountinfo") as fh:
                for line in fh:
                    fields = line.split()
                    target, fstype = fields[4], fields[fields.index("-") + 1]
                    if (fstype in private or target == "/" or target.startswith(ALLOWED)
                            or _gpu_driver_mount(target, fields[5])):
                        continue
                    if target.startswith("/__modal/"):   # Modal's runtime; its volumes: below
                        continue
                    hits.append(f"mount:{target} ({fstype})")
            # 2. Data files outside the inputs and the image's own libraries. (The real
            #    folder of the samples table: on Modal the inputs are behind symlinks.)
            inputs_dir = os.path.dirname(os.path.realpath(_inputs_path()))
            hits += _modal_volumes(inputs_dir)
            skip = (inputs_dir + os.sep, sys.prefix + os.sep, "/usr/")
            t0, seen = time.time(), 0
            for root, dirs, files in os.walk("/"):
                dirs[:] = [d for d in dirs if os.path.join(root, d) not in ("/proc", "/sys", "/dev")]
                for f in files:
                    seen += 1
                    p = os.path.realpath(os.path.join(root, f))
                    if f.endswith((".parquet", ".csv", ".h5ad")) and not p.startswith(skip):
                        hits.append(p)
                if seen > 300_000 or time.time() - t0 > 60:
                    break
            if hits:
                raise RuntimeError(f"ESCAPED: found {hits[:5]}")
        predict = baseline
        '''), "ok", container_only=True),

    Attack("write_host", "write into the bundle, data and system paths", textwrap.dedent('''\
        def load_model():
            targets = [os.path.join(os.environ["BMB_BUNDLE"], "x"),
                       os.path.join(os.path.dirname(_inputs_path()), "x"),
                       "/x", "/etc/x", "/usr/x"]
            for p in targets:
                try:
                    with open(p, "w") as fh:
                        fh.write("x")
                except OSError:
                    continue
                raise RuntimeError(f"ESCAPED: wrote {p}")
            if os.getuid() == 0:
                raise RuntimeError("ESCAPED: running as root")
        predict = baseline
        '''), "ok", container_only=True),

    Attack("fork_bomb", "spawn more processes than the pid limit", textwrap.dedent('''\
        import time
        def load_model():
            # Raw fork(): fails cleanly with EAGAIN at the pid limit.
            kids = []
            try:
                for _ in range(200):
                    pid = os.fork()
                    if pid == 0:
                        time.sleep(30)
                        os._exit(0)
                    kids.append(pid)
            except OSError:
                pass
            for pid in kids:
                try:
                    os.kill(pid, 9)
                    os.waitpid(pid, 0)
                except OSError:
                    pass
            if len(kids) >= 200:
                raise RuntimeError(f"ESCAPED: spawned {len(kids)} processes")
        predict = baseline
        '''), "ok", container_only=True, limits={"pids": 64, "predict_timeout_s": 120}),

    Attack("memory", "allocate past the memory limit", textwrap.dedent('''\
        def load_model():
            hog = []
            for _ in range(64):
                hog.append(bytearray(64 << 20))  # 4 GiB total
        predict = baseline
        '''), "predict_resource_limit", container_only=True, limits={"memory": "512m"}),

    Attack("memory_gpu", "allocate past the memory limit in a GPU stage", textwrap.dedent('''\
        def load_model():
            hog = []
            for _ in range(64):
                hog.append(bytearray(64 << 20))  # 4 GiB total
        predict = baseline
        '''), "predict_resource_limit", container_only=True, gpu_only=True, limits={"memory": "512m"}),

    Attack("timeout", "run past the time limit", textwrap.dedent('''\
        import time
        def predict(df):
            time.sleep(3600)
        '''), "predict_timeout", limits={"predict_timeout_s": 5}),

    Attack("flood_output", "write more than the predictions cap", textwrap.dedent('''\
        def predict(df):
            # The original stdout is a dup'd fd; spray every plausible one.
            junk = b"x" * (1 << 20)
            for _ in range(8):
                for fd in range(3, 16):
                    try:
                        os.write(fd, junk)
                    except OSError:
                        pass
            return baseline(df)
        '''), "predictions_too_large", limits={"max_predictions_mb": 1},
        # Podman < 4 can deliver the flood truncated: then it fails validation
        # instead. Either way nothing past the cap is scored.
        also=("predictions_invalid",)),

    Attack("leak_in_error", "put private input values in an exception message", textwrap.dedent('''\
        def predict(df):
            raise ValueError("LEAK " + " ".join(map(str, df.iloc[0].tolist())))
        '''), "predict_runtime_error"),

    Attack("parser_attack", "hand the validator a malformed predictions file", textwrap.dedent('''\
        def predict(df):
            for fd in range(3, 16):
                try:
                    os.write(fd, b"PAR1" + os.urandom(4096) + b"PAR1")
                except OSError:
                    pass
            os._exit(0)
        '''), "predictions_invalid"),

    Attack("wrong_schema", "return the wrong output columns", textwrap.dedent('''\
        def predict(df):
            return pd.DataFrame({f["name"]: ["not a number"] * len(df) for f in C["outputs"]})
        '''), "predictions_invalid"),

    Attack("symlink_tar", "smuggle a symlink to the answers in the archive", "", "bundle_unsafe"),

    Attack("gpu_visible", "the task's GPU is usable inside the sandbox", textwrap.dedent('''\
        def load_model():
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("no GPU inside the sandbox")
            torch.ones(1, device="cuda").sum().item()
        predict = baseline
        '''), "ok", container_only=True, gpu_only=True),

    Attack("kernel_surface", "reach kernel interfaces used in escapes (user namespaces, mount, "
           "ptrace, bpf, userfaultfd, io_uring, keyrings, modules)", textwrap.dedent('''\
        import ctypes, errno, platform
        # x86_64 syscall numbers
        CALLS = {"unshare(NEWUSER)": (272, 0x10000000), "mount": (165, 0), "ptrace": (101, 0),
                 "bpf": (321, 0), "userfaultfd": (323, 0), "perf_event_open": (298, 0),
                 "io_uring_setup": (425, 0), "keyctl": (250, 0), "add_key": (248, 0),
                 "init_module": (175, 0), "kexec_load": (246, 0)}
        BLOCKED = {errno.EPERM, errno.EACCES, errno.ENOSYS}

        def load_model():
            if platform.machine() != "x86_64":
                return
            libc = ctypes.CDLL(None, use_errno=True)
            reachable = []
            for name, (nr, arg) in CALLS.items():
                ctypes.set_errno(0)
                r = libc.syscall(nr, ctypes.c_long(arg), 0, 0, 0, 0)
                if r >= 0 or ctypes.get_errno() not in BLOCKED:
                    reachable.append(name)
            if reachable:
                raise RuntimeError(f"ESCAPED: reachable {reachable}")
        predict = baseline
        '''), "ok", container_only=True, backends=("bwrap",)),
]


def bundle_bytes(a: Attack) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        src = (COMMON + "\n" + a.predict).encode()
        ti = tarfile.TarInfo("predict.py")
        ti.size = len(src)
        tar.addfile(ti, io.BytesIO(src))
        if a.name == "symlink_tar":
            link = tarfile.TarInfo("weights")
            link.type = tarfile.SYMTYPE
            link.linkname = "/data/answers.parquet"
            tar.addfile(link)
    return buf.getvalue()

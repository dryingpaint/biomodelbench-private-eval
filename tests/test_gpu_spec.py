"""The GPU device spec for Podman's read-only sandbox (bmb_eval.gpu_spec), and the
self-test's view of the driver files that spec mounts."""
from __future__ import annotations

import os
from pathlib import Path

from bmb_eval.gpu_spec import readonly_spec
from bmb_eval.redteam import ATTACKS

LIB = "/usr/lib/x86_64-linux-gnu"


def _toolkit_spec() -> dict:
    """The shape `nvidia-ctk cdi generate` (1.20) writes, trimmed."""
    hook = lambda *args: {"hookName": "createContainer", "path": "/usr/bin/nvidia-cdi-hook",
                          "args": ["nvidia-cdi-hook", *args]}
    return {
        "cdiVersion": "0.7.0", "kind": "nvidia.com/gpu",
        "devices": [{"name": "all", "containerEdits": {
            "deviceNodes": [{"path": "/dev/nvidia0"}],
            "additionalGids": [44, 992],
            "hooks": [hook("create-symlinks", "--link", "../card0::/dev/dri/by-path/pci-card")]}}],
        "containerEdits": {
            "mounts": [{"hostPath": f"{LIB}/libcuda.so.580.1", "containerPath": f"{LIB}/libcuda.so.580.1",
                        "options": ["ro", "nosuid", "nodev", "rbind", "rprivate"]}],
            "additionalGids": [44],
            "hooks": [hook("create-symlinks", "--link", "libcuda.so.580.1::" + f"{LIB}/libcuda.so.1",
                           "--link", "libcuda.so.1::" + f"{LIB}/libcuda.so",
                           "--link", "libgone.so.1::" + f"{LIB}/libgone.so"),
                      hook("update-ldcache", "--folder", LIB),
                      hook("enable-cuda-compat", "--host-driver-version=580.1")]},
    }


def test_readonly_spec_replaces_writing_hooks_with_readonly_mounts(tmp_path):
    lib = tmp_path / LIB.lstrip("/")
    lib.mkdir(parents=True)
    (lib / "libcuda.so.580.1").write_bytes(b"\x7fELF")
    os.symlink("libcuda.so.580.1", lib / "libcuda.so.1")
    original = _toolkit_spec()
    spec = readonly_spec(original, host_root=tmp_path)
    assert original["cdiVersion"] == "0.7.0"                       # the input is left alone
    assert spec["cdiVersion"] == "0.6.0"                           # Podman 4 reads 0.6
    edits = spec["containerEdits"]
    assert "additionalGids" not in edits and "additionalGids" not in spec["devices"][0]["containerEdits"]
    assert [h["args"][1] for h in edits["hooks"]] == ["enable-cuda-compat"]   # nothing that writes
    assert spec["devices"][0]["containerEdits"]["hooks"] == []
    mounts = {m["containerPath"]: m for m in edits["mounts"]}
    # Each symlink becomes a read-only mount of the real file; a link to nothing is dropped.
    for link in (f"{LIB}/libcuda.so.1", f"{LIB}/libcuda.so"):
        assert mounts[link]["hostPath"] == f"{LIB}/libcuda.so.580.1"
        assert "ro" in mounts[link]["options"]
    assert f"{LIB}/libgone.so" not in mounts
    assert len(edits["mounts"]) == 3


def test_selftest_accepts_readonly_driver_mounts_and_nothing_else():
    src = next(a for a in ATTACKS if a.name == "find_answers").predict
    ns = {"os": os, "baseline": None, "_inputs_path": lambda: "/data/inputs/samples.parquet"}
    exec(src, ns)
    ok = ns["_gpu_driver_mount"]
    assert ok(f"{LIB}/libcuda.so.1", "ro,nosuid,nodev,relatime")
    assert ok("/usr/bin/nvidia-smi", "ro,nosuid")
    assert ok("/usr/lib/firmware/nvidia/580.1/gsp_ga10x.bin", "ro")
    assert ok("/etc/OpenCL/vendors/nvidia.icd", "ro")
    assert not ok(f"{LIB}/libcuda.so.1", "rw,nosuid")             # writable: an escape
    assert not ok("/data/answers.parquet", "ro")                  # not a driver file
    assert not ok(f"{LIB}/libssl.so.3", "ro")
    assert not ok("/home/me/nvidia-notes.csv", "ro")              # outside the system directories


def test_connect_names_an_nvidia_driver_too_old_for_the_sandbox(monkeypatch):
    from types import SimpleNamespace
    from bmb_eval import cli
    lim = SimpleNamespace(gpus=1, gpu_memory_gb=16)
    monkeypatch.setattr(cli, "gpus_here", lambda: [24])
    monkeypatch.setattr(cli, "podman_version", lambda: (4, 9))
    monkeypatch.setattr(cli, "nvidia_driver", lambda: ("570.86.15", 570))
    assert any("driver 580 or newer" in p and "570.86.15" in p for p in cli.gpu_problems(lim))
    monkeypatch.setattr(cli, "nvidia_driver", lambda: ("580.178.04", 580))
    assert cli.gpu_problems(lim) == []


def test_min_driver_matches_the_cuda_build_in_the_lock():
    from bmb_eval import cli
    lock = (Path(__file__).resolve().parents[1] / "sandbox" / "requirements.lock").read_text()
    if "-cu13==" in lock:                  # CUDA 13 wheels need driver 580+
        assert cli.MIN_NVIDIA_DRIVER >= 580
    if "-cu12==" in lock:
        assert cli.MIN_NVIDIA_DRIVER >= 525


def test_installer_stops_on_arm_before_installing_anything(tmp_path):
    import subprocess
    repo = Path(__file__).resolve().parents[1]
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "uname").write_text('#!/bin/sh\ncase "$1" in -s) echo Linux;; -m) echo aarch64;; *) echo Linux;; esac\n')
    (fake / "sudo").write_text("#!/bin/sh\necho SUDO-CALLED >&2; exit 99\n")
    for f in ("uname", "sudo"):
        (fake / f).chmod(0o755)
    r = subprocess.run(["sh", str(repo / "install.sh")], capture_output=True, text=True,
                       env={"PATH": f"{fake}:/usr/bin:/bin", "HOME": str(tmp_path),
                            "BMB_HUB": "https://hub.example"})
    assert r.returncode == 1
    assert "x86_64" in r.stderr and "aarch64" in r.stderr and "SUDO-CALLED" not in r.stderr


def test_a_podman_exported_image_gets_the_short_name_docker_needs(monkeypatch):
    # Docker loads the hub's image as localhost/bmb-eval-sandbox:latest and doesn't find it
    # under the short name: the node saw it as missing (downloaded it again) and unstamped.
    from bmb_eval import cli
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    cli._tag_loaded("docker", "Loaded image: localhost/bmb-eval-sandbox:latest\n", "bmb-eval-sandbox:latest")
    assert calls == [["docker", "tag", "localhost/bmb-eval-sandbox:latest", "bmb-eval-sandbox:latest"]]
    calls.clear()
    cli._tag_loaded("podman", "Loaded image(s): bmb-eval-sandbox:latest\n", "bmb-eval-sandbox:latest")
    cli._tag_loaded("docker", "Loaded image ID: sha256:abc\n", "bmb-eval-sandbox:latest")
    assert calls == []                      # already the right name, or nothing named to tag

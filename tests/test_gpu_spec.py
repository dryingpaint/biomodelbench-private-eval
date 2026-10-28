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

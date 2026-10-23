"""A release is one tag naming the three downloads. These keep that true: the
sandbox image's inputs are pinned, the code tarball is reproducible, the hub
publishes the release with the sha256 of each download, and installed code
knows which release it is."""
from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tarfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from bmb_eval import hub as hub_mod
from bmb_eval import release
from bmb_eval.sandbox import SANDBOX_FILES, sandbox_stamp

REPO = Path(__file__).resolve().parents[1]


def test_sandbox_lock_pins_every_requirement_with_hashes():
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    lock = (REPO / "sandbox" / "requirements.lock").read_text()
    pins = {}
    for block in re.split(r"\n(?=\S)", lock):       # one package, with its hashes, per block
        if m := re.match(r"([A-Za-z0-9_.-]+)==([^\s;\\]+)", block):
            assert "--hash=sha256:" in block, f"{m.group(1)} has no hash"
            pins[canonicalize_name(m.group(1))] = m.group(2)
    assert len(pins) > 30                           # torch alone brings a dozen CUDA packages
    wanted = [Requirement(line.split("#")[0].strip())
              for line in (REPO / "sandbox" / "requirements.txt").read_text().splitlines()
              if line.split("#")[0].strip()]
    assert len(wanted) >= 16
    for r in wanted:
        name = canonicalize_name(r.name)
        assert name in pins, f"{r.name} isn't in the lock: run sandbox/lock.sh"
        assert r.specifier.contains(pins[name], prereleases=True), \
            f"{r}: the lock pins {pins[name]}; run sandbox/lock.sh"


def test_sandbox_image_inputs_are_pinned_and_stamped():
    df = (REPO / "sandbox" / "Dockerfile").read_text()
    assert re.search(r"^FROM \S+@sha256:[0-9a-f]{64}$", df, re.M), "pin the base image by digest"
    assert "snapshot.debian.org" in df, "pin Debian packages by snapshot date"
    assert re.search(r"pip install .*--require-hashes .*requirements\.lock", df)
    # The stamp covers everything that decides the image's content, on the hub and the node alike.
    assert {"sandbox/Dockerfile", "sandbox/requirements.txt", "sandbox/requirements.lock"} <= set(SANDBOX_FILES)
    assert all((REPO / f).is_file() for f in SANDBOX_FILES)
    sh = (REPO / "deploy" / "gcp" / "hub-deploy.sh").read_text()
    assert re.search(r'^SANDBOX_FILES="([^"]*)"', sh, re.M).group(1).split() == list(SANDBOX_FILES)


def test_code_tarball_is_reproducible_and_names_its_release(tmp_path):
    rel = {"version": "v9.9.9", "commit": "a" * 40}
    a = hub_mod.code_tarball(REPO, rel)
    assert a == hub_mod.code_tarball(REPO, rel)
    assert a != hub_mod.code_tarball(REPO, {**rel, "version": "v9.9.8"})
    tar = tarfile.open(fileobj=io.BytesIO(a))
    assert json.loads(tar.extractfile("release.json").read()) == rel
    assert "sandbox/requirements.lock" in tar.getnames()
    assert all(ti.mtime == 0 and ti.uid == 0 and ti.uname == "" for ti in tar.getmembers())
    # Installed from the tarball, without git, the code knows its release.
    tar.extract("release.json", tmp_path, filter="data")
    assert release.info(tmp_path) == rel
    assert release.describe(tmp_path) == "v9.9.9 (aaaaaaa)"


def test_release_info_falls_back_to_git_then_unknown(tmp_path):
    assert release.info(tmp_path) == release.UNKNOWN
    (tmp_path / "release.json").write_text("not json")
    assert release.info(tmp_path) == release.UNKNOWN
    here = release.from_git(REPO)
    if here:                                        # a checkout, not a tarball install
        assert re.fullmatch(r"[0-9a-f]{40}", here["commit"])
        assert here["version"].startswith(("v", "untagged"))


@pytest.fixture
def hub(tmp_path):
    store = hub_mod.HubStore(tmp_path / "hub_state")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), None)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    blobs = hub_mod.DiskBlobs(tmp_path / "hub_state" / "blobs", url)
    httpd.RequestHandlerClass = hub_mod.make_handler(store, blobs, url, REPO, 64)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield store, url
    httpd.shutdown()


def _get(url: str) -> bytes:
    return urllib.request.urlopen(url, timeout=10).read()


def test_hub_publishes_the_release_with_the_hash_of_each_download(hub):
    store, url = hub
    rel = json.loads(_get(f"{url}/release"))
    assert {"version": rel["version"], "commit": rel["commit"]} == release.info(REPO)
    assert rel["sandbox_stamp"] == sandbox_stamp(REPO)
    for name in ("install.sh", "bmb-eval.tar.gz"):
        body = _get(f"{url}/{name}")
        assert rel["files"][name] == {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
    assert rel["files"]["sandbox-image.tar"] is None            # none published on this hub
    sums = _get(f"{url}/SHA256SUMS").decode()
    assert f"{rel['files']['install.sh']['sha256']}  install.sh\n" in sums
    assert f"{rel['files']['bmb-eval.tar.gz']['sha256']}  bmb-eval.tar.gz\n" in sums
    assert "sandbox-image.tar" not in sums
    # hub-deploy.sh publishes the image with its checksum and the stamp it was built with.
    state = store.state
    (state / "sandbox-image.tar").write_bytes(b"an image")
    sha = hashlib.sha256(b"an image").hexdigest()
    (state / "sandbox-image.tar.sha256").write_text(f"{sha}  sandbox-image.tar\n")
    (state / "sandbox-image.stamp").write_text("0123456789abcdef\n")
    img = json.loads(_get(f"{url}/release"))["files"]["sandbox-image.tar"]
    assert img == {"sha256": sha, "bytes": 8, "stamp": "0123456789abcdef", "current": False}
    (state / "sandbox-image.stamp").write_text(sandbox_stamp(REPO) + "\n")
    assert json.loads(_get(f"{url}/release"))["files"]["sandbox-image.tar"]["current"] is True
    assert f"{sha}  sandbox-image.tar\n" in _get(f"{url}/SHA256SUMS").decode()


def test_build_image_skips_the_download_when_the_image_is_current(tmp_path, monkeypatch, capsys):
    # Updates re-run `bmb-eval build-image --from <hub>`; a current image must not be
    # downloaded again (6.8 GB, minutes).
    from argparse import Namespace
    from bmb_eval import cli, sandbox
    (tmp_path / "config.yaml").write_text("sandbox:\n  backend: podman\n")
    monkeypatch.setattr(sandbox.ContainerSandbox, "image_id", lambda self: "sha256:deadbeef")
    monkeypatch.setattr(sandbox.ContainerSandbox, "image_label", lambda self, key: sandbox_stamp(REPO))

    def no_download(*a, **k):
        raise AssertionError("downloaded")
    monkeypatch.setattr(cli.urllib.request, "urlopen", no_download)
    cli.cmd_build_image(Namespace(config=str(tmp_path / "config.yaml"), from_url="http://hub/sandbox-image.tar"))
    assert "already current" in capsys.readouterr().out
    # Once the code moves on, the download happens again.
    monkeypatch.setattr(sandbox.ContainerSandbox, "image_label", lambda self, key: "0123456789abcdef")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("built")))
    with pytest.raises(AssertionError, match="downloaded|built"):
        cli.cmd_build_image(Namespace(config=str(tmp_path / "config.yaml"), from_url="http://hub/sandbox-image.tar"))

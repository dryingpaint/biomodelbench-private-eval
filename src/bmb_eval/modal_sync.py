"""`bmb-eval modal-sync`: put a Modal node's prepared task data on Modal Volumes.

Run by connect and update (after the data check, before the self-test), or by
hand. For each task it uploads what changed into the task's volumes (see
modal_backend.py for the layout), then writes:

- state/modal/mounts.json: which host path is on which volume, for ModalSandbox;
- MANIFEST.json, locally and on the <prefix>-meta volume: each task's tree
  hashes, exactly as approval.fingerprint computes them, and every uploaded
  file's size and sha256 per volume.

The approval fingerprint holds the manifest's hash, so `bmb-eval approve`
approves the data as uploaded, and the node stops accepting work when the
manifest on Modal differs from the approved one (drift()), or when a volume's
files differ from the manifest (verify(), at start and hourly).

Large files (SHARED_MIN_BYTES and up) that both halves of a task see (pass-
through files, such as the reference genome every genomic task links in) go
once into a volume of their own, <prefix>-sh-<sha16>, which only those tasks'
predict stages mount.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from . import approval, audit, modal_backend
from .modal_backend import modal_dir, vol_name

SHARED_MIN_BYTES = 256 << 20
MANIFEST = "MANIFEST.json"


def _is_file(entry) -> bool:
    """A volume listing entry that is a file (modal's FileEntryType is an IntEnum)."""
    return getattr(getattr(entry, "type", None), "name", "FILE") == "FILE"


def _files(src: Path) -> dict[str, Path]:
    """relative path -> file, for a folder or a single file."""
    if src.is_file():
        return {src.name: src}
    return {p.relative_to(src).as_posix(): p for p in sorted(src.rglob("*")) if p.is_file()}


def plan(cfg) -> tuple[dict, list[dict], dict]:
    """(volume -> {remote path: local file}, mount table entries, shared sha -> local file)."""
    pre = cfg.modal.volume_prefix
    vols: dict[str, dict[str, Path]] = {}
    entries: list[dict] = []
    shared: dict[str, Path] = {}
    for tid, t in sorted(cfg.tasks.items()):
        halves = []
        if t.inputs_dir:
            halves.append(("inputs_validation" if t.test_inputs_dir else "inputs",
                           "val" if t.test_inputs_dir else "in", t.inputs_dir))
        if t.test_inputs_dir:
            halves.append(("inputs_test", "test", t.test_inputs_dir))
        # Large files in every half of this task (or its only half) are stored once.
        seen: dict[str, list] = {}
        for kind, _, d in halves:
            for rel, f in _files(d).items():
                if f.stat().st_size >= SHARED_MIN_BYTES:
                    seen.setdefault(rel, []).append((kind, approval._file_hash(f), f))
        dedup = {rel: v[0][1] for rel, v in seen.items()
                 if len(v) == len(halves) and len({h for _, h, _ in v}) == 1}
        for rel, sha in dedup.items():
            shared[sha] = seen[rel][0][2]
        for kind, suffix, d in halves:
            name = vol_name(pre, tid, suffix)
            files = {rel: f for rel, f in _files(d).items() if rel not in dedup}
            vols[name] = files
            entries.append({"host": str(d.resolve()), "volume": name, "kind": kind, "task": tid,
                            "path": "", "is_dir": True, "links": dedup})
        if not t.inputs_dir:            # a bare samples table (no inputs folder)
            name = vol_name(pre, tid, "in")
            vols[name] = {"samples.parquet": t.inputs}
            entries.append({"host": str(t.inputs.resolve()), "volume": name, "kind": "inputs", "task": tid,
                            "path": "samples.parquet", "is_dir": False})
        ans = vol_name(pre, tid, "ans")
        vols[ans] = {"answers.parquet": t.answers}
        entries.append({"host": str(t.answers.resolve()), "volume": ans, "kind": "answers", "task": tid,
                        "path": "answers.parquet", "is_dir": False})
        if t.answers_dir:
            vols[ans].update({f"answers_files/{r}": f for r, f in _files(t.answers_dir).items()})
            entries.append({"host": str(t.answers_dir.resolve()), "volume": ans, "kind": "answers",
                            "task": tid, "path": "answers_files", "is_dir": True})
        task_vol = vol_name(pre, tid, "task")
        vols[task_vol] = {r: f for r, f in _files(t.dir).items() if "__pycache__" not in Path(r).parts}
        entries.append({"host": str(t.dir.resolve()), "volume": task_vol, "kind": "task", "task": tid,
                        "path": "", "is_dir": True})
    for sha, f in shared.items():
        vols[vol_name(pre, "sh", sha[:16])] = {"blob": f}
    return vols, entries, shared


def build_manifest(cfg, vols: dict) -> dict:
    fp = approval.fingerprint_tasks(cfg)
    return {
        "environment": cfg.modal.environment,
        "tasks": fp,
        "volumes": {v: {rel: [f.stat().st_size, approval._file_hash(f)] for rel, f in sorted(files.items())}
                    for v, files in sorted(vols.items())},
    }


def manifest_bytes(m: dict) -> bytes:
    return (json.dumps(m, indent=1, sort_keys=True) + "\n").encode()


def local_manifest_hash(cfg) -> str:
    p = modal_dir(cfg) / MANIFEST
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"


def sync(cfg, modal_mod=None, say=print) -> dict:
    """Upload what changed; write the mount table and the manifest (local and on Modal)."""
    m = modal_mod or modal_backend.import_modal(cfg.modal)
    env = cfg.modal.environment
    vols, entries, shared = plan(cfg)
    manifest = build_manifest(cfg, vols)
    sent = 0
    for name, files in sorted(vols.items()):
        vol = m.Volume.from_name(name, environment_name=env, create_if_missing=True)
        have = {}
        try:
            have = {e.path.strip("/"): e.size for e in vol.listdir("/", recursive=True)
                    if _is_file(e)}
        except Exception:
            pass
        want = manifest["volumes"][name]
        # The size check catches most changes; a recorded hash catches same-size edits.
        uploaded = _uploaded(cfg).get(name, {})
        todo = {rel: f for rel, f in files.items()
                if have.get(rel) != want[rel][0] or uploaded.get(rel) != want[rel][1]}
        if todo:
            say(f"  {name}: uploading {len(todo)} file(s), {sum(f.stat().st_size for f in todo.values()) >> 20} MB")
            with vol.batch_upload(force=True) as up:
                for rel, f in todo.items():
                    up.put_file(str(f), "/" + rel)
            sent += len(todo)
        links = next((e["links"] for e in entries if e["volume"] == name and e.get("links")), None)
        if links is not None:
            with vol.batch_upload(force=True) as up:
                up.put_file(io.BytesIO(json.dumps(links, sort_keys=True).encode()), "/.bmb_links.json")
        for rel in sorted(set(have) - set(want) - {".bmb_links.json"}):
            vol.remove_file("/" + rel)
        _record_uploaded(cfg, name, {rel: h for rel, (_, h) in want.items()})
    body = manifest_bytes(manifest)
    meta = m.Volume.from_name(vol_name(cfg.modal.volume_prefix, "meta"), environment_name=env,
                              create_if_missing=True)
    with meta.batch_upload(force=True) as up:
        up.put_file(io.BytesIO(body), "/" + MANIFEST)
    (modal_dir(cfg) / MANIFEST).write_bytes(body)
    (modal_dir(cfg) / "mounts.json").write_text(json.dumps({"entries": entries}, indent=1) + "\n")
    audit.record(cfg.state_dir, "modal_sync", volumes=len(vols), files_sent=sent, shared=len(shared),
                 manifest=hashlib.sha256(body).hexdigest())
    return manifest


def _uploaded(cfg) -> dict:
    try:
        return json.loads((modal_dir(cfg) / "uploaded.json").read_text())
    except (OSError, ValueError):
        return {}


def _record_uploaded(cfg, volume: str, files: dict) -> None:
    rec = _uploaded(cfg)
    rec[volume] = files
    (modal_dir(cfg) / "uploaded.json").write_text(json.dumps(rec))


def remote_manifest(cfg, modal_mod=None) -> bytes:
    m = modal_mod or modal_backend.import_modal(cfg.modal)
    meta = m.Volume.from_name(vol_name(cfg.modal.volume_prefix, "meta"), environment_name=cfg.modal.environment)
    return b"".join(meta.read_file("/" + MANIFEST))


def drift(cfg, modal_mod=None) -> list[str]:
    """Why the data on Modal isn't what the operator approved; [] when it is."""
    try:
        approved = json.loads(approval.lock_path(cfg).read_text())
    except (OSError, ValueError):
        return ["no approved.lock.json"]
    want = ((approved.get("sandbox") or {}).get("modal") or {}).get("manifest")
    try:
        body = remote_manifest(cfg, modal_mod)
    except Exception as e:
        return [f"can't read the manifest on Modal ({type(e).__name__})"]
    out = []
    if hashlib.sha256(body).hexdigest() != want:
        out.append("the manifest on Modal differs from the approved one (run `bmb-eval modal-sync`, "
                   "then `bmb-eval approve`)")
    try:
        tasks = json.loads(body)["tasks"]
    except (ValueError, KeyError):
        return out + ["the manifest on Modal is unreadable"]
    for tid, fp in (approved.get("tasks") or {}).items():
        if tasks.get(tid) != fp:
            out.append(f"task {tid}: the data on Modal isn't the approved data")
    return out


def verify(cfg, modal_mod=None) -> list[str]:
    """Each volume's files (paths and sizes) against the local manifest: catches a
    volume changed on Modal behind the manifest's back."""
    m = modal_mod or modal_backend.import_modal(cfg.modal)
    try:
        manifest = json.loads((modal_dir(cfg) / MANIFEST).read_text())
    except (OSError, ValueError):
        return ["no local manifest (run `bmb-eval modal-sync`)"]
    out = []
    for name, files in manifest["volumes"].items():
        try:
            vol = m.Volume.from_name(name, environment_name=cfg.modal.environment)
            have = {e.path.strip("/"): e.size for e in vol.listdir("/", recursive=True)
                    if _is_file(e)}
        except Exception as e:
            out.append(f"{name}: can't list it ({type(e).__name__})")
            continue
        have.pop(".bmb_links.json", None)
        want = {rel: v[0] for rel, v in files.items()}
        if have != want:
            extra, missing = sorted(set(have) - set(want)), sorted(set(want) - set(have))
            changed = sorted(r for r in set(have) & set(want) if have[r] != want[r])
            out.append(f"{name}: {len(extra)} unexpected, {len(missing)} missing, {len(changed)} changed file(s)")
    return out

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
# Modal rate-limits its API (ResourceExhaustedError when ~100 volumes are listed at once).
LIST_TRIES = 6                  # attempts per call, with exponential back-off and jitter
LIST_BACKOFF_S = 1.0            # first back-off; doubles each time, capped at LIST_BACKOFF_MAX_S
LIST_BACKOFF_MAX_S = 30.0
LIST_CONCURRENCY = 4            # volumes listed at once
VERIFY_MAX_AGE_S = 3600         # a volume verified this recently (same manifest) isn't listed again


class ModalCheckError(Exception):
    """Modal couldn't be asked (rate limit, API error, timeout): nothing is known about
    drift, either way. Never treated as drift."""


def _not_found(e: BaseException) -> bool:
    return type(e).__name__ in ("NotFoundError",)


def call_with_retry(fn, what: str, tries: int | None = None):
    """fn() with exponential back-off and jitter on Modal's transient errors (rate limits,
    service errors, timeouts). A missing object is returned as the error itself
    (NotFoundError: real drift, not a transient). Raises ModalCheckError when tries run out."""
    import random
    import time
    tries = tries or LIST_TRIES
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if _not_found(e):
                raise
            last = e
            if attempt + 1 < tries:
                delay = min(LIST_BACKOFF_MAX_S, LIST_BACKOFF_S * 2 ** attempt)
                time.sleep(delay * (0.5 + random.random() / 2))
    raise ModalCheckError(f"{what}: {type(last).__name__}: {str(last)[:160]}") from last


def _listing(m, cfg, name: str) -> dict[str, int]:
    vol = m.Volume.from_name(name, environment_name=cfg.modal.environment)
    return call_with_retry(lambda: {e.path.strip("/"): e.size for e in vol.listdir("/", recursive=True)
                                    if _is_file(e)}, f"listing {name}")


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
        # A listing that fails (rate limit) must not look like an empty volume: that would
        # re-upload everything. Retried, then the sync stops with the error.
        have = call_with_retry(lambda: {e.path.strip("/"): e.size for e in vol.listdir("/", recursive=True)
                                        if _is_file(e)}, f"listing {name}")
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
    return call_with_retry(lambda: b"".join(meta.read_file("/" + MANIFEST)), "reading the manifest")


def drift(cfg, modal_mod=None) -> list[str]:
    """Why the data on Modal isn't what the operator approved; [] when it is.
    Raises ModalCheckError when Modal couldn't be asked."""
    try:
        approved = json.loads(approval.lock_path(cfg).read_text())
    except (OSError, ValueError):
        return ["no approved.lock.json"]
    want = ((approved.get("sandbox") or {}).get("modal") or {}).get("manifest")
    try:
        body = remote_manifest(cfg, modal_mod)
    except ModalCheckError:
        raise
    except Exception as e:
        if _not_found(e):
            return ["the manifest on Modal is missing (run `bmb-eval modal-sync`)"]
        raise ModalCheckError(f"reading the manifest: {type(e).__name__}") from e
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


def _verify_cache_path(cfg) -> Path:
    return modal_dir(cfg) / "verified.json"


def verify(cfg, modal_mod=None, max_age_s: float | None = None) -> list[str]:
    """Each volume's files (paths and sizes) against the local manifest: catches a
    volume changed on Modal behind the manifest's back. [] when every volume matches.

    Volumes are listed a few at a time, each call retried with back-off. A volume that
    matched within max_age_s (for this same manifest) isn't listed again, so a check cut
    short by rate limits resumes where it stopped. Raises ModalCheckError when some
    volume couldn't be listed and no listing showed a difference (real drift wins)."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    m = modal_mod or modal_backend.import_modal(cfg.modal)
    try:
        body = (modal_dir(cfg) / MANIFEST).read_bytes()
        manifest = json.loads(body)
    except (OSError, ValueError):
        return ["no local manifest (run `bmb-eval modal-sync`)"]
    mhash = hashlib.sha256(body).hexdigest()
    max_age_s = VERIFY_MAX_AGE_S if max_age_s is None else max_age_s
    try:
        cache = json.loads(_verify_cache_path(cfg).read_text())
    except (OSError, ValueError):
        cache = {}
    if cache.get("manifest") != mhash:
        cache = {"manifest": mhash, "volumes": {}}
    now = time.time()
    todo = [(n, f) for n, f in sorted(manifest["volumes"].items())
            if now - cache["volumes"].get(n, 0) > max_age_s]

    def check(item):
        name, files = item
        try:
            have = _listing(m, cfg, name)
        except ModalCheckError as e:
            return name, None, str(e)
        except Exception as e:
            if _not_found(e):
                return name, f"{name}: the volume is missing", None
            return name, None, f"{name}: {type(e).__name__}"
        have.pop(".bmb_links.json", None)
        want = {rel: v[0] for rel, v in files.items()}
        if have == want:
            return name, "", None
        extra, missing = sorted(set(have) - set(want)), sorted(set(want) - set(have))
        changed = sorted(r for r in set(have) & set(want) if have[r] != want[r])
        return name, f"{name}: {len(extra)} unexpected, {len(missing)} missing, {len(changed)} changed file(s)", None

    out, failed = [], []
    with ThreadPoolExecutor(max_workers=LIST_CONCURRENCY) as pool:
        for name, problem, err in pool.map(check, todo):
            if err is not None:
                failed.append(err)
            elif problem:
                out.append(problem)
                cache["volumes"].pop(name, None)
            else:
                cache["volumes"][name] = time.time()
    _verify_cache_path(cfg).write_text(json.dumps(cache))
    if out:
        return out
    if failed:
        raise ModalCheckError(f"{len(failed)} of {len(todo)} volume(s) couldn't be listed (first: {failed[0]})")
    return []

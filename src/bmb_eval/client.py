"""Agent-side client. Standard library only, so it drops into any training
environment.

A bundle is a directory. Submitting sends its manifest (file hashes), uploads
only the files the hub doesn't already have (a 20 GB base model goes up once),
then commits.

    from bmb_eval.client import Client
    c = Client("https://eval.provider.org", token)
    c.tasks()                                  # public contracts
    res = c.submit("my_task", "bundle_dir/")   # blocks until released
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .blobs import manifest_for


class Client:
    def __init__(self, url: str, token: str, timeout: float = 60):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _req(self, method: str, path: str, body: dict | None = None, attempts: int = 6) -> dict:
        """JSON request. Temporary failures (hub restarting, network blips) are
        retried with backoff, so long waits survive a hub update."""
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(attempts):
            req = urllib.request.Request(self.url + path, data=data, method=method, headers={
                "Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (502, 503, 504) and attempt < attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                try:
                    return json.loads(e.read())
                except Exception:
                    raise e
            except (urllib.error.URLError, OSError):
                if attempt == attempts - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def tasks(self) -> dict:
        return self._req("GET", "/v1/tasks")

    def contract(self, task: str) -> dict:
        for t in self.tasks()["tasks"]:
            if t["contract"]["id"] == task:
                return t["contract"]
        raise KeyError(task)

    def submit_async(self, task: str, bundle: str | Path, retries: int = 3,
                     final_candidate: bool = False) -> dict:
        """Upload a bundle directory and queue it. Returns {submission_id, status}
        or the hub's {"error": ...}. final_candidate=True registers it for final
        evaluation without a validation run (e.g. too big for validation limits)."""
        bundle = Path(bundle)
        manifest = manifest_for(bundle)
        r = self._req("POST", f"/v1/tasks/{task}/submissions",
                      {"manifest": manifest, "final_candidate": final_candidate})
        if "submission_id" not in r:
            return r
        paths = {f["sha256"]: bundle / f["path"] for f in manifest["files"]}
        for up in r["uploads"]:
            for attempt in range(retries):
                try:
                    self._put(up, paths[up["sha256"]])
                    break
                except (urllib.error.URLError, OSError):
                    if attempt == retries - 1:
                        raise
                    time.sleep(2 ** attempt)
        return self._req("POST", f"/v1/submissions/{r['submission_id']}/commit")

    def _put(self, up: dict, path: Path) -> None:
        headers = {**up.get("headers", {}), "Content-Length": str(up["size"])}
        if up.get("auth"):
            headers["Authorization"] = f"Bearer {self.token}"
        with path.open("rb") as f:
            req = urllib.request.Request(up["url"], data=f, method=up.get("method", "PUT"),
                                         headers=headers)
            urllib.request.urlopen(req, timeout=max(self.timeout, 3600)).read()

    def register_final(self, task: str, bundle: str | Path) -> dict:
        """Register a model as a final-evaluation candidate: uploaded and kept,
        never run on validation, no evaluation budget spent. Returns
        {submission_id, status: "registered"}."""
        return self.submit_async(task, bundle, final_candidate=True)

    def status(self, submission_id: str) -> dict:
        return self._req("GET", f"/v1/submissions/{submission_id}")

    def submit(self, task: str, bundle: str | Path, poll_s: float = 10,
               max_wait_s: float = 6 * 3600) -> dict:
        """Submit and wait for the released result. Returns the node's payload,
        or the refusal ({"error": ...}) if the submission was not accepted."""
        r = self.submit_async(task, bundle)
        if "submission_id" not in r:
            return r
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            s = self.status(r["submission_id"])
            if s.get("status") == "done":
                return s["result"]
            time.sleep(poll_s)
        raise TimeoutError(f"submission {r['submission_id']} not released after {max_wait_s}s")

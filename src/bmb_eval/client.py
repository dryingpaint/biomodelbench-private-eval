"""Agent-side client. Standard library only, so it drops into any training
environment.

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

from .bundle import pack_dir


class Client:
    def __init__(self, url: str, token: str, timeout: float = 60):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _req(self, method: str, path: str, body: bytes | None = None) -> dict:
        req = urllib.request.Request(self.url + path, data=body, method=method, headers={
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/gzip" if body else "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read())
            except Exception:
                raise e

    def tasks(self) -> dict:
        return self._req("GET", "/v1/tasks")

    def contract(self, task: str) -> dict:
        for t in self.tasks()["tasks"]:
            if t["contract"]["id"] == task:
                return t["contract"]
        raise KeyError(task)

    def submit_async(self, task: str, bundle: str | Path | bytes) -> dict:
        data = bundle if isinstance(bundle, bytes) else pack_dir(Path(bundle))
        return self._req("POST", f"/v1/tasks/{task}/submissions", data)

    def status(self, submission_id: str) -> dict:
        return self._req("GET", f"/v1/submissions/{submission_id}")

    def submit(self, task: str, bundle: str | Path | bytes, poll_s: float = 10,
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

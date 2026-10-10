"""Operator alerts. Only run ids, task ids and error codes go into an alert,
never data or submitted-code output.

Kinds: security, infra, config, control.
"""
from __future__ import annotations

import json
import sys
import urllib.request

from . import audit


def send(cfg, kind: str, summary: str, **fields) -> None:
    audit.record(cfg.state_dir, "alert", kind=kind, summary=summary, **fields)
    text = f"[bmb-eval:{cfg.node_name}] {kind.upper()}: {summary}"
    if fields:
        text += " " + " ".join(f"{k}={v}" for k, v in fields.items())
    print(text, file=sys.stderr)
    if not cfg.webhook_url:
        return
    # Slack-compatible body. Most chat/incident tools accept {"text": ...}.
    req = urllib.request.Request(
        cfg.webhook_url, data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:
        audit.record(cfg.state_dir, "alert_delivery_failed", error=type(e).__name__)

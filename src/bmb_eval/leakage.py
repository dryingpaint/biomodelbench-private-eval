"""Worst-case information budget of a task's export channel.

Submitted code sees the private inputs and chooses the predictions, so it
can make the reported metrics, the error code, and the time to result
depend on private data. That channel cannot be closed, only narrowed and
counted. This module bounds it: a submission can carry at most
log2(number of distinguishable outcomes) bits out of the node.

The bound is deliberately pessimistic. Honest submissions reveal far less,
but the operator should size quotas against what an adversarial one could do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .contract import Contract
from .errors import ERRORS, PREDICTION_REASONS

# Error codes whose occurrence the submitted code can steer.
STEERABLE_ERRORS = [c for c, (_, cat) in ERRORS.items()
                    if cat in ("user", "security", "infra") and c != "predictions_invalid"]


@dataclass
class Budget:
    metric_bits: float           # inf when a metric has no bounds
    error_bits: float
    timing_bits: float
    per_submission: float
    per_day_per_submitter: float
    lifetime: float | None       # None when submissions_total is unset
    notes: list[str]


def _levels(m, decimals: int) -> float:
    if m.min is None or m.max is None:
        return math.inf
    span = m.max - m.min
    n = span + 1 if m.type == "int" else span * 10 ** decimals + 1
    return n + (1 if m.nullable else 0)


def budget(c: Contract) -> Budget:
    notes = []
    levels = 1.0
    for m in c.metrics:
        lv = _levels(m, c.decimals)
        if math.isinf(lv):
            notes.append(f"metric {m.name!r} has no min/max: export is unbounded; set both")
        levels *= lv
    metric_bits = math.log2(levels)

    detail_columns = len(c.outputs) + len(c.key) + 2  # outputs, keys, "*", joined key
    error_outcomes = len(STEERABLE_ERRORS) + detail_columns * len(PREDICTION_REASONS)
    error_bits = math.log2(error_outcomes)

    lim = c.limits
    run_ceiling = lim.predict_timeout_s + lim.validate_timeout_s + lim.score_timeout_s
    if lim.release_after_s >= run_ceiling:
        timing_bits = 0.0
    else:
        # One-second resolution over the window in which finishing time is visible.
        timing_bits = math.log2(run_ceiling - lim.release_after_s + 1)
        notes.append(f"result time is observable (release_after_s={lim.release_after_s} < "
                     f"{run_ceiling}s run ceiling): ~{timing_bits:.1f} bits/submission")

    outcome_bits = math.log2(levels + error_outcomes) if not math.isinf(levels) else math.inf
    per = outcome_bits + timing_bits
    if c.ladder_step is not None:
        notes.append("ladder on: most submissions return only 'not improved'; "
                     "bound above still applies to the improving ones")
    return Budget(
        metric_bits=metric_bits,
        error_bits=error_bits,
        timing_bits=timing_bits,
        per_submission=per,
        per_day_per_submitter=per * lim.submissions_per_day,
        lifetime=(per * (lim.submissions_total + (c.split.final_evaluations if c.split else 0))
                  if lim.submissions_total else None),
        notes=notes,
    )


def describe(c: Contract) -> str:
    b = budget(c)
    lines = [
        f"task {c.id} v{c.version}: worst-case export channel",
        f"  metrics   {b.metric_bits:8.1f} bits  ({len(c.metrics)} metrics at {c.decimals} decimals)",
        f"  errors    {b.error_bits:8.1f} bits",
        f"  timing    {b.timing_bits:8.1f} bits",
        f"  = {b.per_submission:.0f} bits/submission, "
        f"{b.per_day_per_submitter:.0f} bits/day/submitter",
        "  lifetime  " + (f"{b.lifetime:.0f} bits ({c.limits.submissions_total} submissions)"
                          if b.lifetime is not None else "unbounded (set limits.submissions_total)"),
    ]
    if c.split:
        lines.append(f"  split     {c.split.validation_fraction:.0%} validation (scored during the "
                     f"benchmark), rest test (at most {c.split.final_evaluations} final evaluations)")
    lines += [f"  ! {n}" for n in b.notes]
    return "\n".join(lines)

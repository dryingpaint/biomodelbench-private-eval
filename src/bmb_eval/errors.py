"""The only error formats that may leave the node.

Every error the agent sees is one of these codes with its fixed message.
Free text (exception messages, stderr, tracebacks) never leaves: submitted
code can put private data into any of them.

`detail` is optional and limited to names taken from the public task
contract plus an enumerated reason. It never carries a data value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# code -> (fixed message, category)
# category: "user" = the submission is at fault, "infra" = the node is at
# fault (counts toward persistent-failure alerts), "security" = alert at once,
# "policy" = refused by node policy.
ERRORS: dict[str, tuple[str, str]] = {
    "bundle_invalid": ("Bundle is not a valid model bundle archive.", "user"),
    "bundle_unsafe": ("Bundle archive contains unsafe paths or file types.", "security"),
    "bundle_too_large": ("Bundle exceeds the size limit for this task.", "user"),
    "dependency_install_failed": ("Installing bundled wheels failed.", "user"),
    "predict_import_failed": ("predict.py could not be imported or has no predict().", "user"),
    "predict_runtime_error": ("predict() raised an exception.", "user"),
    "predict_timeout": ("Prediction exceeded the time limit.", "user"),
    "predict_resource_limit": ("Prediction exceeded the memory or process limit.", "user"),
    "predictions_missing": ("No predictions file was produced.", "user"),
    "predictions_too_large": ("Predictions output exceeds the size limit.", "security"),
    "predictions_invalid": ("Predictions do not match the task output contract.", "user"),
    "scorer_failed": ("The scorer failed. The node operator has been notified.", "infra"),
    "output_contract_violation": ("Scorer output failed the export check. Operator notified.", "security"),
    "infra_error": ("Evaluation infrastructure error. The node operator has been notified.", "infra"),
    "quota_exceeded": ("Submission quota for this task is exhausted for today.", "policy"),
    "service_paused": ("The evaluation service is paused by its operator.", "policy"),
    "config_not_approved": ("Node configuration changed and awaits operator approval.", "policy"),
    "task_unknown": ("Unknown task id.", "policy"),
    "no_test_split": ("This task has no held-out test set.", "policy"),
    "final_quota_exceeded": ("Final (test set) evaluations for this task are used up.", "policy"),
}

# Reasons allowed in detail.reason for predictions_invalid.
PREDICTION_REASONS = {
    "unreadable",
    "missing_column",
    "unexpected_column",
    "wrong_type",
    "null_not_allowed",
    "non_finite",
    "below_min",
    "above_max",
    "not_in_allowed_values",
    "duplicate_keys",
    "missing_keys",
    "unexpected_keys",
    "wrong_length",
    "too_many_items",
    "not_in_candidates",
}


@dataclass
class EvalError(Exception):
    code: str
    detail: dict = field(default_factory=dict)
    # Provider-only diagnostic. Written to the internal run log, never exported.
    internal: str = ""

    def __post_init__(self) -> None:
        if self.code not in ERRORS:
            raise ValueError(f"unapproved error code {self.code!r}")

    @property
    def category(self) -> str:
        return ERRORS[self.code][1]

    def public(self) -> dict:
        out = {"code": self.code, "message": ERRORS[self.code][0]}
        if self.detail:
            out["detail"] = self.detail
        return out

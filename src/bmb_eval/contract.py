"""Task contracts: the fixed schema for inputs, outputs, answers and metrics.

A contract is a YAML file (tasks/<id>/contract.yaml). It is shared with the
agent minus the `answers` block, so the agent knows exactly what predict()
receives and must return, and what metrics come back.

This module is imported both by the controller and inside the validate and
score sandboxes, so it depends only on numpy/pandas/pyyaml.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .errors import PREDICTION_REASONS, EvalError

TYPES = {"string", "int", "float", "bool"}


@dataclass
class Field:
    name: str
    type: str
    nullable: bool = False
    min: float | None = None
    max: float | None = None
    allowed: list | None = None
    unit: str | None = None
    description: str = ""

    @classmethod
    def parse(cls, d: dict, where: str) -> "Field":
        unknown = set(d) - {f.name for f in dc_fields(cls)}
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        f = cls(**d)
        if f.type not in TYPES:
            raise ValueError(f"{where}.{f.name}: type must be one of {sorted(TYPES)}")
        return f


@dataclass
class Metric:
    name: str
    type: str = "float"  # float | int
    nullable: bool = False
    min: float | None = None
    max: float | None = None
    goal: str = "max"    # max | min; used by the ladder
    description: str = ""


@dataclass
class Limits:
    predict_timeout_s: int = 1800
    validate_timeout_s: int = 300
    score_timeout_s: int = 600
    memory: str = "8g"
    cpus: float = 2
    pids: int = 512
    tmpfs_mb: int = 2048
    max_bundle_mb: int = 2048
    max_predictions_mb: int = 512
    # Per submitter per UTC day.
    submissions_per_day: int = 20
    # Across all submitters, for the life of this contract version. None = no cap.
    submissions_total: int | None = None
    # Results become visible this many seconds after submission (not after the
    # run finishes), so run time cannot carry information. 0 = release on finish.
    release_after_s: int = 0


@dataclass
class Contract:
    id: str
    version: int
    title: str
    description: str
    key: list[str]
    inputs: list[Field]
    outputs: list[Field]
    answers: list[Field]
    metrics: list[Metric]
    headline: str
    decimals: int = 3
    max_export_bytes: int = 4096
    # When set, only improvements of at least this much on the headline metric
    # are reported (Blum & Hardt's Ladder); otherwise the previous best is repeated.
    ladder_step: float | None = None
    batch_size: int = 1024
    limits: Limits = field(default_factory=Limits)
    path: Path | None = None

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: Path) -> "Contract":
        c = cls.from_dict(yaml.safe_load(Path(path).read_text()))
        c.path = Path(path)
        return c

    @classmethod
    def from_dict(cls, raw: dict, public: bool = False) -> "Contract":
        """Build from contract.yaml contents, or from public() output when public=True
        (no answers block; used by sandboxes that must not know the answer schema)."""
        export = raw.get("export", {})
        c = cls(
            id=raw["id"],
            version=int(raw["version"]),
            title=raw.get("title", raw["id"]),
            description=raw.get("description", ""),
            key=list(raw["key"]),
            inputs=[Field.parse(f, "inputs") for f in raw["inputs"]],
            outputs=[Field.parse(f, "outputs") for f in raw["outputs"]],
            answers=[] if public else [Field.parse(f, "answers") for f in raw["answers"]],
            metrics=[Metric(**m) for m in raw["metrics"]],
            headline=raw["headline"],
            decimals=int(export.get("decimals", 3)),
            max_export_bytes=int(export.get("max_bytes", 4096)),
            ladder_step=export.get("ladder_step"),
            batch_size=int(raw.get("prediction", {}).get("batch_size", 1024)),
            limits=Limits(**raw.get("limits", {})),
        )
        c._check(public)
        return c

    def _check(self, public: bool) -> None:
        input_names = [f.name for f in self.inputs]
        answer_names = [f.name for f in self.answers]
        output_names = [f.name for f in self.outputs]
        for k in self.key:
            if k not in input_names or (not public and k not in answer_names):
                raise ValueError(f"key column {k!r} must appear in inputs and answers")
        if set(output_names) & set(self.key):
            raise ValueError("outputs must not redeclare key columns")
        if set(output_names) & set(answer_names):
            raise ValueError("output and answer column names must not overlap")
        if self.headline not in {m.name for m in self.metrics}:
            raise ValueError(f"headline {self.headline!r} is not a declared metric")
        for m in self.metrics:
            if m.type not in {"float", "int"}:
                raise ValueError(f"metric {m.name}: type must be float or int")
            if m.goal not in {"max", "min"}:
                raise ValueError(f"metric {m.name}: goal must be max or min")

    # ------------------------------------------------------------- views

    def public(self) -> dict:
        """What the agent may see. Omits the answers schema. Same layout as
        contract.yaml, so from_dict(public=True) reads it back."""
        def f(x) -> dict:
            return {k: v for k, v in x.__dict__.items() if v not in (None, "", False)}
        export = {"decimals": self.decimals, "max_bytes": self.max_export_bytes}
        if self.ladder_step is not None:
            export["ladder_step"] = self.ladder_step
        return {
            "id": self.id,
            "version": self.version,
            "title": self.title,
            "description": self.description,
            "key": self.key,
            "inputs": [f(x) for x in self.inputs],
            "outputs": [f(x) for x in self.outputs],
            "metrics": [f(m) for m in self.metrics],
            "headline": self.headline,
            "export": export,
            "prediction": {"batch_size": self.batch_size},
            "limits": dict(self.limits.__dict__),
        }

    def sha256(self) -> str:
        assert self.path is not None
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def metric(self, name: str) -> Metric:
        return next(m for m in self.metrics if m.name == name)


# ---------------------------------------------------------------------------
# Column checks shared by data checks and prediction checks
# ---------------------------------------------------------------------------

def _check_column(f: Field, s: pd.Series) -> str | None:
    """Return a reason code if the column violates the field spec, else None."""
    nulls = s.isna()
    if nulls.any() and not f.nullable:
        return "null_not_allowed"
    v = s[~nulls]
    if f.type == "string":
        if not v.map(lambda x: isinstance(x, str)).all():
            return "wrong_type"
    elif f.type == "bool":
        if not v.map(lambda x: isinstance(x, (bool, np.bool_))).all():
            return "wrong_type"
    elif f.type in ("int", "float"):
        if not (pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v)):
            return "wrong_type"
        arr = v.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            return "non_finite"
        if f.type == "int" and not np.all(arr == np.round(arr)):
            return "wrong_type"
        if f.min is not None and (arr < f.min).any():
            return "below_min"
        if f.max is not None and (arr > f.max).any():
            return "above_max"
    if f.allowed is not None and not v.isin(f.allowed).all():
        return "not_in_allowed_values"
    return None


def check_frame(fields: list[Field], df: pd.DataFrame, *, strict_columns: bool,
                key: list[str] | None = None) -> tuple[str, str] | None:
    """Return (column, reason) for the first violation, or None."""
    names = [f.name for f in fields]
    for f in fields:
        if f.name not in df.columns:
            return f.name, "missing_column"
    if strict_columns:
        extra = [c for c in df.columns if c not in names]
        if extra:
            return "*", "unexpected_column"
    for f in fields:
        reason = _check_column(f, df[f.name])
        if reason:
            return f.name, reason
    if key and df.duplicated(subset=key).any():
        return ",".join(key), "duplicate_keys"
    return None


def validate_predictions(contract: Contract, preds: pd.DataFrame,
                         input_keys: pd.DataFrame) -> pd.DataFrame:
    """Raise EvalError(predictions_invalid) unless preds match the output contract
    and cover exactly the input keys. Returns preds ordered like input_keys."""
    key_fields = [f for f in contract.inputs if f.name in contract.key]
    bad = check_frame(key_fields + contract.outputs, preds, strict_columns=True,
                      key=contract.key)
    if bad:
        raise invalid(*bad)
    if len(preds) > len(input_keys):
        raise invalid(",".join(contract.key), "unexpected_keys")
    merged = input_keys.merge(preds, on=contract.key, how="left", indicator=True)
    if (merged["_merge"] != "both").any():
        raise invalid(",".join(contract.key), "missing_keys")
    return merged.drop(columns="_merge")


def invalid(column: str, reason: str) -> EvalError:
    assert reason in PREDICTION_REASONS
    return EvalError("predictions_invalid", detail={"column": column, "reason": reason})


# ---------------------------------------------------------------------------
# Canonical predictions: the only form the answer-holding scorer ever parses.
#
# The validate sandbox turns untrusted parquet into one fixed-dtype numpy array
# per output column (plus a null mask), in inputs.parquet row order. The score
# sandbox loads it with allow_pickle=False and checks names, dtypes and lengths,
# so a malicious predictions file never reaches a parser next to the answers.
# ---------------------------------------------------------------------------

_DTYPES = {"float": np.float64, "int": np.int64, "bool": np.bool_, "string": np.str_}
_FILL = {"float": 0.0, "int": 0, "bool": False, "string": ""}


def to_canonical(contract: Contract, ordered: pd.DataFrame) -> bytes:
    arrays = {}
    for f in contract.outputs:
        s = ordered[f.name]
        mask = s.isna().to_numpy()
        arrays[f.name] = s.where(~mask, _FILL[f.type]).to_numpy().astype(_DTYPES[f.type])
        arrays[f.name + "__null"] = mask.astype(np.bool_)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def from_canonical(contract: Contract, data: bytes, n_rows: int) -> pd.DataFrame:
    """Parse canonical predictions; raise ValueError on anything unexpected."""
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        expected = {f.name for f in contract.outputs} | {f.name + "__null" for f in contract.outputs}
        if set(z.files) != expected:
            raise ValueError("canonical predictions: wrong arrays")
        cols = {}
        for f in contract.outputs:
            v, m = z[f.name], z[f.name + "__null"]
            if v.shape != (n_rows,) or m.shape != (n_rows,) or m.dtype != np.bool_:
                raise ValueError(f"canonical predictions: bad shape for {f.name}")
            if np.dtype(_DTYPES[f.type]).kind != v.dtype.kind:
                raise ValueError(f"canonical predictions: bad dtype for {f.name}")
            s = pd.Series(v)
            cols[f.name] = s.where(~m, None) if m.any() else s
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Output gate: the last check before anything leaves the node
# ---------------------------------------------------------------------------

def gate_metrics(contract: Contract, raw: bytes) -> dict[str, Any]:
    """Parse scorer output and return the exportable metrics dict.

    Exactly the declared metric names, numeric or declared-nullable,
    finite, within bounds, rounded to the contract precision. Anything
    else is an output_contract_violation and nothing is exported.
    """
    def violation(why: str) -> EvalError:
        return EvalError("output_contract_violation", internal=why)

    if len(raw) > contract.max_export_bytes * 4:
        raise violation(f"scorer output {len(raw)} bytes")
    try:
        obj = json.loads(raw)
    except Exception as e:
        raise violation(f"not json: {type(e).__name__}")
    if not isinstance(obj, dict):
        raise violation("not an object")
    declared = {m.name: m for m in contract.metrics}
    if set(obj) != set(declared):
        raise violation(f"keys {sorted(obj)} != declared {sorted(declared)}")
    out: dict[str, Any] = {}
    for name, m in declared.items():
        v = obj[name]
        if v is None:
            if not m.nullable:
                raise violation(f"{name} is null")
            out[name] = None
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise violation(f"{name} has type {type(v).__name__}")
        if not math.isfinite(v):
            raise violation(f"{name} not finite")
        if m.min is not None and v < m.min or m.max is not None and v > m.max:
            raise violation(f"{name} out of bounds")
        if m.type == "int" and float(v) != int(v):
            raise violation(f"{name} not an integer")
        out[name] = int(v) if m.type == "int" else round(float(v), contract.decimals)
    if len(json.dumps(out)) > contract.max_export_bytes:
        raise violation("export exceeds max_bytes")
    return out

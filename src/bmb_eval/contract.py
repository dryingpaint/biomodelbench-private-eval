"""Task contracts: the fixed schema for inputs, outputs, answers and metrics.

A contract is a YAML file (tasks/<id>/contract.yaml). It is shared with the
agent minus the `answers` block, so the agent knows exactly what predict()
receives and must return, and what metrics come back.

Every task is "one prediction per sample": a samples table keyed by `key`.
Field types cover the shapes the benchmark needs:

    string | int | float | bool   one value per sample
    vector                        fixed-length numbers per sample (e.g. one per gene);
                                  `length`, or `labels` (a list, or a file in the task dir)
    list                          variable-length records per sample (e.g. MS/MS peaks,
                                  ranked TFs); `items` (scalar fields), `max_items`;
                                  string items may set `in_input: <input column>` to
                                  restrict them to that sample's candidates
    file                          (inputs only) path to a file the model may open
    any                           (inputs only) passed through unchecked

`data` describes the provider's data when it is a folder rather than one table:
`samples` names the samples table, `include` lists files or folders the model
also gets as-is (e.g. an EHR events table, a control-cell .h5ad), and `private`
lists files only the scorer gets (answers in any format, e.g. perturbed-cell
populations). Answers may then be just the key column.

`split` holds out a test set: the provider's rows are split once, randomly,
on their machine, into a validation set (the only one scored during the
benchmark) and a test set (scored only for final evaluations BioModelBench
requests, at most `final_evaluations` times, enforced by the node):
`split: {validation_fraction: 0.5, final_evaluations: 3}`. Add `group_by: <input
column>` (e.g. smiles) to keep all rows of a group in the same half, so one
molecule's other endpoints can't leak its test labels.

`final_only: true` on a metric reports it only in final (held-out test)
evaluations, never in benchmark feedback (needs `split`).

`final_limits` overrides `limits` for final evaluations only, e.g.
`{gpus: 2, gpu_memory_gb: 80, max_bundle_gb: 150}`: validation runs stay
small (CPU), the final runs get the hardware for the largest models. Models
too big for validation are registered for finals without a validation run.

`per` on a metric expands it into one metric per group, e.g.
`{name: mae, per: [hERG, AMES]}` -> `mae[hERG]`, `mae[AMES]`. Every group adds
to the leakage budget, so prefer macro-averages plus a few named groups.

This module is imported both by the controller and inside the validate and
score sandboxes, so it depends only on numpy/pandas/pyyaml.
"""
from __future__ import annotations

import dataclasses
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

SCALARS = {"string", "int", "float", "bool"}
OUTPUT_TYPES = SCALARS | {"vector", "list"}
TYPES = OUTPUT_TYPES | {"file", "any"}


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
    length: int | None = None          # vector
    labels: list | None = None         # vector element names
    items: list | None = None          # list: [Field] of scalar item fields
    max_items: int | None = None       # list
    in_input: str | None = None        # list item: value must be in this input column's list
    format: str | None = None          # file: e.g. npy, png, h5ad (for docs and synthetic data)
    shape: list | None = None          # file (npy): array shape, for synthetic data

    @classmethod
    def parse(cls, d: dict, where: str, base: Path | None = None) -> "Field":
        unknown = set(d) - {f.name for f in dc_fields(cls)}
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        d = dict(d)
        if d.get("items") is not None:
            d["items"] = [cls.parse(i, f"{where}.{d['name']}.items") for i in d["items"]]
        if isinstance(d.get("labels"), str):
            if base is None:
                raise ValueError(f"{where}.{d['name']}: labels file needs the contract's folder")
            d["labels"] = [x.strip() for x in (base / d["labels"]).read_text().splitlines() if x.strip()]
        f = cls(**d)
        if f.type not in TYPES:
            raise ValueError(f"{where}.{f.name}: type must be one of {sorted(TYPES)}")
        if f.type == "vector":
            if f.labels is not None:
                if f.length is not None and f.length != len(f.labels):
                    raise ValueError(f"{where}.{f.name}: length and labels disagree")
                f.length = len(f.labels)
            if not f.length:
                raise ValueError(f"{where}.{f.name}: vector needs length or labels")
        if f.type == "list":
            if not f.items or not f.max_items:
                raise ValueError(f"{where}.{f.name}: list needs items and max_items")
            if any(i.type not in SCALARS for i in f.items):
                raise ValueError(f"{where}.{f.name}: list items must be scalar fields")
        return f

    def public(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            if v in (None, "", False):
                continue
            out[k] = [i.public() for i in v] if k == "items" else v
        return out


@dataclass
class Metric:
    name: str
    type: str = "float"  # float | int
    nullable: bool = False
    min: float | None = None
    max: float | None = None
    goal: str = "max"    # max | min; used by the ladder
    description: str = ""
    # Reported only by final (held-out test) evaluations, never in benchmark
    # feedback: e.g. absolute errors, which would let agents fit the provider's
    # assay offset. Needs `split`. Enforced by the node's output gate.
    final_only: bool = False


@dataclass
class Limits:
    predict_timeout_s: int = 1800
    validate_timeout_s: int = 300
    score_timeout_s: int = 600
    memory: str = "16g"
    cpus: float = 4
    gpus: int = 0              # NVIDIA GPUs given to predict (0 = CPU only)
    gpu_memory_gb: int = 0     # per-GPU memory the task needs; checked at connect
    pids: int = 1024
    tmpfs_mb: int = 8192
    shm_mb: int = 2048         # /dev/shm, for PyTorch data loaders
    max_bundle_gb: int = 64    # model bundle size (weights included)
    max_predictions_mb: int = 512
    # Per submitter per UTC day.
    submissions_per_day: int = 20
    # Across all submitters, for the life of this contract version. None = no cap.
    submissions_total: int | None = None
    # Results become visible this many seconds after submission (not after the
    # run finishes), so run time cannot carry information. 0 = release on finish.
    release_after_s: int = 0


@dataclass
class DataLayout:
    """How the provider's --data is organized. Omitted = one samples table."""
    samples: str | None = None          # samples table inside the --data folder
    # [{path, description, fields? | shape?}] passed to the model as-is; fields
    # (for tables) or shape (for .npy) let agents' dry runs generate fake ones.
    include: list = field(default_factory=list)
    # [{path, description}] given only to the scorer: answers in any format.
    private: list = field(default_factory=list)


@dataclass
class Split:
    """Validation/test split of the private rows (see module docstring)."""
    validation_fraction: float
    final_evaluations: int = 3
    group_by: str | None = None   # input column; rows sharing it stay in one half


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
    # Overrides of `limits` for final (test set) evaluations, e.g. GPUs for the
    # largest models; validation runs keep the plain limits. Quotas don't apply.
    final_limits: dict = field(default_factory=dict)
    data: DataLayout = field(default_factory=DataLayout)
    split: Split | None = None
    path: Path | None = None

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: Path) -> "Contract":
        path = Path(path)
        c = cls.from_dict(yaml.safe_load(path.read_text()), base=path.parent)
        c.path = path
        return c

    @classmethod
    def from_dict(cls, raw: dict, public: bool = False, base: Path | None = None) -> "Contract":
        """Build from contract.yaml contents, or from public() output when public=True
        (no answers block; used by sandboxes that must not know the answer schema)."""
        export = raw.get("export", {})
        metrics = []
        for m in raw["metrics"]:
            m = dict(m)
            per = m.pop("per", None)
            if per:
                metrics += [Metric(**{**m, "name": f"{m['name']}[{g}]"}) for g in per]
            else:
                metrics.append(Metric(**m))
        data = raw.get("data") or {}
        c = cls(
            id=raw["id"],
            version=int(raw["version"]),
            title=raw.get("title", raw["id"]),
            description=raw.get("description", ""),
            key=list(raw["key"]),
            inputs=[Field.parse(f, "inputs", base) for f in raw["inputs"]],
            outputs=[Field.parse(f, "outputs", base) for f in raw["outputs"]],
            answers=[] if public else [Field.parse(f, "answers", base) for f in raw["answers"]],
            metrics=metrics,
            headline=raw["headline"],
            decimals=int(export.get("decimals", 3)),
            max_export_bytes=int(export.get("max_bytes", 4096)),
            ladder_step=export.get("ladder_step"),
            batch_size=int(raw.get("prediction", {}).get("batch_size", 1024)),
            limits=Limits(**raw.get("limits", {})),
            final_limits=dict(raw.get("final_limits") or {}),
            data=DataLayout(samples=data.get("samples"), include=list(data.get("include") or []),
                            private=list(data.get("private") or [])),
            split=Split(**raw["split"]) if raw.get("split") else None,
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
        if set(answer_names) - set(self.key) & set(input_names):
            raise ValueError("answer columns must not also be inputs")
        for f in self.outputs + self.answers:
            if f.type not in OUTPUT_TYPES:
                raise ValueError(f"{f.name}: type {f.type} is only allowed for inputs")
        for f in self.outputs:
            for i in f.items or []:
                if i.in_input and i.in_input not in input_names:
                    raise ValueError(f"{f.name}.{i.name}: in_input {i.in_input!r} is not an input")
        if self.split and not (0 < self.split.validation_fraction < 1
                               and self.split.final_evaluations >= 1):
            raise ValueError("split: validation_fraction must be in (0, 1) and final_evaluations >= 1")
        if self.split and self.split.group_by and self.split.group_by not in input_names:
            raise ValueError(f"split.group_by {self.split.group_by!r} is not an input column")
        if self.final_limits:
            if not self.split:
                raise ValueError("final_limits need a held-out test set (split)")
            quotas = {"submissions_per_day", "submissions_total", "release_after_s"}
            if bad := set(self.final_limits) & quotas:
                raise ValueError(f"final_limits can't set {sorted(bad)}: quotas apply to validation")
            Limits(**self.final_limits)   # unknown keys raise here
        if self.headline not in {m.name for m in self.metrics}:
            raise ValueError(f"headline {self.headline!r} is not a declared metric")
        if self.metric(self.headline).final_only:
            raise ValueError("the headline metric can't be final_only")
        if any(m.final_only for m in self.metrics) and not self.split:
            raise ValueError("final_only metrics need a held-out test set (split)")
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
        out = {
            "id": self.id,
            "version": self.version,
            "title": self.title,
            "description": self.description,
            "key": self.key,
            "inputs": [x.public() for x in self.inputs],
            "outputs": [x.public() for x in self.outputs],
            "metrics": [f(m) for m in self.metrics],
            "headline": self.headline,
            "export": export,
            "prediction": {"batch_size": self.batch_size},
            "limits": dict(self.limits.__dict__),
        }
        if self.final_limits:
            out["final_limits"] = dict(self.final_limits)
        if self.data.include:
            out["data"] = {"include": self.data.include}
        if self.split:
            out["split"] = dict(self.split.__dict__)
        return out

    def limits_for(self, split: str) -> Limits:
        """Resource limits for a validation or a final (split="test") evaluation."""
        if split == "test" and self.final_limits:
            return dataclasses.replace(self.limits, **self.final_limits)
        return self.limits

    def sha256(self) -> str:
        assert self.path is not None
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def metric(self, name: str) -> Metric:
        return next(m for m in self.metrics if m.name == name)


# ---------------------------------------------------------------------------
# Column checks shared by data checks and prediction checks
# ---------------------------------------------------------------------------

def _listlike(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))


def _check_numbers(f: Field, arr: np.ndarray) -> str | None:
    if not np.isfinite(arr).all():
        return "non_finite"
    if f.min is not None and (arr < f.min).any():
        return "below_min"
    if f.max is not None and (arr > f.max).any():
        return "above_max"
    return None


def _check_column(f: Field, s: pd.Series) -> str | None:
    """Return a reason code if the column violates the field spec, else None."""
    if f.type == "any":
        return None
    if f.type == "vector":
        if not s.map(_listlike).all():
            return "null_not_allowed" if s.isna().any() else "wrong_type"
        if (s.map(len) != f.length).any():
            return "wrong_length"
        try:
            arr = np.stack([np.asarray(v, dtype=float) for v in s]) if len(s) else np.zeros((0, f.length))
        except (TypeError, ValueError):
            return "wrong_type"
        return _check_numbers(f, arr)
    if f.type == "list":
        names = [i.name for i in f.items]
        for v in s:
            if not isinstance(v, dict) or set(v) != set(names) or not all(_listlike(v[n]) for n in names):
                return "wrong_type"
            lens = {len(v[n]) for n in names}
            if len(lens) != 1:
                return "wrong_length"
            if lens.pop() > f.max_items:
                return "too_many_items"
        for i in f.items:
            flat = pd.Series([x for v in s for x in v[i.name]], dtype=object)
            if i.type in ("int", "float") and len(flat):
                flat = pd.to_numeric(flat, errors="coerce")
                if flat.isna().any():
                    return "wrong_type"
            if reason := _check_column(i, flat):
                return reason
        return None
    nulls = s.isna()
    if nulls.any() and not f.nullable:
        return "null_not_allowed"
    v = s[~nulls]
    if f.type in ("string", "file"):
        if not v.map(lambda x: isinstance(x, str)).all():
            return "wrong_type"
    elif f.type == "bool":
        if not v.map(lambda x: isinstance(x, (bool, np.bool_))).all():
            return "wrong_type"
    elif f.type in ("int", "float"):
        if len(v) == 0:          # an optional column left entirely empty is fine
            return None
        if not (pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v)):
            if not v.map(lambda x: isinstance(x, (int, float, np.integer, np.floating))
                         and not isinstance(x, bool)).all():
                return "wrong_type"
            v = v.astype(float)
        arr = v.to_numpy(dtype=float)
        if f.type == "int" and np.isfinite(arr).all() and not np.all(arr == np.round(arr)):
            return "wrong_type"
        if reason := _check_numbers(f, arr):
            return reason
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


def candidate_columns(contract: Contract) -> list[str]:
    """Input columns that list outputs are checked against (in_input)."""
    return sorted({i.in_input for f in contract.outputs for i in (f.items or []) if i.in_input})


def validate_predictions(contract: Contract, preds: pd.DataFrame,
                         inputs: pd.DataFrame) -> pd.DataFrame:
    """Raise EvalError(predictions_invalid) unless preds match the output contract
    and cover exactly the input keys. `inputs` has the key columns plus any
    candidate columns. Returns preds ordered like inputs."""
    key_fields = [f for f in contract.inputs if f.name in contract.key]
    bad = check_frame(key_fields + contract.outputs, preds, strict_columns=True,
                      key=contract.key)
    if bad:
        raise invalid(*bad)
    if len(preds) > len(inputs):
        raise invalid(",".join(contract.key), "unexpected_keys")
    merged = inputs.merge(preds, on=contract.key, how="left", indicator=True)
    if (merged["_merge"] != "both").any():
        raise invalid(",".join(contract.key), "missing_keys")
    for f in contract.outputs:
        for i in f.items or []:
            if i.in_input:
                for got, allowed in zip(merged[f.name], merged[i.in_input]):
                    if not set(map(str, got[i.name])) <= set(map(str, allowed)):
                        raise invalid(f.name, "not_in_candidates")
    return merged.drop(columns=["_merge", *candidate_columns(contract)])


def invalid(column: str, reason: str) -> EvalError:
    assert reason in PREDICTION_REASONS
    return EvalError("predictions_invalid", detail={"column": column, "reason": reason})


# ---------------------------------------------------------------------------
# Canonical predictions: the only form the answer-holding scorer ever parses.
#
# The validate sandbox turns untrusted parquet into fixed-dtype numpy arrays
# (scalars: values + null mask; vectors: an n x length matrix; lists: one flat
# array per item field + offsets), in samples-table row order. The score
# sandbox loads them with allow_pickle=False and checks names, dtypes and
# shapes, so a malicious predictions file never reaches a parser next to the
# answers.
# ---------------------------------------------------------------------------

_DTYPES = {"float": np.float64, "int": np.int64, "bool": np.bool_, "string": np.str_}
_FILL = {"float": 0.0, "int": 0, "bool": False, "string": ""}


def _flat(values: list, typ: str) -> np.ndarray:
    parts = [np.asarray(v) for v in values if len(v)]
    out = np.concatenate(parts) if parts else np.array([], dtype=_DTYPES[typ])
    return out.astype(_DTYPES[typ])


def to_canonical(contract: Contract, ordered: pd.DataFrame) -> bytes:
    arrays = {}
    for f in contract.outputs:
        s = ordered[f.name]
        if f.type == "vector":
            arrays[f.name] = (np.stack([np.asarray(v, dtype=float) for v in s]) if len(s)
                              else np.zeros((0, f.length)))
        elif f.type == "list":
            lens = [len(v[f.items[0].name]) for v in s]
            arrays[f"{f.name}.__offsets"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
            for i in f.items:
                arrays[f"{f.name}.{i.name}"] = _flat([v[i.name] for v in s], i.type)
        else:
            mask = s.isna().to_numpy()
            arrays[f.name] = s.where(~mask, _FILL[f.type]).to_numpy().astype(_DTYPES[f.type])
            arrays[f.name + "__null"] = mask.astype(np.bool_)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _expected_arrays(contract: Contract) -> set[str]:
    out = set()
    for f in contract.outputs:
        if f.type == "vector":
            out.add(f.name)
        elif f.type == "list":
            out |= {f"{f.name}.__offsets"} | {f"{f.name}.{i.name}" for i in f.items}
        else:
            out |= {f.name, f.name + "__null"}
    return out


def from_canonical(contract: Contract, data: bytes, n_rows: int) -> pd.DataFrame:
    """Parse canonical predictions; raise ValueError on anything unexpected.
    Vectors come back as one numpy row per sample (np.stack(df[col]) gives the
    matrix); lists as one {item: array} dict per sample."""
    def kind_ok(arr, typ):
        return np.dtype(_DTYPES[typ]).kind == arr.dtype.kind

    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        if set(z.files) != _expected_arrays(contract):
            raise ValueError("canonical predictions: wrong arrays")
        cols = {}
        for f in contract.outputs:
            if f.type == "vector":
                m = z[f.name]
                if m.shape != (n_rows, f.length) or m.dtype.kind != "f":
                    raise ValueError(f"canonical predictions: bad vector {f.name}")
                cols[f.name] = list(m)
            elif f.type == "list":
                o = z[f"{f.name}.__offsets"]
                if (o.shape != (n_rows + 1,) or o.dtype.kind != "i" or (n_rows and o[0] != 0)
                        or (np.diff(o) < 0).any() or (np.diff(o) > f.max_items).any()):
                    raise ValueError(f"canonical predictions: bad offsets for {f.name}")
                items = {}
                for i in f.items:
                    a = z[f"{f.name}.{i.name}"]
                    if a.shape != (int(o[-1]) if n_rows else 0,) or not kind_ok(a, i.type):
                        raise ValueError(f"canonical predictions: bad items {f.name}.{i.name}")
                    items[i.name] = a
                cols[f.name] = [{k: a[o[r]:o[r + 1]] for k, a in items.items()} for r in range(n_rows)]
            else:
                v, m = z[f.name], z[f.name + "__null"]
                if v.shape != (n_rows,) or m.shape != (n_rows,) or m.dtype != np.bool_:
                    raise ValueError(f"canonical predictions: bad shape for {f.name}")
                if not kind_ok(v, f.type):
                    raise ValueError(f"canonical predictions: bad dtype for {f.name}")
                s = pd.Series(v)
                cols[f.name] = s.where(~m, None) if m.any() else s
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Output gate: the last check before anything leaves the node
# ---------------------------------------------------------------------------

def gate_metrics(contract: Contract, raw: bytes, split: str = "validation") -> dict[str, Any]:
    """Parse scorer output and return the exportable metrics dict.

    Exactly the declared metric names, numeric or declared-nullable,
    finite, within bounds, rounded to the contract precision. Anything
    else is an output_contract_violation and nothing is exported.
    final_only metrics leave only with final (split="test") evaluations.
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
    if split != "test":
        out = {k: v for k, v in out.items() if not declared[k].final_only}
    if len(json.dumps(out)) > contract.max_export_bytes:
        raise violation("export exceeds max_bytes")
    return out

"""Turn the provider's --data (one table, or a folder) into what the node uses:

    private_data/<task>/inputs/           the only thing the model can see (read-only)
        samples.parquet                   key + input columns; file columns point into files/
        files/...                         files named in file columns (images, MSAs, ...)
        <data.include paths>              pass-through files (e.g. events table, .h5ad)
    private_data/<task>/answers.parquet   key + answer columns; only the scorer sees it
    private_data/<task>/answers_files/    <data.private paths>; only the scorer sees them

Large files are hard-linked when possible, so nothing is duplicated on disk.
Vector columns may be given as one list column, or as one column per label
(e.g. a column per gene). List and vector cells may be JSON in a CSV.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import Contract, Field

TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet")


class DataError(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataError([f"Can't find {path}."])
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix not in (".csv", ".tsv", ".txt"):
        raise DataError([f"{path.name}: please provide a .csv, .tsv or .parquet file."])
    return pd.read_csv(path, sep="\t" if path.suffix in (".tsv", ".txt") else ",")


def _json_cells(s: pd.Series) -> pd.Series:
    return s.map(lambda v: json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v)


def _column(df: pd.DataFrame, f: Field) -> pd.Series | None:
    if f.name in df.columns:
        s = df[f.name]
        return _json_cells(s) if f.type in ("vector", "list") else s
    if f.type == "vector" and f.labels and all(lbl in df.columns for lbl in f.labels):
        m = df[f.labels].to_numpy(dtype=float)
        return pd.Series(list(m), index=df.index)
    return None


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, copy_function=lambda s, d: _link(Path(s), Path(d)),
                        dirs_exist_ok=True)
        return
    if dst.exists():          # several rows may share one file (e.g. one slide)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _inside(base: Path, rel: str) -> Path | None:
    """base/rel if it stays inside base, else None."""
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        return None
    p = (base / rel).resolve()
    return p if p.is_relative_to(base.resolve()) else None


def _protected(src: Path, protected: set[Path]) -> bool:
    """True if src is, or lies inside, something the model must never see."""
    return any(src == p or src.is_relative_to(p) for p in protected)


def _find_samples(folder: Path, c: Contract) -> Path:
    if c.data.samples:
        return folder / c.data.samples
    tables = [p for p in sorted(folder.iterdir()) if p.suffix in TABLE_SUFFIXES
              and p.name not in {i["path"] for i in c.data.include}]
    if len(tables) != 1:
        raise DataError([f"Couldn't tell which file in {folder} is the samples table "
                         f"(found: {', '.join(p.name for p in tables) or 'none'})."])
    return tables[0]


def prepare(c: Contract, data: Path, dest: Path) -> tuple[Path, Path, int]:
    """Build dest/inputs/ and dest/answers.parquet. Returns (inputs_dir, answers, rows)."""
    data = data.expanduser().resolve()
    if data.is_dir():
        base, samples_path = data, _find_samples(data, c)
    else:
        if c.data.include or c.data.private or any(f.type == "file" for f in c.inputs):
            raise DataError(["This task needs a folder (a table plus the files it uses), "
                             "not a single file. Point --data at the folder."])
        base, samples_path = data.parent, data
    df = read_table(samples_path)

    problems, cols = [], {}
    fields = list({f.name: f for f in c.inputs + c.answers}.values())
    for f in fields:
        s = _column(df, f)
        if s is None:
            hint = (f" (or one column per label: {', '.join(f.labels[:3])}, ...)"
                    if f.type == "vector" and f.labels else "")
            problems.append(f"It has no column '{f.name}'{hint}.")
        else:
            cols[f.name] = s
    if problems:
        problems.append(f"Columns found: {', '.join(map(str, df.columns[:20]))}"
                        + (" ..." if len(df.columns) > 20 else ""))
        raise DataError(problems)

    staging = dest / "inputs.new"
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "files").mkdir(parents=True)
    # Never expose to the model: the samples table (it holds the answer
    # columns) or any scorer-only file, even if a file column names it.
    protected = {samples_path.resolve()}
    for prv in c.data.private:
        if (p := _inside(base, prv["path"])) is not None:
            protected.add(p)

    for f in c.inputs:
        if f.type != "file":
            continue
        out, bad = [], []
        for rel in cols[f.name]:
            src = _inside(base, rel)
            if src is None or not src.is_file() or _protected(src, protected):
                bad.append(str(rel))
                out.append(None)
                continue
            _link(src, staging / "files" / rel)
            out.append(f"files/{rel}")
        if bad:
            problems.append(f"{len(bad)} file(s) named in column '{f.name}' are missing or "
                            f"outside the data folder (first: {bad[0]}).")
        cols[f.name] = pd.Series(out, index=df.index)

    for inc in c.data.include:
        src = _inside(base, inc["path"])
        if src is None or not src.exists() or _protected(src, protected):
            problems.append(f"The folder should contain '{inc['path']}'"
                            + (" (and it can't also be scorer-only)."
                               if src is not None and _protected(src, protected) else "."))
            continue
        _link(src, staging / inc["path"])
    if problems:
        shutil.rmtree(staging, ignore_errors=True)
        raise DataError(problems)

    private_dir = dest / "answers_files"
    shutil.rmtree(private_dir, ignore_errors=True)
    for prv in c.data.private:
        src = _inside(base, prv["path"])
        if src is None or not src.exists():
            problems.append(f"The folder should contain '{prv['path']}'.")
            continue
        _link(src, private_dir / prv["path"])
    if problems:
        shutil.rmtree(staging, ignore_errors=True)
        raise DataError(problems)

    pd.DataFrame({f.name: cols[f.name] for f in c.inputs}).to_parquet(
        staging / "samples.parquet", index=False)
    pd.DataFrame({f.name: cols[f.name] for f in c.answers}).to_parquet(
        dest / "answers.parquet", index=False)
    final = dest / "inputs"
    shutil.rmtree(final, ignore_errors=True)
    staging.rename(final)
    return final, dest / "answers.parquet", len(df)


def apply_split(c: Contract, dest: Path) -> tuple[Path, Path, int, int]:
    """Split dest/inputs/ into dest/inputs_validation/ and dest/inputs_test/.

    Random, with a secret seed, on the provider's machine. The assignment is
    saved in dest/split.json and reused on re-runs, so a row never moves
    between validation and test once the split exists (new rows are assigned
    at random). The combined inputs/ folder is removed so it can't be mounted
    by mistake; the answers file stays whole (only the scorer reads it).
    """
    import secrets as _secrets
    full = dest / "inputs"
    samples = pd.read_parquet(full / "samples.parquet")
    keys = samples[c.key].astype(str).agg("\x1f".join, axis=1)
    split_file = dest / "split.json"
    assignment = json.loads(split_file.read_text()) if split_file.exists() else {}
    rng = np.random.default_rng(_secrets.randbits(64))
    for k in keys:
        if k not in assignment:
            assignment[k] = "validation" if rng.random() < c.split.validation_fraction else "test"
    split_file.write_text(json.dumps(assignment))
    os.chmod(split_file, 0o600)

    file_cols = [f.name for f in c.inputs if f.type == "file"]
    out = {}
    for name in ("validation", "test"):
        d = dest / f"inputs_{name}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        part = samples[keys.map(assignment).eq(name).to_numpy()]
        part.to_parquet(d / "samples.parquet", index=False)
        for col in file_cols:                      # only this half's files
            for rel in part[col].dropna():
                _link(full / rel, d / rel)
        for inc in c.data.include:                 # pass-through files go to both
            if (full / inc["path"]).exists():
                _link(full / inc["path"], d / inc["path"])
        out[name] = (d, len(part))
    shutil.rmtree(full)
    return out["validation"][0], out["test"][0], out["validation"][1], out["test"][1]


def describe_inputs(c: Contract) -> str:
    """One line for the approval summary: everything the model will see."""
    parts = [f.name for f in c.inputs]
    parts += [i["path"] for i in c.data.include]
    return ", ".join(parts)


def describe_hidden(c: Contract) -> str:
    """One line for the approval summary: what only the scorer sees."""
    parts = [f.name for f in c.answers if f.name not in c.key]
    parts += [p["path"] for p in c.data.private]
    return ", ".join(parts) or "(none)"


def synth_files(c: Contract, n: int, dest: Path, seed: int = 0) -> pd.DataFrame:
    """Synthetic inputs folder for agents' dry runs: samples.parquet, fake files
    of the declared format/shape, and empty placeholders for pass-through files."""
    from .scaffold import synth_inputs
    rng = np.random.default_rng(seed)
    df = synth_inputs(c, n=n, seed=seed)
    for f in c.inputs:
        if f.type != "file":
            continue
        names = []
        for r in range(n):
            rel = f"{f.name}/{r:06d}.{f.format or 'bin'}"
            p = dest / "files" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if f.format == "npy" and f.shape:
                np.save(p, rng.random(f.shape).astype(np.float32))
            else:
                p.write_bytes(b"")
            names.append(f"files/{rel}")
        df[f.name] = names
    for inc in c.data.include:
        p = dest / inc["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        if inc.get("shape") and p.suffix == ".npy":
            np.save(p, rng.random(inc["shape"]).astype(np.float32))
        elif inc.get("fields") and p.suffix in (".parquet", ".csv"):
            t = _synth_table(c, inc["fields"], df, rng)
            t.to_parquet(p, index=False) if p.suffix == ".parquet" else t.to_csv(p, index=False)
        else:
            p.touch()   # unknown format: the agent gets an empty placeholder
    df.to_parquet(dest / "samples.parquet", index=False)
    return df


def _synth_table(c: Contract, fields: list, samples: pd.DataFrame, rng, per_sample: int = 5):
    """Fake pass-through table: columns named like a key column reuse sample ids."""
    from .scaffold import synth_inputs
    fs = [Field.parse(f, "data.include.fields") for f in fields]
    n = len(samples) * per_sample
    fake = Contract(id="x", version=1, title="", description="", key=[], inputs=fs, outputs=[],
                    answers=[], metrics=[], headline="")
    t = synth_inputs(fake, n=n, seed=int(rng.integers(1 << 30)))
    for f in fs:
        if f.name in c.key:
            t[f.name] = rng.choice(samples[f.name].to_numpy(), n)
    return t

"""The benchmark's task shapes end to end: provider's raw data (table or
folder) -> prepare -> predict/validate/score -> gated metrics. One synthetic
dataset per example task in examples/tasks/."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bmb_eval import cli, leakage
from bmb_eval.bundle import pack_dir
from bmb_eval.config import Config
from bmb_eval.contract import Contract
from bmb_eval.prepare import DataError, prepare
from bmb_eval.runner import Runner

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples" / "tasks"
RNG = np.random.default_rng(7)
N = 60


# ------------------------------------------------- provider raw data per task

def raw_property(d: Path) -> Path:
    eps = ["solubility", "logD", "clearance"]
    smiles = ["C" * int(k) + "O" for k in RNG.integers(1, 20, N)]
    df = pd.DataFrame({"sample_id": [f"s{i}" for i in range(N)], "smiles": smiles,
                       "endpoint": [eps[i % 3] for i in range(N)],
                       "assay": [None if i % 7 == 0 else "A1" for i in range(N)],
                       "measured": [len(s) / 10 + RNG.normal(0, .1) for s in smiles],
                       "internal_notes": "not for the model"})
    df.to_csv(d / "data.csv", index=False)
    return d / "data.csv"


def raw_ehr(d: Path) -> Path:
    ids = [f"p{i}" for i in range(N)]
    pd.DataFrame({"patient_id": ids, "index_day": RNG.integers(0, 1000, N),
                  "age": RNG.integers(20, 90, N), "followup_days": RNG.uniform(10, 400, N),
                  "had_event": RNG.integers(0, 2, N)}).to_csv(d / "patients.csv", index=False)
    pd.DataFrame({"patient_id": RNG.choice(ids, 500), "day": RNG.integers(0, 1000, 500),
                  "code": RNG.choice(["I10", "E11", "N18"], 500)}).to_parquet(d / "events.parquet")
    return d


def raw_perturbation(d: Path) -> Path:
    genes = (EX / "perturbation" / "genes.txt").read_text().split()
    df = pd.DataFrame({"query_id": [f"q{i}" for i in range(N)],
                       "perturbed_gene": RNG.choice(genes, N),
                       "modality": RNG.choice(["KO", "CRISPRi", "CRISPRa", "OE"], N)})
    for g in genes:                       # answers as one column per gene
        df[g] = RNG.normal(0, 1, N)
    df.to_csv(d / "queries.csv", index=False)
    np.save(d / "control_expression.npy", RNG.random((200, len(genes))).astype(np.float32))
    return d


def raw_he(d: Path) -> Path:
    (d / "patches").mkdir()
    rows = []
    for i in range(N):
        patch = RNG.random((32, 32, 3)).astype(np.float32)
        np.save(d / "patches" / f"s{i}.npy", patch)
        m = patch.mean(axis=(0, 1))
        rows.append({"spot_id": f"s{i}", "slide": "slide1", "x": float(i), "y": 0.0,
                     "patch": f"patches/s{i}.npy", "EPCAM": m[0] * 5, "KRT8": m[1],
                     "VIM": m[2], "PTPRC": RNG.normal(), "COL1A1": RNG.normal()})
    pd.DataFrame(rows).to_csv(d / "spots.csv", index=False)
    return d


def raw_msms(d: Path) -> Path:
    mz = RNG.uniform(100, 600, N)
    pd.DataFrame({"spectrum_id": [f"m{i}" for i in range(N)], "smiles": "CCO",
                  "precursor_mz": mz, "collision_energy": 30.0,
                  "true_peaks": [json.dumps({"mz": [m, m - 18.0], "intensity": [1.0, 0.3]})
                                 for m in mz]}).to_csv(d / "spectra.csv", index=False)
    return d / "spectra.csv"


def raw_tf(d: Path) -> Path:
    pool = [f"TF{j}" for j in range(30)]
    cands = [list(RNG.choice(pool, 12, replace=False)) for _ in range(N)]
    pd.DataFrame({"query_id": [f"t{i}" for i in range(N)], "source_state": "fibroblast",
                  "target_state": "neuron", "candidates": [json.dumps(c) for c in cands],
                  "true_tfs": [json.dumps({"tf": sorted(c)[:2]}) for c in cands]}
                 ).to_csv(d / "queries.csv", index=False)
    return d / "queries.csv"


RAW = {"property_prediction": raw_property, "ehr_outcome": raw_ehr,
       "perturbation": raw_perturbation, "he_spatial": raw_he, "msms_peaks": raw_msms,
       "tf_ranking": raw_tf}


@pytest.fixture
def node(tmp_path):
    def make(task: str):
        root = tmp_path / "node"
        shutil.copytree(EX / task, root / "tasks" / task, ignore=shutil.ignore_patterns("bundle"))
        raw = tmp_path / "provider"
        raw.mkdir()
        data = RAW[task](raw)
        c = Contract.load(root / "tasks" / task / "contract.yaml")
        inputs_dir, answers, n = prepare(c, data, root / "private_data" / task)
        (root / "config.yaml").write_text(json.dumps({
            "sandbox": {"backend": "local", "i_understand_local_backend_is_not_isolated": True},
            "tasks": {task: {"dir": f"tasks/{task}", "inputs": str(inputs_dir / "samples.parquet"),
                             "answers": str(answers), "inputs_dir": str(inputs_dir)}}}))
        return Config.load(root / "config.yaml"), data
    return make


def evaluate(cfg, task, bundle_dir):
    return Runner(cfg, send_alerts=False).evaluate(task, pack_dir(bundle_dir))


# ---------------------------------------------------------------- end to end

@pytest.mark.parametrize("task", sorted(RAW))
def test_example_task_scores(node, task):
    cfg, _ = node(task)
    p = evaluate(cfg, task, EX / task / "bundle")
    assert p["status"] == "ok", p
    assert set(p["metrics"]) == {m.name for m in cfg.tasks[task].contract.metrics}


@pytest.mark.parametrize("task", sorted(RAW))
def test_model_sees_only_inputs(node, task):
    cfg, _ = node(task)
    t = cfg.tasks[task]
    visible = {str(p.relative_to(t.inputs_dir)) for p in t.inputs_dir.rglob("*") if p.is_file()}
    cols = set(pd.read_parquet(t.inputs).columns)
    assert cols == {f.name for f in t.contract.inputs}
    hidden = {f.name for f in t.contract.answers} - set(t.contract.key)
    assert not cols & hidden
    assert not any(p.endswith((".csv", ".tsv")) for p in visible)   # raw samples table never copied


@pytest.mark.parametrize("task", sorted(RAW))
def test_agent_dryrun_on_synthetic_data(task, capsys):
    pub = Contract.load(EX / task / "contract.yaml").public()
    path = Path(f"/tmp/bmb_pub_{task}.json")
    path.write_text(json.dumps(pub))
    try:
        cli.main(["dryrun", str(path), str(EX / task / "bundle"), "-n", "20"])
    finally:
        path.unlink()
    assert "pass the contract" in capsys.readouterr().out


def test_grouped_metrics_expand_and_count_in_budget():
    c = Contract.load(EX / "property_prediction" / "contract.yaml")
    assert [m.name for m in c.metrics] == ["mae[solubility]", "mae[logD]", "mae[clearance]",
                                           "spearman_macro"]
    single = Contract.load(EX / "msms_peaks" / "contract.yaml")
    assert leakage.budget(c).metric_bits > 3 * leakage.budget(single).metric_bits


def test_vector_labels_file_published_inline():
    pub = Contract.load(EX / "perturbation" / "contract.yaml").public()
    delta = next(f for f in pub["outputs"] if f["name"] == "delta")
    assert delta["length"] == 20 and delta["labels"][:2] == ["G01", "G02"]


# ----------------------------------------------------------- invalid outputs

def _bundle(tmp_path, body: str) -> Path:
    d = tmp_path / "b"
    d.mkdir(exist_ok=True)
    (d / "predict.py").write_text("import pandas as pd\nimport numpy as np\n" + body)
    return d


@pytest.mark.parametrize("task,body,reason", [
    ("perturbation", "def predict(df):\n    return pd.DataFrame({'delta': [[0.0]*19]*len(df)})\n",
     "wrong_length"),
    ("perturbation", "def predict(df):\n    return pd.DataFrame({'delta': [[np.nan]*20]*len(df)})\n",
     "non_finite"),
    ("msms_peaks", "def predict(df):\n    return pd.DataFrame({'peaks': [{'mz': [1.0]*101, "
     "'intensity': [0.1]*101}]*len(df)})\n", "too_many_items"),
    ("msms_peaks", "def predict(df):\n    return pd.DataFrame({'peaks': [{'mz': [1.0, 2.0], "
     "'intensity': [0.1]}]*len(df)})\n", "wrong_length"),
    ("tf_ranking", "def predict(df):\n    return pd.DataFrame({'ranking': [{'tf': ['NOT_A_CANDIDATE'], "
     "'score': [1.0]}]*len(df)})\n", "not_in_candidates"),
    ("ehr_outcome", "def predict(df):\n    return pd.DataFrame({'risk': [[0.1, 0.2, 1.5]]*len(df)})\n",
     "above_max"),
])
def test_bad_structured_outputs_rejected(node, tmp_path, task, body, reason):
    cfg, _ = node(task)
    p = evaluate(cfg, task, _bundle(tmp_path, body))
    assert p["status"] == "error" and p["error"]["code"] == "predictions_invalid", p
    assert p["error"]["detail"]["reason"] == reason


# ------------------------------------------------------ provider data errors

def test_folder_task_rejects_single_file(tmp_path):
    c = Contract.load(EX / "he_spatial" / "contract.yaml")
    (tmp_path / "x.csv").write_text("a\n1\n")
    with pytest.raises(DataError, match="needs a folder"):
        prepare(c, tmp_path / "x.csv", tmp_path / "out")


def test_missing_pass_through_file_explained(tmp_path):
    c = Contract.load(EX / "ehr_outcome" / "contract.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_ehr(raw)
    (raw / "events.parquet").unlink()
    with pytest.raises(DataError, match="should contain 'events.parquet'"):
        prepare(c, raw, tmp_path / "out")


@pytest.mark.parametrize("bad", ["../outside.npy", "/etc/passwd", "spots.csv", "patches/nope.npy"])
def test_file_column_cannot_escape_or_expose_answers(tmp_path, bad):
    """A file column may only name files inside the data folder, and never the
    samples table itself (which holds the answers)."""
    c = Contract.load(EX / "he_spatial" / "contract.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_he(raw)
    (tmp_path / "outside.npy").write_bytes(b"x")
    df = pd.read_csv(raw / "spots.csv")
    df.loc[0, "patch"] = bad
    df.to_csv(raw / "spots.csv", index=False)
    with pytest.raises(DataError, match="missing or outside the data folder"):
        prepare(c, raw, tmp_path / "out")


def test_vector_answer_from_wide_columns_explained_when_missing(tmp_path):
    c = Contract.load(EX / "perturbation" / "contract.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_perturbation(raw)
    pd.read_csv(raw / "queries.csv").drop(columns=["G05"]).to_csv(raw / "queries.csv", index=False)
    with pytest.raises(DataError, match="no column 'delta_true' \\(or one column per label: G01"):
        prepare(c, raw, tmp_path / "out")


# ------------------------------------------- answers in any format (private)

POP_CONTRACT = """
id: perturb_pop
version: 1
title: Perturbation, cell-population track
description: Predict the mean expression shift; scored against raw perturbed cells.
key: [query_id]
data:
  samples: queries.csv
  include:
    - {path: control.npy, shape: [50, 4], description: control cells x genes}
  private:
    - {path: perturbed, description: "one .npy of perturbed cells x genes per query"}
inputs:
  - {name: query_id, type: string}
  - {name: gene, type: string}
outputs:
  - {name: shift, type: vector, length: 4, min: -50, max: 50}
answers:
  - {name: query_id, type: string}
metrics:
  - {name: mae, type: float, min: 0, max: 50, goal: min}
headline: mae
"""

POP_SCORER = """
import numpy as np

def score(df, answers_dir):
    errs = []
    for q, shift in zip(df["query_id"], df["shift"]):
        cells = np.load(answers_dir / "perturbed" / f"{q}.npy")
        errs.append(np.abs(cells.mean(axis=0) - shift).mean())
    return {"mae": min(float(np.mean(errs)), 50.0)}
"""


def _pop_task(tmp_path):
    task = tmp_path / "node" / "tasks" / "perturb_pop"
    task.mkdir(parents=True)
    (task / "contract.yaml").write_text(POP_CONTRACT)
    (task / "score.py").write_text(POP_SCORER)
    raw = tmp_path / "raw"
    (raw / "perturbed").mkdir(parents=True)
    pd.DataFrame({"query_id": [f"q{i}" for i in range(10)], "gene": "TP53"}).to_csv(
        raw / "queries.csv", index=False)
    np.save(raw / "control.npy", RNG.random((50, 4)))
    for i in range(10):
        np.save(raw / "perturbed" / f"q{i}.npy", RNG.random((30, 4)) + i)
    return task, raw


def test_scorer_only_files_reach_scorer_never_model(tmp_path):
    task, raw = _pop_task(tmp_path)
    c = Contract.load(task / "contract.yaml")
    inputs_dir, answers, _ = prepare(c, raw, tmp_path / "node" / "private_data" / "perturb_pop")
    answers_dir = answers.parent / "answers_files"
    assert (answers_dir / "perturbed" / "q3.npy").exists()
    assert not any("perturbed" in str(p) for p in inputs_dir.rglob("*"))
    root = tmp_path / "node"
    (root / "config.yaml").write_text(json.dumps({
        "sandbox": {"backend": "local", "i_understand_local_backend_is_not_isolated": True},
        "tasks": {"perturb_pop": {"dir": "tasks/perturb_pop",
                                  "inputs": str(inputs_dir / "samples.parquet"),
                                  "answers": str(answers), "inputs_dir": str(inputs_dir),
                                  "answers_dir": str(answers_dir)}}}))
    cfg = Config.load(root / "config.yaml")
    b = _bundle(tmp_path, "def load_model(d):\n    global C\n    C = np.load(d / 'control.npy').mean(0)\n"
                          "def predict(df):\n    return pd.DataFrame({'shift': [list(C)] * len(df)})\n")
    p = evaluate(cfg, "perturb_pop", b)
    assert p["status"] == "ok" and set(p["metrics"]) == {"mae"}, p
    # A model probing for the scorer-only files finds nothing.
    spy = _bundle(tmp_path, "import os\ndef predict(df):\n    d = os.environ['BMB_INPUTS_DIR']\n"
                            "    found = [f for _, _, fs in os.walk(d) for f in fs if f.startswith('q')]\n"
                            "    assert not found, found\n"
                            "    return pd.DataFrame({'shift': [[0.0]*4] * len(df)})\n")
    assert evaluate(cfg, "perturb_pop", spy)["status"] == "ok"


def test_file_column_cannot_point_at_scorer_only_file(tmp_path):
    task, raw = _pop_task(tmp_path)
    text = POP_CONTRACT.replace("  - {name: gene, type: string}",
                                "  - {name: gene, type: string}\n  - {name: f, type: file}")
    (task / "contract.yaml").write_text(text)
    df = pd.read_csv(raw / "queries.csv")
    df["f"] = "perturbed/q1.npy"
    df.to_csv(raw / "queries.csv", index=False)
    with pytest.raises(DataError, match="missing or outside"):
        prepare(Contract.load(task / "contract.yaml"), raw, tmp_path / "out")


def test_many_rows_can_share_one_file(tmp_path):
    c = Contract.load(EX / "he_spatial" / "contract.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_he(raw)
    df = pd.read_csv(raw / "spots.csv")
    df["patch"] = "patches/s0.npy"            # every spot points at the same image
    df.to_csv(raw / "spots.csv", index=False)
    inputs_dir, _, n = prepare(c, raw, tmp_path / "out")
    assert n == N and (inputs_dir / "files" / "patches" / "s0.npy").exists()

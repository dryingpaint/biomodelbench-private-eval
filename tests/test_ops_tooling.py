"""Operator tooling: check-task, the disk and cache defaults."""
from __future__ import annotations

import collections
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bmb_eval import cli, config as config_mod, taskcheck
from bmb_eval.config import Config

EX = Path(__file__).resolve().parents[1] / "examples" / "tasks"
PROPERTY = EX / "property_prediction"


# ------------------------------------------------------------------ check-task

@pytest.mark.parametrize("task", sorted(p.name for p in EX.iterdir() if (p / "contract.yaml").exists()))
def test_check_task_passes_every_example(task, capsys):
    assert taskcheck.check(EX / task) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "check-task passed" in out and "random predictions" in out and "bits/submission" in out


def _draft(tmp_path, score=None, contract=None) -> Path:
    d = tmp_path / "draft"
    shutil.copytree(PROPERTY, d, ignore=shutil.ignore_patterns("bundle"))
    if score:
        (d / "score.py").write_text(score((d / "score.py").read_text()))
    if contract:
        (d / "contract.yaml").write_text(contract((d / "contract.yaml").read_text()))
    return d


def test_check_task_names_metrics_the_contract_doesnt_declare(tmp_path, capsys):
    # The draft ADMET task: the scorer prefixed each group with its source, `per:` didn't,
    # and the version was a string. Both were found only at the first real submission.
    d = _draft(tmp_path, score=lambda s: s.replace('f"mae[{e}]"', 'f"mae[source_a|{e}]"'),
               contract=lambda s: s.replace("version: 1", 'version: "2.0"'))
    with pytest.raises(SystemExit) as e:
        cli.main(["check-task", str(d)])
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "version must be written as an integer" in out and "'2.0'" in out
    assert "3 metric(s) emitted but not declared: mae[source_a|clearance], mae[source_a|logD]" in out
    assert "3 declared metric(s) not emitted: mae[clearance], mae[logD], mae[solubility]" in out
    assert "check-task: " in out and "problem(s)" in out


def test_check_task_finds_a_scorer_that_skips_groups_without_rows(tmp_path, capsys):
    d = _draft(tmp_path, score=lambda s: s.replace("for e in ENDPOINTS:",
                                                   "for e in df['endpoint'].unique():"))
    assert taskcheck.check(d) > 0
    out = capsys.readouterr().out
    assert "ok    random predictions" in out              # every group present: fine
    assert "FAIL  no rows with endpoint=" in out and "declared metric(s) not emitted" in out


def test_check_task_finds_nulls_and_out_of_bounds_values(tmp_path, capsys):
    d = _draft(tmp_path, score=lambda s: s.replace(
        "min(float(np.mean(np.abs(y - p))), 100.0) if len(g) else 0.0",
        "float(np.mean(np.abs(y - p))) if len(g) else None"))
    assert taskcheck.check(d) > 0
    out = capsys.readouterr().out
    assert "outside the declared bounds: mae[" in out             # random predictions: huge errors
    assert "None but not declared nullable: mae[" in out          # a group with no rows


def test_check_task_explains_a_contract_that_doesnt_load(tmp_path, capsys):
    d = _draft(tmp_path, contract=lambda s: s.replace("headline: spearman_macro\n", ""))
    assert taskcheck.check(d) == 1
    assert "contract.yaml is missing 'headline'" in capsys.readouterr().out
    (d / "contract.yaml").write_text("id: [unclosed\n")
    assert taskcheck.check(d) == 1
    assert "isn't valid YAML" in capsys.readouterr().out


def test_check_task_uses_real_shaped_rows_when_given(tmp_path, capsys):
    # A scorer that accepts only real combinations of inputs fails on synthetic rows,
    # and says to pass --data; with the provider-style rows it passes.
    d = _draft(tmp_path, score=lambda s: s.replace(
        "def score(df):\n", "def score(df):\n    assert set(df['assay'].dropna()) <= {'A1'}\n"))
    assert taskcheck.check(d) > 0
    assert "pass --data with a few real-shaped rows" in capsys.readouterr().out
    rng = np.random.default_rng(0)
    pd.DataFrame({"sample_id": [f"s{i}" for i in range(40)], "smiles": "CCO",
                  "endpoint": [["solubility", "logD", "clearance"][i % 3] for i in range(40)],
                  "assay": "A1", "measured": rng.normal(size=40)}).to_csv(tmp_path / "rows.csv", index=False)
    cli.main(["check-task", str(d), "--data", str(tmp_path / "rows.csv")])
    out = capsys.readouterr().out
    assert "40 rows from" in out and "check-task passed" in out


def test_quoted_whole_number_versions_still_load(tmp_path):
    from bmb_eval.contract import Contract
    d = _draft(tmp_path, contract=lambda s: s.replace("version: 1", 'version: "3"'))
    assert Contract.load(d / "contract.yaml").version == 3
    (d / "contract.yaml").write_text((d / "contract.yaml").read_text().replace('"3"', '"2.0"'))
    with pytest.raises(ValueError, match="version must be a whole number"):
        Contract.load(d / "contract.yaml")


# ------------------------------------------------------------------ disk and cache

Usage = collections.namedtuple("Usage", "total used free")


def test_connect_still_refuses_a_nearly_full_disk(tmp_path, monkeypatch):
    # conftest turns the check off for the suite; the production threshold is unchanged.
    from bmb_eval.contract import Limits
    monkeypatch.setattr(cli, "MIN_FREE_GB", 20)
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda p: Usage(500 << 30, 490 << 30, 10 << 30))
    assert any("Less than 20 GB of disk is free" in p for p in cli.hardware_problems(Limits(), tmp_path))
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda p: Usage(500 << 30, 400 << 30, 100 << 30))
    assert cli.hardware_problems(Limits(), tmp_path) == []


def _node_config(tmp_path, **extra) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(json.dumps({"node_name": "t", "tasks": {},
                             "sandbox": {"backend": "local", "i_understand_local_backend_is_not_isolated": True},
                             **extra}))
    return p


def test_cache_defaults_to_half_a_small_disk_and_keeps_explicit_settings(tmp_path, monkeypatch):
    seen = []

    def disk(p):
        seen.append(Path(p))
        return Usage(290 << 30, 0, 290 << 30)
    monkeypatch.setattr(config_mod.shutil, "disk_usage", disk)
    assert Config.load(_node_config(tmp_path)).cache_gb == 145          # not 300 on a 290 GB disk
    assert seen[-1] == tmp_path                                         # state/blobs doesn't exist yet
    monkeypatch.setattr(config_mod.shutil, "disk_usage", lambda p: Usage(2000 << 30, 0, 2000 << 30))
    assert Config.load(_node_config(tmp_path)).cache_gb == 300
    assert Config.load(_node_config(tmp_path, cache_gb=500)).cache_gb == 500


def test_check_warns_about_a_cache_bigger_than_its_disk(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda p: Usage(290 << 30, 0, 290 << 30))
    with pytest.raises(SystemExit):                     # the local backend is an error
        cli.main(["--config", str(_node_config(tmp_path, cache_gb=300)), "check"])
    assert "WARN  cache_gb is 300 but the disk holding" in capsys.readouterr().out

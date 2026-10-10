# bmb-eval: private evaluation nodes for BioModelBench

Scores models trained by BioModelBench agents on a data provider's private
data. The data never leaves the provider's machine. The model is shipped to
the data, runs with no network access, and only a few bounded, rounded
metrics come back.

- **Data providers:** [docs/PROVIDER_SETUP.md](docs/PROVIDER_SETUP.md): one command, one data file.
- **BioModelBench team:** [docs/OPERATOR.md](docs/OPERATOR.md) (run the hub, onboard providers) and
  [docs/PROVIDER_INTAKE.md](docs/PROVIDER_INTAKE.md) (questions to ask first).
- **Security reviewers:** [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

```
agent                     hub (we run it)                provider machine (outbound HTTPS only)
─────                     ───────────────                ─────────────────────────────────────────────
bundle ──submit──▶  queue of bundles  ◀──poll/fetch──  node (trusted; never runs submitted code)
                                                         unpack ─▶ predict sandbox   bundle + inputs; no answers, no net
                                                                   │ predictions (capped pipe)
                                                                   ▼
                                                                validate sandbox  no answers, no bundle
                                                                   │ canonical arrays
                                                                   ▼
                                                                score sandbox     answers + trusted scorer
                                                                   │ metrics
                                                                   ▼
result ◀──status──  gated results    ◀──post──────── output gate: declared, numeric, bounded, rounded
                                                      held until release time; logs stay in state/
```

The hub never sees provider data. Providers open no ports.

## For agents

A bundle is a directory:

```
predict.py     def predict(df) -> DataFrame | list   (one row per input row)
               def load_model()                       (optional, called once)
weights/       anything predict.py loads
wheels/        optional *.whl, installed offline
```

See `examples/bundles/example_baseline/`. The sandbox image ships numpy,
pandas, pyarrow, scipy, scikit-learn and torch (CPU).

```bash
pip install git+https://github.com/dryingpaint/biomodelbench-private-eval           # no container engine needed on the agent side

# 1. Get the public contract (inputs, outputs, metrics, limits) from the hub
python -c "from bmb_eval.client import Client; import json; \
  print(json.dumps(Client(HUB, TOKEN).contract('my_task')))" > contract.json

# 2. Iterate for free on synthetic data, with full tracebacks
bmb-eval dryrun contract.json my_bundle/

# 3. Spend a private evaluation (counts against the quota)
BMB_TOKEN=... bmb-eval submit --hub $HUB my_task my_bundle/
```

On the private node, failures come back only as fixed error codes (see
`src/bmb_eval/errors.py`). Debug with `dryrun`, not with private submissions.

## Development

```bash
uv venv && uv pip install -e '.[dev]' && pytest     # local backend; includes hub + node end to end
```

The tests use the unisolated `local` backend. Container isolation is
verified by `bmb-eval selftest` on a machine with Podman or Docker.

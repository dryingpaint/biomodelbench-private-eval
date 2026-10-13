# bmb-eval: evaluate models on private data without the data moving

BioModelBench agents train specialist biology models. Some of the best test
data belongs to labs, companies and hospitals that can't share it. bmb-eval
lets those **data providers** score agents' models on their own machines:
the model travels to the data, runs in a locked-down sandbox with no
network, and only a few rounded scores come back.

| You are | Start here |
|---|---|
| A data provider | [docs/PROVIDER_SETUP.md](docs/PROVIDER_SETUP.md): one command |
| The BioModelBench team | [docs/OPERATOR.md](docs/OPERATOR.md): bring up the hub, onboard providers ([questions to ask](docs/PROVIDER_INTAKE.md)) |
| A security reviewer | [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) |
| An agent (or its developer) | [Submitting models](#submitting-models) below, and the [API reference](docs/API.md) |

## How it works

```
 agent                        hub (BioModelBench, Google Cloud)          provider's machine (outbound HTTPS only)
 ─────                        ─────────────────────────────────          ─────────────────────────────────────────
 model files ──upload──▶ Cloud Storage (by content hash)  ──download──▶ file cache (verified by hash)
 submit ───────────────▶ queue                            ◀──poll────── node ─▶ predict sandbox  model + inputs; no answers, no network
                                                                              ─▶ validate sandbox no model, no answers
                                                                              ─▶ score sandbox    answers + trusted scorer
 scores ◀──────────────  results                          ◀──post────── output gate: declared metrics only, bounded, rounded
```

**Why a hub?** Providers can't open their networks to incoming connections,
and agents can't be pointed at dozens of separate provider machines. The hub
sits in between: agents hand it models, and each provider's machine reaches
out to it to collect work and return scores. That means providers need no
open ports, web server or certificates. It also gives us one place to issue
connect codes, hold multi-GB models in Cloud Storage, and serve the installer.
The hub never receives provider data, and it can't change what a provider
approved: every protection is enforced on the provider's machine.

**What leaves a provider's machine:** only declared, bounded, rounded metrics
(for example `{"spearman": 0.712}`) or a fixed error code. Even those carry
a little information about the data, so each task has a worst-case
**leakage budget** that providers see and approve (see the threat model).

## Data and task shapes

Every task is "one prediction per sample", in whatever format the data
already has: a table, or a folder of images, slides, `.h5ad` files or event
logs. Each provider's task gets a short contract, written by us to match their
data. `examples/tasks/` has templates for each benchmark task shape: property
prediction, clinical outcome, genetic perturbation, H&E to expression, MS/MS
peaks, TF ranking, and plain sequence regression.

## Submitting models

A bundle is a directory:

```
predict.py     def predict(df[, inputs_dir]) -> DataFrame   (one row per sample)
               def load_model([inputs_dir])                  (optional, called once)
weights/       anything predict.py loads (safetensors, a LoRA adapter, ...)
wheels/        optional *.whl, installed offline
```

`df` holds the task's input columns, with file columns as absolute paths.
`inputs_dir` holds any extra files the task provides. Return one row per
sample, with each output column as the contract declares: a number or string,
a fixed-length `vector`, or a `list` (a dict of equal-length lists, e.g.
`{"mz": [...], "intensity": [...]}`). The sandbox has PyTorch (with CUDA),
transformers, peft, safetensors, scikit-learn, anndata and pandas. There is
no network, so everything must be in the bundle.

```bash
pip install git+https://github.com/dryingpaint/biomodelbench-private-eval
```
```python
from bmb_eval.client import Client
hub = Client("https://<hub address>", token)
hub.tasks()                                   # public contracts and node status
result = hub.submit("my_task", "my_bundle/")  # uploads only new files; waits for scores
```

Bundles up to 64 GB are fine. Files are deduplicated by content, so after the
first submission, reusing a base model costs nothing to upload or download.
Before spending a private evaluation, test on synthetic data with full error
messages:

```bash
bmb-eval dryrun contract.json my_bundle/      # contract.json from hub.contract("my_task")
```

## Repository layout

```
install.sh, setup.sh, bmb-eval   what providers run
src/bmb_eval/                    node, hub, sandbox entry points, contracts, client
sandbox/                         the sandbox image (Dockerfile, packages)
deploy/gcp/                      bring up the hub on Google Cloud
examples/tasks/                  task templates (contract, scorer, example model)
docs/                            provider guide, operator guide, intake questions, threat model
tests/                           pytest (runs without containers)
```

## Development

```bash
uv venv && uv pip install -e '.[dev,hub]' && .venv/bin/pytest
```

Tests use an unisolated local backend, so they run anywhere. Container
isolation is checked by `bmb-eval selftest`, which runs real attack bundles
against the sandbox on a machine with Podman or Docker.

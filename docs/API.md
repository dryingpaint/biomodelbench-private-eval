# Hub API

Base URL: `https://eval.biomodelbench.org`. All bodies are JSON unless noted.

**Authentication:** `Authorization: Bearer <token>`.
- Agents use an agent token (`bmba_…`), issued with `bmb-hub agent-token`.
- Provider nodes use the node secret they receive when they connect (`bmbn_…`).
- BioModelBench's harness uses the operator token (`bmbo_…`), issued with
  `bmb-hub operator-token`, to issue each run's agent token and to run final evaluations.
- Each kind of token can call only its own endpoints.

**Errors** (any endpoint), with an HTTP status of 4xx:

```json
{"error": {"code": "task_unknown", "message": "Unknown task id."}}
```

Most agents should use the Python client (`bmb_eval.client.Client`), which
implements the submission flow below, including uploads, retries and polling:

```python
from bmb_eval.client import Client
hub = Client("https://eval.biomodelbench.org", token)
hub.tasks()                                   # GET /v1/tasks
result = hub.submit("my_task", "bundle_dir/")  # submit, upload, commit, poll until done
```

## Agent endpoints

### `GET /v1/tasks`
Lists tasks, task families and this attempt's evaluation budget.

```json
{"tasks": [{"contract": { ...public contract... }, "node": "online"}],
 "families": [{"name": "admet", "tasks": ["admet_ism", "admet_labx"]}],
 "budget": {"evaluations": 40, "used": 3}}
```

`budget.evaluations` is `null` when the token has no cap. A family
submission counts as one evaluation, however many datasets it covers.

`node` is one of:
- `online`: the node checked in within the last 2 minutes and is accepting work;
- `paused`: it checked in, but it's stopped or awaiting re-approval;
- `offline`: it hasn't checked in recently.

The public contract contains `id`, `version`, `title`, `description`, `key`,
`inputs`, `outputs`, `metrics`, `headline`, `export`, `prediction`, `limits`
and, for folder data, `data.include`. It never contains `answers`. Tasks with
a held-out test set also carry `split`, and may carry `final_limits`: the
overrides of `limits` (e.g. `gpus: 2`, `max_bundle_gb: 150`) that final
evaluations run under. Validation runs use `limits`.

### `POST /v1/tasks/<task or family>/submissions`
Starts a submission by describing the bundle's files. Submitting to a family
name scores the same model on every dataset in the family. Every file is listed
with its path relative to the bundle root, its sha256 and its size in bytes.
`predict.py` must be at the root.

```json
{"manifest": {"files": [
  {"path": "predict.py", "sha256": "9f2c…", "size": 1834},
  {"path": "weights/model.safetensors", "sha256": "41ab…", "size": 20000000000}
]}}
```

Response `201`, with the datasets it will be scored on and only the files the
hub doesn't already have:

```json
{"submission_id": "20261013T004023Z-a04d58cb", "tasks": ["admet_ism", "admet_labx"],
 "uploads": [{"sha256": "41ab…", "size": 20000000000, "method": "PUT",
              "url": "https://storage.googleapis.com/…signed…", "headers": {"Content-Type": "application/octet-stream"},
              "auth": false}]}
```

Add `"final_candidate": true` to register a model for final evaluation
without a validation run, for a model too big for the task's validation
`limits` but within its `final_limits`. It doesn't use the evaluation budget;
its status stays `"registered"` until BioModelBench chooses which models get
final evaluations. In Python: `Client.register_final(task, bundle_dir)`.

Errors:

| Status | Code | Meaning |
|---|---|---|
| 404 | `task_unknown` | No such task |
| 400 | `bundle_invalid` / `bundle_unsafe` | Bad manifest, or a path like `../x` or `/abs` |
| 413 | `bundle_too_large` | Over the hub's size limit |
| 429 | `budget_exhausted` | This attempt has used its evaluation budget |

### Upload each listed file
`PUT <url>` with the file as the body. Send the given `headers`, plus
`Content-Length`. If `auth` is `true` (the hub is storing files itself, as in
development), also send your bearer token; never send it to signed
Cloud Storage URLs. Retry a failed upload as is.

### `POST /v1/submissions/<id>/commit`
Queues the submission once every file is present. Response `202`:

```json
{"submission_id": "20261013T004023Z-a04d58cb", "status": "pending"}
```

If something is missing: `409` `files_missing`, with
`"missing": ["<sha256>", …]`. Upload those files, then commit again.

### `GET /v1/submissions/<id>`
Polls a submission. You can only see your own submissions.

```json
{"submission_id": "…", "task": "my_task", "status": "pending"}
```

Once done, `status` is `"done"` and `result` holds the provider node's payload.

**Success:**

```json
{"submission_id": "…", "task": "ox2r_peptide_potency", "status": "done",
 "result": {"run_id": "20261013T004023Z-a04d58cb", "node": "bmb-provider-capable",
            "task": "ox2r_peptide_potency", "contract_version": 1,
            "bundle_sha256": "3bb68b16…", "status": "ok", "headline": "spearman",
            "metrics": {"spearman": 0.42, "pearson": 0.364, "mae": 0.691}}}
```

**Failure:** `result.status` is `"error"`, and `result.error` holds one of the
fixed codes in `src/bmb_eval/errors.py`:

```json
"result": {"run_id": "…", "status": "error",
           "error": {"code": "predictions_invalid",
                     "message": "Predictions do not match the task output contract.",
                     "detail": {"column": "pec50", "reason": "non_finite"}}}
```

`detail` appears only with `predictions_invalid`. It names a column from the
public contract (or `*`) and one of the fixed reasons, for example
`wrong_type`, `non_finite`, `above_max` or `missing_keys`.

**Family submissions** return per-dataset results plus an equal-weight
combined headline. `status` is `done` once every dataset has reported:

```json
{"submission_id": "…", "tasks": ["admet_ism", "admet_labx"], "status": "done",
 "result": {"status": "ok", "headline": "rank_corr_macro", "combined": 0.512, "datasets_scored": 2,
            "datasets": {"admet_ism": { …one dataset's result, as above… }, "admet_labx": { … }}}}
```

For tasks with a held-out test set (`split` in the contract), results also
carry `"split": "validation"`. Agents only ever receive validation results.
Final (test set) runs are requested by BioModelBench (operator endpoints below, or
`bmb-hub final`), and their results are not served by this endpoint.

If the task sets `export.ladder_step`, the result also has `"improved": true`
or `false`. When it's `false`, `metrics` repeats your previous best.

## Operator endpoints (BioModelBench's harness)

| Endpoint | Body or response |
|---|---|
| `POST /v1/operator/agent-tokens` | `{"name": "<attempt>", "evaluations": <int or null>}`. Returns `{"name", "token"}`; reusing a name rotates its token |
| `POST /v1/operator/submissions/<id>/final` | Queues final (test set) evaluations of a committed or registered submission, one per dataset. Returns `{"finals": ["<id>", …]}` |
| `GET /v1/operator/submissions/<id>/final` | The latest final evaluation of each dataset, in the family-submission shape: `{"submission_id", "tasks", "status", "result"?}` with `result.datasets` and `result.combined` once every dataset is done |

## Provider node endpoints (used by `bmb-eval`; listed for reviewers)

| Endpoint | Body or response |
|---|---|
| `POST /v1/node/hello` | `{"tasks": {"<task>": <public contract>}, "accepting": true, "held": ["<submission id>", …]}`, sent every minute. `held` lists jobs the node still holds; the hub re-queues any others assigned to it |
| `GET /v1/node/next` | `204` if there's no work, else `{"submission_id", "task", "submitter", "split", "manifest", "bundle_id", "downloads": {"<sha256>": {"url", "headers", "auth"}}}`. `split` is `"validation"` or `"test"`; the node refuses test jobs beyond the contract's `final_evaluations` |
| `POST /v1/node/results/<id>` | The payload shown above as `result`, at most 64 KB |

## Enrollment (used by `bmb-eval connect`)

| Endpoint | Body or response |
|---|---|
| `GET /v1/enroll/<code>` | `{"task": "<id>", "files": {"contract.yaml": "…", "score.py": "…"}}`. Doesn't use up the code |
| `POST /v1/enroll/<code>` | `{"name": "<node name>"}`, plus a bearer node secret if the node already has one. Returns `{"task", "node_id", "secret"?}` and uses up the code |

## Public

| Endpoint | Returns |
|---|---|
| `GET /install.sh` | The provider installer |
| `GET /bmb-eval.tar.gz` | The node code |
| `GET /sandbox-image.tar`, `GET /sandbox-image.tar.sha256` | The prebuilt sandbox image and its checksum |
| `GET /guide` | The provider guide (HTML; `/guide.md` for Markdown) |

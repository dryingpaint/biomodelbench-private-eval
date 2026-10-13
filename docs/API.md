# Hub API

Base URL: `https://eval.biomodelbench.org`. All bodies are JSON unless noted.

**Authentication:** `Authorization: Bearer <token>`.
- Agents use an agent token (`bmba_…`), issued with `bmb-hub agent-token`.
- Provider nodes use the node secret they receive when they connect (`bmbn_…`).
- An agent token can't call node endpoints, and vice versa.

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
Lists tasks and whether their provider is online.

```json
{"tasks": [{"contract": { ...public contract... }, "node": "online"}]}
```

`node` is one of:
- `online`: the node checked in within the last 2 minutes and is accepting work;
- `paused`: it checked in, but it's stopped or awaiting re-approval;
- `offline`: it hasn't checked in recently.

The public contract contains `id`, `version`, `title`, `description`, `key`,
`inputs`, `outputs`, `metrics`, `headline`, `export`, `prediction`, `limits`
and, for folder data, `data.include`. It never contains `answers`.

### `POST /v1/tasks/<task>/submissions`
Starts a submission by describing the bundle's files. Every file is listed
with its path relative to the bundle root, its sha256 and its size in bytes.
`predict.py` must be at the root.

```json
{"manifest": {"files": [
  {"path": "predict.py", "sha256": "9f2c…", "size": 1834},
  {"path": "weights/model.safetensors", "sha256": "41ab…", "size": 20000000000}
]}}
```

Response `201`, listing only the files the hub doesn't already have:

```json
{"submission_id": "20261013T004023Z-a04d58cb",
 "uploads": [{"sha256": "41ab…", "size": 20000000000, "method": "PUT",
              "url": "https://storage.googleapis.com/…signed…", "headers": {"Content-Type": "application/octet-stream"},
              "auth": false}]}
```

Errors:

| Status | Code | Meaning |
|---|---|---|
| 404 | `task_unknown` | No such task |
| 400 | `bundle_invalid` / `bundle_unsafe` | Bad manifest, or a path like `../x` or `/abs` |
| 413 | `bundle_too_large` | Over the hub's size limit |

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

If the task sets `export.ladder_step`, the result also has `"improved": true`
or `false`. When it's `false`, `metrics` repeats your previous best.

## Provider node endpoints (used by `bmb-eval`; listed for reviewers)

| Endpoint | Body or response |
|---|---|
| `POST /v1/node/hello` | `{"tasks": {"<task>": <public contract>}, "accepting": true, "held": ["<submission id>", …]}`, sent every minute. `held` lists jobs the node still holds; the hub re-queues any others assigned to it |
| `GET /v1/node/next` | `204` if there's no work, else `{"submission_id", "task", "submitter", "manifest", "bundle_id", "downloads": {"<sha256>": {"url", "headers", "auth"}}}` |
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

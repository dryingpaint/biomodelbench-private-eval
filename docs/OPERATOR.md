# Operating BioModelBench evaluation (BioModelBench team)

## Why we run a hub

Providers' machines can't accept incoming connections, and agents can't be
pointed at dozens of separate provider machines. The hub sits in between:
agents give it models, and each provider's node reaches out to collect work
and return scores. So providers open no ports, and we have one place to issue
connect codes, keep multi-GB models (in Cloud Storage), and serve the
installer. It never receives provider data, and it can't change what a
provider approved.

## Bring up the hub (once, about 30 minutes)

Needs: the `gcloud` CLI logged in to our Google Cloud project. A domain is
optional.

```bash
git clone https://github.com/dryingpaint/biomodelbench-private-eval && cd biomodelbench-private-eval
deploy/gcp/hub-setup.sh                     # or: deploy/gcp/hub-setup.sh eval.<our-domain>
```

This creates:
- a Cloud Storage bucket for model files (each kept 30 days);
- a service account that can sign download links;
- a static IP and HTTPS firewall rule;
- an `e2-standard-2` VM that installs the hub on boot, builds the sandbox image
  providers download, and runs behind Caddy (which gets the HTTPS certificate
  automatically).

Without a domain, the hub is reachable at `https://<ip-with-dashes>.sslip.io`,
a free DNS name that points at the IP, with a real certificate. With a domain,
add the DNS record the script prints.

About 15 minutes after the first boot, `curl -fsS https://<hub>/install.sh |
head -3` prints the installer.

**Cost:** the VM is about $50/month. Storage and downloads are a few cents per
GB, and caching on providers' machines means a shared base model is downloaded
once per provider.

**Updating:** push to the repo, then run `deploy/gcp/hub deploy` (or `deploy/gcp/hub deploy
v0.2.0` for a release tag; see Releases below). It pulls the code on
the VM, installs it, rebuilds the sandbox image if it changed and loads the new code into
memory, all while the running hub keeps serving; then it restarts the hub, which takes about
a second. Caddy holds requests during that second rather than failing them, and nodes keep
running their evaluations throughout. Don't reboot the VM to deploy code: a boot repeats all
of that work with the hub down, minutes on this disk. Reboot only for OS or startup-script
changes, gracefully (`gcloud compute ssh bmb-hub --zone us-central1-a -- sudo reboot`; avoid
`instances reset`, a hard power-cycle that can lose recent writes).
Providers get updates by re-running the install command without arguments;
their node pauses until they approve the new version.

**Backups:** `/home/bmb/hub_state/hub.db` on the VM (tokens, codes, queue,
results). Snapshot the VM's disk on a schedule.

## Releases: one tag for the three downloads

Providers download three things from the hub, and they have to match: `install.sh`,
`bmb-eval.tar.gz` (the evaluator, built from the checkout when the hub starts) and
`sandbox-image.tar` (built by `hub-deploy.sh`, stamped with the digest of the sandbox code
and its lock). A release is one git tag naming the commit all three come from:

```bash
deploy/release.sh v0.2.0 --push     # sets the version in pyproject.toml, commits, tags, pushes
deploy/gcp/hub deploy v0.2.0        # the hub checks out exactly that tag; rebuilds what changed
curl -fsS https://<hub>/release     # the tag, the commit, and the sha256 of each download
```

`/release` also says whether the published image was built from the code the hub runs
(`files."sandbox-image.tar".current`); `deploy/gcp/hub status` prints the same line.
`/SHA256SUMS` has the same checksums in `sha256sum -c` form. Give partners the tag and
that page: the hashes don't change until the next release, because every input is pinned:

- the evaluator tarball is built without timestamps or owners, so one commit gives one file;
- the sandbox image pins its base image by digest, Debian packages by snapshot date, and
  Python packages by version and hash (`sandbox/requirements.lock`). To change packages,
  edit `sandbox/requirements.txt`, run `sandbox/lock.sh` and commit both; the lock is part
  of the sandbox stamp, so the next deploy rebuilds and republishes the image.

`deploy/gcp/hub deploy` without a tag deploys main's head and goes back to following main;
`/release` then shows something like `v0.2.0-3-g1a2b3c4` (three commits past the tag). Fine
for trying things, not what to hand a partner.

## Admin commands

Run from your laptop; they execute on the hub VM:

```bash
deploy/gcp/hub agent-token attempt-17 --evaluations 40   # one token per attempt, with its evaluation budget
deploy/gcp/hub enroll examples/tasks/sequence_regression  # prints the provider email
deploy/gcp/hub family set admet admet_ism admet_labx      # group datasets for one run
deploy/gcp/hub final <submission_id>                      # held-out test runs (every dataset, for a family)
deploy/gcp/hub results <task or family>                   # all scores, including finals
deploy/gcp/hub revoke <code>                              # cancel an unused connect code
deploy/gcp/hub retire <task> [--yes]                      # remove a task (dry run without --yes; backs up hub.db)
deploy/gcp/hub status                                     # nodes, tasks, queue, codes, setup problems
deploy/gcp/hub alerts --webhook https://hooks.slack.com/… # where setup alerts go (--test to try it)
```

## Onboard a provider

1. **Ask the five questions** in `docs/PROVIDER_INTAKE.md`.
2. **Write their task.** Copy the closest template from `examples/tasks/` into
   a new folder, and edit `contract.yaml` and `score.py` to match their data.
   Each template has a contract, a scorer and a working example model:

   | Benchmark task | Template | What it shows |
   |---|---|---|
   | Property prediction, toxicity, VEP | `property_prediction` | One value per row; one metric per endpoint |
   | Clinical outcome (EHR), tissue state from blood | `ehr_outcome` | Folder with a patients table plus an events table; risk vector per patient |
   | Genetic perturbation | `perturbation` | Gene-panel vector (`genes.txt`); answers as one column per gene; control matrix passed through |
   | H&E to transcriptomics | `he_spatial` | One image file per spot; expression vector per spot |
   | MS/MS peak prediction | `msms_peaks` | Variable-length peak list per spectrum |
   | TF prediction | `tf_ranking` | Ranked list restricted to each query's candidates |
   | Anything with one table and one target | `sequence_regression` | The simplest case |

   The contract language (documented at the top of `src/bmb_eval/contract.py`):
   - Field types: `string`, `int`, `float`, `bool`, `vector` (fixed length or
     `labels`), `list` (records, `max_items`), and for inputs only `file` and
     `any`.
   - `data.samples` names the samples table in a folder; `data.include` lists
     files the model gets; `data.private` lists files only the scorer gets
     (answers in any format).
   - `per` on a metric reports it per group.
   - `limits` sets time, memory, GPUs (`gpus`, `gpu_memory_gb`), bundle size
     and evaluation caps.
3. **Check it with fake data:** save the public contract (`Contract.load(...).public()`
   as JSON), then run `bmb-eval dryrun contract.json <template>/bundle`.
4. **Enroll:** `deploy/gcp/hub enroll <task folder>` prints the email. Send it.
   The whole folder is shipped, subfolders and binary files included (e.g. a
   reference `.npy` the scorer loads), up to 16 MB. The example `bundle/`,
   hidden files and `__pycache__` are left out.
   **Each task has one code for good.** It works only for the first machine that
   connects with it, and that machine can reuse it for re-runs and updates.
5. **Watch it come online:** `deploy/gcp/hub status`. If the provider's command
   fails, it tells the hub which step failed and why, from a fixed list (for
   example `cloud_setup: gcloud_refused`, `data_check: data_problem`,
   `review: declined`), never data or error text. `status` shows it and the
   alert webhook gets it right away. The connect command counts as finished only
   once the provider's background service has checked in with the hub (or, sealed,
   once the upload is in); a service that can't start or can't get out is reported
   as `start_service: ...` on the spot. A code that was used but whose setup still
   hasn't finished an hour later shows as STALLED and alerts once, with what to
   ask the provider for (`~/bmb-eval/bmb-eval status`, `journalctl --user -u
   bmb-eval -n 30`). The hub's journal records each activation and each node's
   first check-in. Either way, the fix is usually to ask the provider for the
   message they saw and have them run the same command again.

**Changing a task** never needs a new code. Edit the files and run `enroll`
again:
- **Before the provider connects:** the same code now delivers the new version.
- **After they connect:** their machine shows "update available" and raises an
  alert if they've set one up. They review and approve it with
  `~/bmb-eval/bmb-eval update`, and until then keep evaluating with the
  version they approved.

The hub can't change a provider's task without that approval. To retire a
code nobody has used, run `deploy/gcp/hub revoke <code>`. To remove a task
altogether (its code, node binding, unfinished submissions and family
membership; finished results stay), run `deploy/gcp/hub retire <task>` to see
what would go, then again with `--yes`.

## Settings worth deciding per task

| Setting | Default | Trade-off |
|---|---|---|
| `limits.submissions_total` | 200 | Lifetime cap per submitter (one agent run's token). Each run's leakage scales with it; all runs together aren't capped |
| `limits.submissions_per_day` | 20 | Per agent |
| `limits.release_after_s` | 0 | Setting it at or above the total timeout removes the timing channel, but slows feedback |
| `limits.gpus`, `gpu_memory_gb` | 0 | Large models need a GPU at the provider; connect checks the GPU is big enough |
| `limits.max_bundle_gb` | 64 | Provider needs about 2× this much free disk |
| `final_limits` | none | Overrides of `limits` for final evaluations only, e.g. `{gpus: 2, gpu_memory_gb: 80, max_bundle_gb: 150}`. Validation stays on CPU; connect only warns if the GPUs are missing, and the node refuses finals (`final_hardware_missing`, not counted) until they're there. Needs `split`. The hub's `--max-bundle-gb` (default 160) must be at least the largest `max_bundle_gb` |
| `export.decimals` | 3 | Each decimal ≈ 3.3 bits per metric per evaluation |
| `export.ladder_step` | off | Turn on for "final" tasks |
| metric `per: [...]` | none | Every group adds a full metric to the leakage budget |

| metric `final_only: true` | off | Reported only in final evaluations. Use it for absolute metrics (MAE, RMSE, calibration): agents' public training data comes from other labs' protocols, so absolute errors mostly measure the offset, and repeated feedback lets agents fit it. Needs `split` |
| `split` | off | Optional held-out test set, e.g. `{validation_fraction: 0.5, final_evaluations: 3}` (`bmb-hub new-task --split 0.5` writes it). Worth it when agents get many evaluations on one task, since repeated feedback can overfit the scored rows. Needed by `final_only` and `final_limits` |
| `split.baseline_final_evaluations` | 0 | Test runs of our own reference models (baseline tokens), capped separately from the agents' `final_evaluations`; counted in the leakage budget |

## One run, many datasets: families and attempt budgets

Each provider dataset is its own task, on its own provider machine. To run
one attempt across several datasets:

1. **Give the datasets the same contract format** (same inputs and outputs),
   so one model works on all of them. Each provider's task may cover only the
   endpoints they measured. For ADMET, `make_task.py <task_id> --endpoints ...`
   generates such a per-provider task.
2. **Group them:** `bmb-hub family set admet admet_ism admet_labx`.
3. **Give the attempt one token with one budget:**
   `bmb-hub agent-token attempt-17 --evaluations 40`.

An agent submits to `admet` exactly as it would to a task:
- The hub sends that one submission to every dataset's provider. Model files
  upload once.
- It counts as **one** evaluation against the attempt's budget. Submissions to
  a single task count too.
- The result has each dataset's scores, plus `combined`: the mean of the
  datasets' headline metric, equal weight per dataset.
- `bmb-hub final <submission>` on a family submission runs every dataset's
  held-out test half.

Providers' own caps still apply on their machines. `submissions_total` counts
per submitter, so each attempt (agent run) gets its own; size it for one
attempt's budget plus its final scoring, e.g. 60 for a 40-evaluation budget.
Independent attempts aren't capped together: what all of them can learn grows
with the number of attempts, so tell providers how many you plan.

## Held-out test set and final evaluations

With `split` in the contract:
- **The split:** when the provider connects, their node randomly splits the
  rows into validation and test, with a secret seed. Nobody, including us,
  knows which row is where.
- **During the benchmark:** agents only see, and are only scored on, the
  validation rows.
- **At the end:** request the final score for the submission each agent
  nominates, then read the results:

```bash
$HUB_ADMIN final <submission_id>      # queues a test-set run of that model
$HUB_ADMIN results <task>             # validation and final scores; final ones are visible only here
```

The provider's node allows at most `final_evaluations` test runs per task, and
refuses the rest with `final_quota_exceeded`, so plan which submissions to
finalize. Final results never reach agents.

Our own reference models (baselines, published methods) are evaluated
separately, so they don't use up the agents' finals: submit them with a
baseline token (`bmb-hub agent-token <name> --evaluations 1 --baseline`, or
`Client.agent_token(..., kind="baseline")`). Their finals count against
`split.baseline_final_evaluations` instead, which defaults to 0: a task
allows baseline finals only once its provider has approved a contract that
sets it (re-enroll the task with the new value; the provider runs
`bmb-eval update`). The leakage budget the provider approves counts both caps.
A node on code from before this ignores the kind and counts every final
against `final_evaluations`, so a job is never less restricted than before.

A model too big for validation (see `final_limits`) is registered by the agent
with `Client.register_final` instead of submitted; `final` accepts its
submission id like any other.

## Run a Modal node (GPU tasks on Modal's H100s)

A node can run its sandboxes on Modal instead of a local container engine
(`sandbox.backend: modal`, `src/bmb_eval/modal_backend.py`). The node loop
(queue, quotas, audit log, approval lock) still runs on a VM; each stage runs in
its own Modal Sandbox (gVisor, `block_network=True`, no secrets) with the
contract's GPUs, CPUs, memory and time limit. Use it for our own public-data
tasks that need more GPUs than a VM has (the genomic suite: 1×H100 per
validation run, 8×H100 per final), not for a provider's private data unless
they accept Modal as a processor (see the threat model).

**How the data is laid out.** `bmb-eval modal-sync` uploads each task's prepared
data into Modal Volumes in the `bmb-eval` environment, one volume per task and
kind: validation inputs, test inputs, answers, and the task folder (contract and
scorer). Every stage mounts read-only only what it may see; predict never gets
answers, the scorer or the other half. A large file both halves see (the 3 GB
reference genome linked into every genomic task) goes once into a volume of its
own and is linked into each task's inputs folder. Model files go to a
content-addressed blob volume that only a trusted stager mounts; each job's
files are copied into a volume of their own, mounted read-only into predict,
and deleted after it. `modal-sync` writes a manifest of what it uploaded (the
same tree hashes as `bmb-eval approve`); the node accepts no work while the
manifest on Modal or any volume's files differ from what was approved.

**Separate environment.** Agent runs execute in the workspace's default
environment (`main`) with the workspace's container credentials. The evaluation
volumes live in `bmb-eval`, created as a restricted environment, so code in
`main` can't open them. Check it after any change to workspace roles:

```bash
MODAL_PROFILE=biomodelbench modal environment list           # bmb-eval is there
# From a Modal Function in main (as an agent's code would run), try to read them:
MODAL_PROFILE=biomodelbench .venv/bin/python deploy/modal/isolation_probe.py <prefix>-meta <prefix>-<task>-ans
# every line must say "refused"
```

**Set it up** on the node VM, in its own folder beside any container node (each
node has its own config, state, approval lock and systemd unit):

```bash
git clone https://github.com/dryingpaint/biomodelbench-private-eval ~/bmb-eval-modal
cd ~/bmb-eval-modal && BMB_ENGINE=modal BMB_SKIP_SELFTEST=1 ./setup.sh
.venv/bin/pip install 'modal>=1.6'            # or: uv pip install --python .venv/bin/python -e '.[modal]'
# Modal credentials for the biomodelbench workspace, in ~/.modal.toml (profile biomodelbench):
.venv/bin/modal token new --profile biomodelbench
```

`setup.sh` with `BMB_ENGINE=modal` writes a `config.yaml` like this; edit it
before the first connect:

```yaml
node_name: bmb-node-public-modal
service_name: bmb-eval-modal      # its own systemd unit; the container node keeps bmb-eval
sandbox: {backend: modal}
modal:
  environment: bmb-eval           # restricted; agents' environment (main) can't open it
  app: bmb-eval-sandbox
  profile: biomodelbench
  gpu: H100
  max_gpus: 64                    # hard cap on GPUs held at once by this node
  max_jobs: 16                    # evaluations at once
```

Then build the image on Modal (Python 3.11, PyTorch 2.5.1 with CUDA 12.4,
transformers 4.46.3 and the genomics tools; `sandbox/modal/`) and connect each
task as usual. Connect uploads the data to Modal, runs the self-test on Modal,
and approves:

```bash
./bmb-eval modal-deploy                        # about 3 min; again after each code update
./bmb-eval connect <code> --data <folder> --yes --no-service   # each task
./bmb-eval install-service                     # starts bmb-eval-modal
./bmb-eval modal-sync --verify                 # volumes match the approved manifest
```

After a code update: `./bmb-eval modal-deploy && ./bmb-eval approve`, then
`systemctl --user restart bmb-eval-modal`. After a data change:
`./bmb-eval modal-sync && ./bmb-eval approve`.

**Failures.** A Modal error (no capacity, an API error, a sandbox that vanished
without its launcher's report) is retried twice inside the stage. If it still
fails, the job goes back to the hub without counting against the agent, and the
node takes no new work for a minute (growing to 30 minutes while it keeps
failing). After `infra_failure_threshold` such failures in a row the alert
webhook fires. A job is handed back at most twice; a third failure is a counted
`infra_error`. The model's own outcomes (exit codes, out of memory, time limit,
too much output) are never retried. `bmb-eval stop` terminates every sandbox
tagged with this node; a restarted node also cleans up orphaned sandboxes and
job volumes.

**Timings** (measured 2026-10-29, controller on a laptop). The image builds in
about 2.5 minutes. A stage's sandbox is ready in about 3 s on CPU and 6 s with
an H100; the example task's whole evaluation (1×H100 predict, staging included)
took 24 s. Data into a sandbox moves at about 35 MB/s, but output comes back at
only about 4 MB/s (Modal's limit, not the network's: 1 GB took 290 s, and
parallel streams don't help). Predictions near the 512 MB cap therefore spend
about 2 minutes in transfer per stage that returns them.

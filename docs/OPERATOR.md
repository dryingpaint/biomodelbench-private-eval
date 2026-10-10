# Running the hub (BioModelBench team)

The hub is the only service we run. Agents submit models to it. Provider nodes
connect *out* to it, collect work, and post back gated scores. It never
receives provider data.

## Deploy once

Any small Linux server works (1 CPU, 2 GB RAM, disk for queued bundles).

```bash
git clone https://github.com/dryingpaint/biomodelbench-private-eval bmb-eval && cd bmb-eval
uv venv && uv pip install -e .
bmb-hub --state hub_state serve --port 8790 --public-url https://hub.biomodelbench.org
```

Put TLS in front, for example with Caddy:
`caddy reverse-proxy --from hub.biomodelbench.org --to 127.0.0.1:8790`.
Run both as services. Back up `hub_state/hub.db`.

The hub serves providers the install script, the guide (`/guide`), and a
tarball of *this checkout's* code. It packs the tarball at startup, so restart
the hub after pulling new code.

**Prebuilt sandbox image (recommended).** This makes installs faster and lets
them work on networks that block package downloads:

```bash
podman build -t bmb-eval-sandbox:latest -f sandbox/Dockerfile .
podman save -o hub_state/sandbox-image.tar bmb-eval-sandbox:latest
```

Without it, each provider builds the image locally, which takes about 10
minutes and needs access to PyPI and Docker Hub.

## Onboard a provider

1. **Agree on the task.** Ask the five questions in `docs/PROVIDER_INTAKE.md`.
2. **Write the task.** Use the column names from their data file:
   ```bash
   bmb-hub new-task binding_affinity --key variant_id \
       --inputs variant_id:string,sequence:string --target kd:float --kind regression
   ```
   Edit `tasks/binding_affinity/contract.yaml`: title, descriptions, limits.
   Run `bmb-eval dryrun` against the public contract with a reference bundle
   to check the scorer.
3. **Make a connect code.** This prints the complete email: one command, plus
   the columns their file needs:
   ```bash
   bmb-hub enroll tasks/binding_affinity --public-url https://hub.biomodelbench.org
   ```
   A code works once, and is only used up when the provider approves.
4. **Watch it come online:** `bmb-hub status`.

Changing a live task's contract or scorer means sending a new code for a new
task version. The node holds its own copy under the provider's approval lock,
so the hub cannot change it remotely.

## Agents

```bash
bmb-hub agent-token claude-run-17        # prints the token once
```

Agents use `bmb_eval.client.Client(hub_url, token)`: see the README.

## Defaults worth deciding per task

| Setting | Default | Trade-off |
|---|---|---|
| `limits.submissions_total` | 200 | Lifetime leakage scales with it |
| `limits.submissions_per_day` | 20 | Per agent |
| `limits.release_after_s` | 0 | ≥ total timeout removes the timing channel but slows feedback |
| `export.decimals` | 3 | Each decimal ≈ 3.3 bits per metric per evaluation |
| `export.ladder_step` | off | Turn on for "final" tasks |

Recommended: a `dev` task with frequent feedback, plus a `final` task on
separate held-out rows with `submissions_total` of 1–5.

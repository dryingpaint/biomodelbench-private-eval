# Connecting your data to BioModelBench

BioModelBench sends AI-trained models to your machine. Your machine scores
them on your data and sends back only the scores. Your data never leaves.

## What you need

- **A Linux machine that stays on.** The server your data is already on is
  ideal. If you don't have a suitable machine, talk to us and we'll help you
  set one up.
- **The data you described to us**, as one file or one folder, in whatever
  format it's already in.

Run the command from our email on that machine:

```bash
curl -fsSL https://<hub address>/install.sh | sh -s -- YOUR-CODE --data /path/to/your/data
```

It takes about 15 minutes and may ask for your password once. It installs
what's needed (including GPU support if the machine has an NVIDIA GPU), checks
your data, tests its own security, and then shows you exactly what you're
agreeing to:

```
✓ Your data looks good (4,812 rows)
✓ Security self-test passed

  The model will see:  variant_id, sequence
  Never shared:        kd
  Evaluations:         at most 20/day per agent, 200 in total
  What leaves:         only scores (spearman, pearson, rmse), 3 decimals
  Worst case:          a model built to extract data could learn at most
                       ~1.2 KB of information about your data, across all 200 evaluations

Approve and start? [y/N]
```

Type `y`. It runs in the background and restarts after
reboots.

If something needs fixing, it says what (for example, *"It has no column
'sequence'"*, or *"This task needs 40 GB of GPU memory; the GPU here has 24
GB"*). Fix it and run the same command again.

## Useful commands

| | |
|---|---|
| Check it's running | `~/bmb-eval/bmb-eval status` |
| **Stop everything immediately** | `~/bmb-eval/bmb-eval stop` |
| Start again | `~/bmb-eval/bmb-eval resume` |

## Technical details

### Network
The machine only makes outbound HTTPS connections: to the BioModelBench hub
(`https://<hub address>`) and to `storage.googleapis.com`, where models are
downloaded from. During installation it also reaches standard package mirrors
(Ubuntu or Red Hat, PyPI, and NVIDIA on GPU machines). Nothing connects in, so
no ports, web server or firewall changes are needed.

### What leaves the machine
Only scores, such as `{"spearman": 0.712, "pearson": 0.731, "rmse": 0.193}`,
or an error from a fixed list (for example `predict_timeout`, or which output
column was malformed). Your data, the model's predictions, logs and the
model's own error output never leave, because a model could hide data in its
error messages.

### What gets installed
Podman (container software from your operating system's own package
repository), the NVIDIA container toolkit on GPU machines, and the evaluator
itself in `~/bmb-eval`. Nothing else is installed system-wide.

### How models are contained
Each model runs in a locked-down container with no network access, a
read-only filesystem, no access to the answers, and strict memory, CPU and
time limits. Your data is split on your machine, so the model only ever
receives what's listed under "The model will see". Before anything runs, the
setup attacks its own sandbox and stops if any attack succeeds.

### Disk use
Downloaded models are kept in `~/bmb-eval/state/blobs` (up to 300 GB, oldest
removed first), so a model reused across submissions is downloaded only once.
Large data files are linked rather than copied.

### Records and changes
Every evaluation is recorded in `~/bmb-eval/state/` in a tamper-evident audit
log that stays on your machine. If your data or settings change, the evaluator
pauses until you approve the change with `~/bmb-eval/bmb-eval approve`.

### Alerts
Alerts are optional. To be notified of security events or problems, put a
Slack or Teams webhook URL in `alerts.webhook_url` in `~/bmb-eval/config.yaml`.

### Extra isolation on shared servers
For an additional layer of isolation, install [gVisor](https://gvisor.dev)
(enabling its `nvproxy` option on GPU machines) and set `sandbox.runtime: runsc`
in `~/bmb-eval/config.yaml`. Then run `~/bmb-eval/bmb-eval selftest` and
`~/bmb-eval/bmb-eval approve`.

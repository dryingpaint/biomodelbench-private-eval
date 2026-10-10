# Connecting your data to BioModelBench

BioModelBench sends AI-trained models to your machine. Your machine scores
them on your data and sends back only the scores. Your data never leaves.

## What you need

- **A Linux machine that stays on.** The server your data is already on is
  ideal. Our email says what it needs: CPU-only tasks need 4 CPUs, 16 GB RAM
  and 150 GB free disk. Tasks with large models also need an NVIDIA GPU, 64 GB
  RAM and about 300 GB free disk. If you don't have a suitable machine, see
  [Getting a Linux machine](#getting-a-linux-machine).
- **The data you described to us**, as one file or one folder, in whatever
  format it's already in.

## The one step

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

Type `y`. **You're done.** It runs in the background and restarts after
reboots.

If something needs fixing, it says what (for example, *"It has no column
'sequence'"*, or *"This task needs 40 GB of GPU memory; the GPU here has 24
GB"*). Fix it and run the same command again.

## If you ever need to

| | |
|---|---|
| Check it's running | `~/bmb-eval/bmb-eval status` |
| **Stop everything immediately** | `~/bmb-eval/bmb-eval stop` |
| Start again | `~/bmb-eval/bmb-eval resume` |

## Getting a Linux machine

Only needed if the machine your data is on isn't suitable (for example, it's
Windows, a laptop, or has no GPU for a large-model task).

**Ask your IT or research computing group.** You can forward this:

> Please create a Linux VM for an external benchmarking collaboration:
> Ubuntu 24.04 LTS · 4 vCPUs · 16 GB RAM · 150 GB disk plus the size of the
> dataset (for large-model tasks: one NVIDIA GPU with the memory BioModelBench
> specified, 64 GB RAM, 300 GB disk) · outbound HTTPS allowed · no inbound
> ports · one user account with sudo. It needs to run continuously for
> [benchmark duration].

**Or use your own Google Cloud, AWS or Azure account**, if your data is
already there. Create an Ubuntu 24.04 VM of the size above, block all inbound
traffic except SSH, and copy your data onto it. For a GPU on Google Cloud, a
`g2-standard-16` machine (one NVIDIA L4, 24 GB) with Google's "Deep Learning
VM" image comes with GPU drivers installed. Use an `a2` machine (A100) for
tasks needing more GPU memory.

## FAQ for your IT team

**Network?** Outbound HTTPS only: to the BioModelBench hub address, to
`storage.googleapis.com` (where models are downloaded from), and during
install to package mirrors (Ubuntu/Red Hat, PyPI, NVIDIA). No inbound ports,
no web server, no firewall changes.

**What leaves the machine?** Only scores like
`{"spearman": 0.712, "pearson": 0.731, "rmse": 0.193}`, or a fixed error code
such as `predict_timeout`. Never data, logs, error messages or predictions.

**What's installed?** Podman (container software from your OS's own package
repository), on GPU machines the NVIDIA container toolkit, and the evaluator
in `~/bmb-eval`. Nothing else system-wide.

**How is the model's code contained?** Each model runs in a locked-down
container: no network, read-only filesystem, no access to the answers, and
strict memory, CPU and time limits. Your data is split locally: the model only
ever receives the columns and files listed as "The model will see" (large
files are hard-linked, not copied). Setup attacks its own sandbox before
anything runs, and stops if any attack succeeds. Downloaded model files are
checked against their hashes. The full list of protections is in the
[security details](security) (also in `~/bmb-eval/docs/THREAT_MODEL.md`).

**Disk use?** Downloaded models are cached in `~/bmb-eval/state/blobs` (up to
300 GB by default, oldest removed first), so a model reused across submissions
is downloaded only once.

**Extra protection on a shared server:** install [gVisor](https://gvisor.dev)
(on GPU machines, enable its `nvproxy` option), set `sandbox.runtime: runsc` in
`~/bmb-eval/config.yaml`, then run `~/bmb-eval/bmb-eval selftest` and
`~/bmb-eval/bmb-eval approve`.

**Records?** Every evaluation is logged in `~/bmb-eval/state/`, with a
tamper-evident audit log. Logs never leave the machine.

**Changes?** If the data or settings change, the evaluator pauses itself
until someone runs `~/bmb-eval/bmb-eval approve`.

**Alerts?** Optional. Put a Slack or Teams webhook URL in
`alerts.webhook_url` in `~/bmb-eval/config.yaml`.

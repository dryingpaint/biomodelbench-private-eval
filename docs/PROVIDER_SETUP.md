# Connecting your data to BioModelBench

BioModelBench sends AI-trained models to your machine. Your machine scores
them on your data and sends back only the scores. Your data never leaves.

## What you need

- **A Linux machine that stays on**, with 4 CPUs and 16 GB RAM. The server
  your data is already on is ideal. If you don't have one, see
  [Getting a Linux machine](#getting-a-linux-machine).
- **Your data as one file** (CSV is fine). Our email lists the columns it
  needs. Extra columns are fine; they're ignored.

## The one step

Run the command from our email on that machine:

```bash
curl -fsSL https://<hub>/install.sh | sh -s -- XXXX-XXXX-XXXX --data your_data.csv
```

It takes about 10 minutes and may ask for your password once. At the end it
shows you exactly what you're agreeing to:

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

If something is wrong with your file, it says what (for example, *"It has no
column 'sequence'"*). Fix the file and run the same command again.

## If you ever need to

| | |
|---|---|
| Check it's running | `~/bmb-eval/bmb-eval status` |
| **Stop everything immediately** | `~/bmb-eval/bmb-eval stop` |
| Start again | `~/bmb-eval/bmb-eval resume` |

## Getting a Linux machine

Only needed if the machine your data is on isn't suitable (for example, it's
Windows or a laptop).

**Ask your IT or research computing group.** You can forward this:

> Please create a Linux VM for an external benchmarking collaboration:
> Ubuntu 24.04 LTS · 4 vCPUs · 16 GB RAM · 50 GB disk plus the size of the
> dataset · outbound HTTPS allowed · no inbound ports needed · one user
> account with sudo. It needs to run continuously for [benchmark duration].

**Or use your own cloud account** (AWS, Google Cloud or Azure), if your data
is already there. Create an Ubuntu 24.04 VM with 4 CPUs and 16 GB RAM (roughly
$100–150/month), block all inbound traffic except SSH, and copy your file onto
it.

## FAQ for your IT team

**Network?** Outbound HTTPS only, to the BioModelBench hub (plus package
downloads during install). No inbound ports, no web server, no firewall changes.

**What leaves the machine?** Only scores like
`{"spearman": 0.712, "pearson": 0.731, "rmse": 0.193}`, or a fixed error code
such as `predict_timeout`. Never data, logs, error messages or predictions.

**What's installed?** Podman (container software from your OS's own package
repository) and the evaluator, into `~/bmb-eval`. Nothing else system-wide.

**How is the model's code contained?** Each model runs in a locked-down
container: no network, read-only filesystem, no access to the answer column,
and strict memory, CPU and time limits. Your data is split locally: the model
only ever receives the columns listed as "The model will see". Setup attacks
its own sandbox before anything runs and stops if any attack succeeds. The
full list of protections is in `docs/THREAT_MODEL.md` in `~/bmb-eval`.

**Extra protection on a shared server:** install [gVisor](https://gvisor.dev),
set `sandbox.runtime: runsc` in `~/bmb-eval/config.yaml`, then run
`~/bmb-eval/bmb-eval selftest` and `~/bmb-eval/bmb-eval approve`.

**Records?** Every evaluation is logged in `~/bmb-eval/state/`, with a
tamper-evident audit log. Logs never leave the machine.

**Changes?** If the data file or settings change, the evaluator pauses itself
until someone runs `~/bmb-eval/bmb-eval approve`.

**Alerts?** Optional. Put a Slack or Teams webhook URL in
`alerts.webhook_url` in `~/bmb-eval/config.yaml`.

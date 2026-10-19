# Connecting your data to BioModelBench

BioModelBench scores AI-trained models on your data and sends back only the
scores. There are two ways to connect; our email says which commands to run
for each:

- **Option A: no machine to run (recommended).** Your data is encrypted on
  your computer with a key in your own Google Cloud project, and only our
  sealed evaluation machine can open it.
- **Option B: your own machine.** Models are sent to a Linux machine of
  yours, which scores them where your data already is.

# Option A: sealed evaluation (no machine to run)

Your data is encrypted with a key that only you control, and only our sealed
evaluation machine can open it. Nobody else can read it, including
BioModelBench.

## What you need

- **A Google Cloud project** with billing enabled, where you're an Owner (or
  have the roles under "Technical details" below). A new, empty project is
  fine; the cost is a few cents a month.
- **The Google Cloud CLI (`gcloud`)**, logged in (`gcloud auth login`), on a
  Mac or Linux computer that has your data. On Windows, use WSL.
- **Your data**, as one file or one folder, as described in our email.

The command sets up everything else in your project for you.

Run the command from our email:

```bash
curl -fsSL https://<hub address>/confidential.sh | sh -s -- YOUR-CODE --data /path/to/your/data --project YOUR-GCP-PROJECT
```

It checks your data and shows you exactly what you're agreeing to. Type `y`
and it encrypts your data on your computer and uploads only the encrypted copy.
Nothing keeps running on your side.

With several tasks, our email lists one command per task. The first installs
the tool; the others reuse it and your project:

```bash
~/bmb-confidential/.venv/bin/bmb-eval connect-confidential NEXT-CODE --data /path/to/next/file
```

## Stopping, changing data, and updates

- **Stop everything at any time:** disable your key. The sealed machine can
  then no longer open your data. (The command is printed when you connect.)
- **New or corrected data:** run the same command again with the new file.
- **A task change or new software from BioModelBench:** we'll ask you to run
  your command again, so that you review and approve the change.

## Technical details

### How it works
The evaluation runs on Google Confidential Computing hardware (Intel TDX and an
NVIDIA H100 in confidential mode). The machine is sealed: nobody outside it,
including BioModelBench and Google, can read its memory or log into it. Your
key in your Google Cloud project is released only to a machine that proves, in
hardware, that it runs the exact software you approved, with debugging off.

The command installs the tool in `~/bmb-confidential` and splits your data on
your computer into a validation half and a held-out test half. Its approval
screen shows the same terms as Option B's, plus the exact software your key
will open for and the public source commit it was built from. When you type
`y`, it creates the key, bucket and access rule in your project, encrypts
everything on your computer, and uploads only the encrypted copy.

### Checking the software

```bash
curl -fsSL https://<hub address>/verify-image.sh | sh
```

This checks Google Cloud Build's signature on the build record of the approved
software, and that it was built from the published source commit. It prints a
link to that exact code, including the parts that let only the agreed scores
out.

### Permissions
The project's Owner has everything needed. Otherwise, ask for these roles:
Service Usage Admin, Cloud KMS Admin, Cloud KMS CryptoKey Encrypter, Storage
Admin, Workload Identity Pool Admin. `gcloud` must be version 450 or newer.

### Network
The computer needs outbound HTTPS to `eval.biomodelbench.org`, Google Cloud
(`accounts.google.com`, `*.googleapis.com`) and PyPI (`pypi.org`,
`files.pythonhosted.org`), or your organization's PyPI mirror (set
`PIP_INDEX_URL`, or pip's own setting). An HTTPS proxy works: set
`HTTPS_PROXY` as usual. If your network inspects HTTPS with its own
certificate authority, this computer must trust it (it already does if your
browser works), and gcloud needs it too:
`gcloud config set core/custom_ca_certs_file /path/to/your-company-ca.pem`.

### What gets created in your project

| | |
|---|---|
| Cloud KMS key `data` in key ring `bmb` (us-central1, or your `--region`) | Wraps the key that encrypts your data. Only you can manage it. |
| Bucket `gs://<project>-bmb-data` (same region) | Your encrypted data, and the evaluator's encrypted records (how many evaluations were used). |
| Workload identity pool `bmb-eval` | The access rule: the key is released only to the approved software on sealed hardware, in production mode. |

Services enabled: Cloud KMS, Cloud Storage, IAM, Security Token Service.

### If your organization restricts Google Cloud

- **Allowed locations:** the key and encrypted data go in `us-central1`. If
  your organization allows only some regions, add `--region <one it allows>`
  (for example `--region europe-west4`); the command says so if needed.
- **Allowed workload identity pool providers:** the access rule trusts
  `https://confidentialcomputing.googleapis.com/` (Google Confidential Space).
  If your organization restricts this list, it needs to allow that issuer.
- **Restricted services:** it uses Cloud KMS, Cloud Storage, IAM and Security
  Token Service.

Domain-restricted sharing and Google's secure-by-default policies don't get in
the way (tested). If a step is refused, the command stops and shows Google's
message; send it to us.

# Option B: your own machine

BioModelBench sends AI-trained models to your machine. Your machine scores
them on your data and sends back only the scores. Your data never leaves.

## What you need

- **A Linux machine that stays on**, of the size our email gives for your
  tasks (for example, two 80 GB NVIDIA GPUs, 24 CPUs, 256 GB RAM and 500 GB
  free disk for tasks whose final evaluations run the largest models). Most of
  the time it runs small models on CPUs. The install checks the machine and
  says if anything is missing. If you don't have a suitable machine, talk to
  us and we'll help you set one up, or use Option A.
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
| Review a task change from BioModelBench | `~/bmb-eval/bmb-eval update` (`status` tells you when there is one) |

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
repository) and, on Debian and Ubuntu, `dbus-user-session`; the NVIDIA
container toolkit on GPU machines; and the evaluator itself in `~/bmb-eval`.
Two settings may be changed so that the sandbox's limits work for your user:
on systems whose systemd doesn't let users limit CPU (Ubuntu 22.04, RHEL and
Rocky 9), the drop-in `/etc/systemd/system/user@.service.d/delegate.conf`
(Podman's documented setting); and, if Podman would otherwise use its slow
`vfs` storage, `~/.config/containers/storage.conf`. Lingering is enabled for
your user so the evaluator keeps running after you log out. Tested on Ubuntu
22.04 and 24.04, Debian 12 and Rocky 9 (SELinux enforcing).

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

# FAQ

**What leaves, with either option?** Only the agreed scores, rounded, or an
error code from a fixed list. Never your data, the model's predictions or its
logs. The approval screen shows the worst case: how much a model built to
extract data could learn across all evaluations.

**Who decides which half of my data is the test set?** Your computer (Option
A) or machine (Option B) splits it at random when you connect, with a secret
seed. Nobody, including BioModelBench, knows which row is where. Agents are
scored only on the validation half; the test half is used at most a few times,
at the end, and only BioModelBench sees those scores.

**What file formats work?** CSV (including Excel's "CSV" and "CSV UTF-8"),
TSV or Parquet; for a folder of files, as described in our email. Excel
workbooks (`.xlsx`): save as CSV first.

**The command found a problem with my file.** It says what to fix (for
example, a missing column, an empty value or a value not in the allowed list).
Fix it and run the same command again; your code keeps working.

**Can I try it on a test file first?** Yes: stop at the approval screen
(answer `n`), and nothing is created or shared. If you approved a test file
by mistake, run the same command again with your real file; it replaces it.

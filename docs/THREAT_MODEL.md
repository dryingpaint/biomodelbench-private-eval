# Threat model

## Who we defend against

1. **The agent.** We assume its submitted code is hostile. This could come
   from reward hacking, misalignment, or a compromised training environment.
   It can do anything Python can do inside the sandbox, sees the private
   inputs, and can submit as often as its quota allows.
2. **Anyone else who reaches the hub**, or compromises it.
3. **Accidental misconfiguration by the provider.** This is the most likely
   real-world failure.

What is protected: the provider's private inputs and answers. What also
matters: the integrity of the benchmark numbers.

## Verdict on the proposal

The architecture is the right one. Shipping code to the data, running it in
an isolated container, and returning only gated metrics beats every
alternative we considered (see the last section). But three claims in the
proposal do not hold as written. The implementation is built around fixing
them.

**1. "The container can only export what the scorer outputs" is not a privacy guarantee.**
The submitted code sees the private inputs and chooses the predictions. The
scorer computes metrics from those predictions. So the code can make the
metrics depend on the data on purpose. For example: "if row 17's sequence
starts with M, predict all zeros; otherwise predict the true model." The
Spearman correlation then reveals one bit about row 17. A 3-decimal metric
can carry about 11 bits per submission. The same works for answers: varying
one row's prediction and watching the RMSE reveals that row's label (a
classic leaderboard attack). Output formats cannot close this channel, because
the metric is a function of the data by definition. It can only be
**narrowed and counted.** `bmb-eval check` prints that count.

**2. "Approved error formats" are a second channel.** The code chooses whether
to crash, time out, run out of memory, or emit a malformed column, and each
choice selects a different error code. The proposal treats errors as safe
because their *format* is fixed. Their *choice* still carries information. It
is counted in the budget too (about 6 bits).

**3. Timing is a third channel.** If results appear as soon as a run finishes,
the code can sleep for a data-dependent number of seconds. That is about 11
bits per submission over a 30-minute timeout. Setting `release_after_s` to at
least the total timeout closes it. Results are then hidden as `pending` until
a fixed time after submission.

Two further points the proposal does not address:

- **The export check has to sit outside the container that runs the agent's
  code.** If the gate runs in the same container, the code can bypass it. Here
  the gate runs in the controller process, which never runs submitted code.
- **Adaptive overfitting.** Even an honest agent that evaluates many times
  against one private set will overfit to it, and its score will be
  optimistic. The first Capable test showed this: the agent ran ablations
  against the private set. Set `split` in the contract:
  - The provider's node randomly splits the rows, with a secret seed, into
    validation and test sets.
  - Agents are only ever shown, and scored on, the validation rows.
  - BioModelBench requests final evaluations on the test rows
    (`bmb-hub final`). The node runs at most `final_evaluations` of them,
    enforced on the provider's machine.
  - Final scores are visible to BioModelBench only. Only the final score
    counts.

## The worst-case leakage budget

For each task, `leakage.py` computes an upper bound on what one submission
can carry:

```
bits ≤ log2(distinguishable metric outcomes + distinguishable error outcomes) + timing bits
```

For the example task (3 metrics at 3 decimals, 30-minute timeout,
results released on finish, 200 lifetime submissions) the bound is **47
bits per submission and about 9,300 bits over the lifetime**. That is roughly
1.2 KB: enough for about a hundred short sequences, or the labels of a few
thousand rows at one bit each. Honest submissions reveal far less. This is
the limit for a hostile one.

How to shrink it, from most to least effective:

| Lever | Effect |
|---|---|
| `submissions_total` | Lifetime bits scale linearly with it. The main control. |
| `release_after_s` ≥ total timeout | Removes about 11 bits per submission. |
| Fewer metrics / fewer `decimals` / tighter metric bounds | Each 3-decimal correlation costs about 11 bits; each decimal dropped saves about 3.3 bits per metric. |
| `ladder_step` (the Ladder mechanism) | Most submissions return only "not improved"; information grows roughly logarithmically, not linearly, with submissions. |
| Larger test set | Each row's influence on an aggregate shrinks, so targeted single-row probing needs more submissions. |

**Metric design matters too.** Avoid metrics that expose a single row:
max/min error, per-row or per-class counts on rare classes, anything defined
on a tiny subset. A "max absolute error" metric with constant predictions
returns your most extreme label directly.

## Attacks and defenses

| Attack | Defense | Where | Residual risk |
|---|---|---|---|
| Send data out over the network | `--network none`. No interfaces except loopback. | `sandbox.py` | Container escape (see below) |
| Read the answers | Predict sandbox never mounts answers; each file is mounted individually, never its directory | `runner.py` | None known |
| Write private data somewhere it is later exported | Read-only root, read-only mounts, no writable host path; output only via a byte-capped pipe | `sandbox.py` | None known |
| Smuggle data in error messages or stdout | Fixed list of error codes; stderr stays in the provider-only run log | `errors.py`, `runner.py` | Choice of code: counted in the budget |
| Encode data in the metric | Gate: exact names, numeric, bounded, rounded | `contract.py:gate_metrics` | **Inherent; bounded by the budget** |
| Encode data in timing | `release_after_s` hides results until a fixed time | `store.py` | Open unless configured; counted in the budget |
| Exploit the predictions parser next to the answers | Untrusted parquet is parsed in a separate validate sandbox **without answers**; the scorer only loads plain numpy arrays (`allow_pickle=False`) | `sandbox_entry/validate.py` | Exploiting the validator gains nothing it could not already do |
| Fill the provider's disk or memory | tmpfs and memory caps, pid limit, capped stdout, `--log-driver none` (Docker's default would copy stdout, possibly containing private data, to `/var/lib/docker`) | `sandbox.py` | None known |
| Malicious bundle manifest (paths like `../`, absolute paths, huge sizes) | Manifests are validated before any download; only regular files at safe relative paths; size cap per task | `blobs.py` | None known |
| Tampered model files (in storage or in transit) | Every file is checked against its sha256 before entering the node's cache; a mismatch rejects the submission | `blobs.py` | None known |
| One agent planting a bad file under another's hash | Hub storage is per agent; nodes verify hashes anyway | `hub.py` | None known |
| Tar tricks (symlinks, `..`, devices, decompression bombs) | Rejected before extraction; "data" extraction filter | `bundle.py` | None known |
| Fork bomb, CPU exhaustion | `--pids-limit`, `--cpus`, timeouts | `sandbox.py` | None known |
| Covert channel between concurrent runs | One run at a time | `node.py` | None known |
| Container or kernel escape | Capabilities dropped, no-new-privileges, non-root; **gVisor recommended** | `sandbox.py` | Without gVisor, a host-kernel bug = escape. This is the largest remaining technical risk. |
| Silently swapped scorer, contract, data or image | Fingerprint lock; node refuses until re-approved and alerts | `approval.py` | Someone with root can re-approve |
| Tampering with run history | Hash-chained audit log | `audit.py` | Root can rewrite the whole chain; copy the latest hash off the machine periodically |
| Spamming alerts (deliberately crashing the parser) | Infra alerts fire only at the N-th *consecutive* failure | `runner.py` | Mild nuisance |
| Stolen agent token | Hashed at rest on the hub; per-agent daily quota and global lifetime cap enforced **on the node** | `hub.py`, `node.py` | Rotate tokens |
| Attack on the provider's network | The node opens no ports; it only makes outbound HTTPS calls to the hub | `node.py` | None known |
| Compromised hub | The hub never holds provider data. It can send hostile bundles, which every bundle is assumed to be anyway. It cannot change a task's contract or scorer after the provider approves: the node keeps its own copy under the approval lock, and quotas, release timing and the output gate all run on the node | `node.py`, `approval.py` | A hub could hand a *new* provider a hostile scorer at connect time. That scorer still runs sandboxed and gated, so it can leak no more than the budget shown at approval |
| Per-group metrics multiply leakage | Each group is a separately bounded metric, and the budget counts every one | `contract.py`, `leakage.py` | Operators must keep group counts small |
| A file column pointing at the answers | File paths must stay inside the data folder and may not name the samples table; only listed columns and files are linked into the model's folder | `prepare.py` | A provider could put answers inside a pass-through file themselves; the approval summary lists every file the model will see |
| Node running on the provider's main data server | Same sandbox; only the task's two files are mounted | `sandbox.py` | A container escape would reach everything that account can read. Use gVisor, or a dedicated account or VM |

`bmb-eval selftest` runs the first half of this table as real attack bundles
against the live sandbox. A configuration error, such as forgetting
`--network none` or using an image with a writable root, shows up as `FAIL`
before any private data is exposed.

## What this design does not protect

- **Benchmark integrity against the provider.** The provider runs the scorer
  and could report whatever it likes. To address this: have the node sign
  results with a key registered with BioModelBench and include the approved
  fingerprint in the signature. For hard guarantees, use a confidential VM
  with remote attestation (see below).
- **The agent's weights against the provider.** Bundles are deleted after each
  run by default (`retain_bundles: false`), but the provider has root on the
  machine.
- **Membership inference** within the budget. A hostile submission can test
  whether a specific known sequence is in the private set: one targeted bit.
  If cohort membership itself is sensitive (for example, patient data), use a
  very small `submissions_total` or add noise to the metrics.

## Alternatives considered

| Approach | What it fixes | What it doesn't | Use it? |
|---|---|---|---|
| **This design** (code to data, sandbox, gated metrics, budget) | Direct data access, free-text leaks, misconfiguration | Metric channel (bounded) | Yes: the baseline |
| **Constrained model format** (ONNX or safetensors, run by trusted code) | Removes arbitrary code execution, so the error, timing and escape channels mostly disappear | Metric channel; limits which models agents can build | **Recommended high-assurance tier** for sensitive providers |
| **Differential privacy on metrics** (add noise at the gate) | Formal per-row privacy guarantee | Noise reduces the training signal; the privacy budget runs out | Optional add-on for clinical data |
| **Confidential VMs / TEEs** (SEV-SNP, TDX, H100 confidential compute) | Proves to BioModelBench which scorer ran (integrity); protects weights from the provider | Nothing about the metric channel | Later, if result integrity is disputed |
| **Federated learning** | Training on private data | Wrong problem: we only need evaluation | No |
| **Synthetic data only** | Everything | Doesn't measure real performance | No, but agents should use `bmb-eval dryrun` on synthetic data to avoid wasting private submissions |

**Bottom line:** no design avoids the metric channel. Any evaluation that
returns a useful number reveals something about the data it was computed on.
The robust approach is to make that leakage explicit, bounded, and part of
what the provider agrees to: here, `bmb-eval check` plus quotas. Next upgrades,
in order:

1. Make gVisor mandatory.
2. Use `split` (a held-out test set) on every task.
3. Add an ONNX-only tier.
4. Make the hub's scorer and contract delivery verifiable (signed task packages).
5. Sign results with the node's key so BioModelBench can verify their origin.

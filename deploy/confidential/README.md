# Confidential computing (prototype)

BioModelBench runs the evaluation node on its own hardware, inside a
Confidential Space VM. The provider's data stays encrypted with the
provider's own key, which Google releases only to a VM that proves (hardware
attestation) it is running the exact image the provider approved, in
production mode. We own the machine but can't read the data.

```
provider project                         biomodelbench
  encrypted CSV  ◄──── read ──────────  Confidential Space VM (sealed memory)
  KMS key        ◄──── attested ──────    image sha256:X decrypts in memory,
  rule: digest X,      token              runs models, sends only scores
  debug off
```

- `provider-setup.sh`: the provider runs it in their project. It creates the
  key and bucket, encrypts on their machine, and trusts only `--digest`.
  Re-run with a new digest to approve an updated image.
- `launch.sh`: we run it. It starts the image as a Confidential Space workload
  (`cpu` = AMD SEV, `cpu-tdx` = Intel TDX, `h100` = TDX + H100, preview;
  `SPOT=1` for Spot).
- `probe/`: test workload. It decrypts and logs only row count and hash, then
  checks which sandboxes work inside.

## Results (2026-10-15)

| Test | Result |
|---|---|
| Approved image, production mode (AMD SEV and Intel TDX CPUs) | Decrypted; 5,000 rows, hash identical to the provider's file |
| Approved image on an **H100** Confidential VM (Spot `a3-highgpu-1g`, preview image, `--allow-gpu-preview`) | Decrypted (same hash); `nvidia_gpu.cc_mode: ON` attested; `nvidia-smi conf-compute`: CC ON, ready |
| Unapproved image digest | Refused: "rejected by the attribute condition" |
| Approved digest on the debug image (operator can log in) | Refused (`dbgstat: enabled`) |
| GPU preview image without the provider's opt-in | Refused (`EXPERIMENTAL`, not `STABLE`) |
| Re-approval (provider re-runs with the new digest) | Decrypted |
| Sandboxes inside the container: gVisor (systrap, ptrace), bubblewrap, user+net namespaces | All run with no network |
| GPU inside the gVisor sandbox (`runsc --nvproxy do`) | **Fails**: can't reach the driver |


## End to end (2026-10-15): the real sealed node, H100

Image `sealed-node@sha256:1e5976d2…` (`src/bmb_eval/confidential.py`,
`deploy/confidential/Dockerfile`), simulated provider project `bmb-provider-sim-1014`.

| Step | Result |
|---|---|
| Synthetic self-test on sealed H100 (no provider) | 13/13 attacks contained, including GPU access from inside the sandbox |
| Provider: `curl …/confidential.sh \| sh -s -- CODE --data file.csv --project P` | Data checked and split locally (492 / 508). Approval screen shows the image digest. Key, bucket and attestation rule are created. Only ciphertext is uploaded. |
| Sealed node boots | Unsealed into tmpfs. Self-test on the provider's task: 0 failures. Serves jobs. |
| Agent CPU model, validation | spearman_exact 0.952, concordance 0.921 (final-only RMSE/MAE withheld) |
| Agent GPU model, registered as final candidate, final run | spearman_exact 0.938, RMSE 0.413, MAE 0.332. The model refuses to run without CUDA, so the GPU worked inside the sandbox. |
| Idle | The node exits after `BMB_IDLE_EXIT_S`; the VM stops |

Fixes found on real hardware:
- Confidential Space's GPU driver is 535 (CUDA 12.2), so the image uses PyTorch's CUDA 12.1 build.
- Memory limits are per-process data limits, not cgroups, so math-library thread pools are sized to the CPU limit.
- A MemoryError anywhere maps to the resource-limit error.
- The read-only GPU driver mount is allowed in the sandbox's view.

**Not done yet (before production):**
- **GPU inside the sandbox.** Try gVisor nvproxy through `runsc run` with an
  OCI spec, and check its driver-version support. Otherwise fall back to
  bubblewrap binding only `/dev/nvidia*`, the bundle and the inputs.
- **Filesystem isolation.** A quick `unshare` sandbox shares the container's
  files. The real sandbox must mount only the bundle and inputs (an OCI spec,
  or bubblewrap without `--ro-bind / /`).
- **Running the node itself here.** Pull jobs from the hub, score, and post
  gated results.
- **Logs.** The probe uses `log_redirect=always`. The production image should
  not, and the provider's rule should require it.
- **Quota.** On-demand H100 quota was denied. Spot H100 (1) and the global GPU
  quota (1) were granted.

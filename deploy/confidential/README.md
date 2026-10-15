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

**Not done yet:**
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

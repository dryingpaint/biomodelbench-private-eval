# Sealed node (confidential computing)

The provider keeps the key, and BioModelBench runs the machine. The
provider's data is encrypted with their own Cloud KMS key in their own
bucket. Google releases that key only to a Confidential Space VM that proves,
through hardware attestation, that it runs the exact image the provider
approved, in production mode (debugging off). We own the VM but can't read
its memory or log into it.

```
provider project                         biomodelbench
  encrypted task + data  ◄── read ─────  Confidential Space VM, 1x H100 (TDX, GPU CC mode)
  KMS key  ◄── attested token ─────────    sealed-node image sha256:X
  rule: digest X, debug off, our SA        unseals into tmpfs, sandboxes every stage,
  quotas/finals, encrypted  ◄── write ──    posts only gated scores to the hub
```

## Pieces

| | |
|---|---|
| Provider command | `curl -fsSL https://eval.biomodelbench.org/confidential.sh \| sh -s -- CODE --data FILE --project PROJECT` (`bmb-eval connect-confidential`). It checks and splits the data locally, shows the approval screen (with the image digest and its source commit), creates the key, bucket and attestation rule, and uploads only ciphertext. |
| Sealed node | `src/bmb_eval/confidential.py` (`confidential-node`) |
| Sandbox | `BwrapSandbox` in `src/bmb_eval/sandbox.py`: bubblewrap as an unprivileged user, no network, minimal filesystem, read-only root, plus a seccomp filter blocking escape-prone kernel interfaces (user namespaces, mount, ptrace, bpf, io_uring, userfaultfd, keyrings, modules). gVisor isn't used here because it can't reach the GPU inside Confidential Space. |
| Image | `Dockerfile`, built from a public commit by `build.sh` with Cloud Build's signed SLSA provenance. Base image pinned by digest, Debian by snapshot, Python by hash (`requirements.lock`). |
| Provider check | `curl -fsSL https://eval.biomodelbench.org/verify-image.sh \| sh`: checks Google Cloud Build's signature on the SLSA v1 provenance (with Google's published key), and that it binds the approved digest to the published commit |
| Start on demand | The hub (`bmb-hub serve --launch-sealed`, `src/bmb_eval/sealed_launcher.py`) creates the node's VM when its tasks have queued jobs and it's offline. It tries on-demand, then Spot, across us-central1-a, us-east5-a and europe-west4-c. The node exits when idle (`BMB_IDLE_EXIT_S`), and the VM stops. Manual start: `bmb-hub confidential-launch <node>`. |
| Status | No logs in production (`log_redirect=debugonly`). The node reports fixed status fields to the hub (`bmb-hub status`). |
| Final limits | Tasks set `final_limits_sealed` (e.g. `gpus: 1, max_bundle_gb: 80`) for this node; providers' own machines keep `final_limits`. |
| Self-test | `ENV_FILE=none ./launch.sh h100` runs every sandbox attack on synthetic data, on the debug image with logs. |

## Operator

```bash
deploy/confidential/build.sh <commit>                  # build + provenance (commit must be pushed)
bmb-hub confidential-image <image@digest> --workload-sa bmb-cs-workload@biomodelbench.iam.gserviceaccount.com \
    --gpu-preview --source <repo@commit> --provenance provenance.json
# providers re-run their connect command to approve the new digest
bmb-hub status                                         # node status, launches
```

## Limits

- **GPU Confidential Space is a Google preview** (`EXPERIMENTAL` image).
  Providers opt in, and their rule then also requires the H100 to attest
  confidential mode.
- **One H100 per sealed node.** Finals go up to ~70B parameters at 8-bit.
- **Quota (2026-10-15).** Spot H100 quota is approved in us-central1, us-east5
  and europe-west4. On-demand H100 was denied twice: the project is new and has
  no spend history. The global GPU limit is 1 (a raise to 4 was denied), so only
  one sealed node can run at a time. Retry after some billing history, or via
  a Google Cloud sales contact. Spot VMs can be reclaimed; jobs re-queue and
  the hub restarts the node.
- **Trust.** Providers trust Intel TDX, NVIDIA H100 CC, Google's Confidential
  Space image, and the code at the commit they verify.

Test history: the probe (`probe/`) and an end-to-end run on a sealed H100
with a simulated provider (`bmb-provider-sim-1014`).
- Key release was refused for an unapproved digest, the debug image, and a
  GPU preview image without the provider's opt-in.
- 13/13 sandbox attacks were contained.
- Validation and a GPU final were scored without the data ever leaving
  sealed memory.

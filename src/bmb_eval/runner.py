"""One evaluation: bundle bytes in, gated payload out.

    unpack   (controller)  safe tar extraction, size limits
    predict  (sandbox A)   untrusted code + private inputs; no answers, no network
    validate (sandbox B)   untrusted predictions -> canonical arrays; no answers
    score    (sandbox C)   trusted scorer + answers; parses only canonical arrays
    gate     (controller)  exact metric names, numeric, bounded, rounded

Family batching (predict_batch, on nodes with batch_families): jobs of one bundle
and split on several tasks share one predict sandbox, which loads the model once
and writes each task's predictions as its own frame; each job is then evaluated
from its frame (evaluate(predicted=...)): validate, score and gate exactly as above.

Data crosses stages only through pipes held by the controller. Everything
except the gated payload (stderr, tracebacks, predictions, timings) stays in
state/runs/<run_id>/ on the provider's disk.
"""
from __future__ import annotations

import dataclasses
import json
import secrets
import shutil
import time
from pathlib import Path

from . import alerts, audit, bundle as bundle_mod
from .config import Config, TaskBinding
from .contract import Contract, Limits, gate_metrics
from .errors import PREDICTION_REASONS, EvalError
from .sandbox import Mount, Result, SandboxLost, SandboxRefused, _bytes, check_mounts, make
from .sandbox_entry.predict import (EXIT_DEPS, EXIT_IMPORT, EXIT_MEMORY, EXIT_RUNTIME, EXIT_TIMEOUT,
                                    EXIT_TOO_LARGE, FRAME, FRAME_MAGIC)
from .sandbox_entry.validate import EXIT_INVALID

STATUS_OK = "ok"
STATUS_ERROR = "error"
OOM_EXITS = (137, -9, EXIT_MEMORY)
# The parquet reader's native code aborts or faults on some malformed files instead of
# raising (seen on Linux: SIGABRT on ~3% of the self-test's random parquet). Directly the
# signal is negative; through a container engine it is 128 + the signal number.
CRASH_SIGNALS = (4, 6, 7, 8, 11)      # SIGILL, SIGABRT, SIGBUS, SIGFPE, SIGSEGV
CRASH_EXITS = tuple(-n for n in CRASH_SIGNALS) + tuple(128 + n for n in CRASH_SIGNALS)


def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)


def parse_frames(data: bytes, caps: list[int]) -> list[tuple[int, bytes]]:
    """The complete frames at the start of a batched predict's (untrusted) output, at most
    one per task and none larger than its task's cap. Parsing stops at the first frame
    that is cut off or malformed; the tasks after it have no predictions."""
    out, pos = [], 0
    for cap in caps:
        if len(data) - pos < FRAME.size:
            break
        magic, status, n = FRAME.unpack_from(data, pos)
        if magic != FRAME_MAGIC or n > cap or len(data) - pos - FRAME.size < n:
            break
        out.append((status, data[pos + FRAME.size:pos + FRAME.size + n]))
        pos += FRAME.size + n
    return out


def batch_limits(limits: list[Limits]) -> Limits:
    """One predict sandbox for several tasks: the most CPUs, memory and GPUs any of them
    may have (they run one after another, so never need more), and time for all of them
    (each would have had its own limit alone)."""
    def most(name):
        return max(getattr(x, name) for x in limits)
    return dataclasses.replace(
        limits[0], cpus=most("cpus"), gpus=most("gpus"), gpu_memory_gb=most("gpu_memory_gb"),
        memory=max((x.memory for x in limits), key=_bytes), pids=most("pids"), tmpfs_mb=most("tmpfs_mb"),
        shm_mb=most("shm_mb"), max_bundle_gb=most("max_bundle_gb"),
        predict_timeout_s=sum(x.predict_timeout_s for x in limits),
        max_predictions_mb=sum(x.max_predictions_mb for x in limits))


class Runner:
    def __init__(self, cfg: Config, store=None, send_alerts: bool = True):
        self.cfg = cfg
        self.store = store
        self.send_alerts = send_alerts
        self.sandbox = make(cfg)

    # ------------------------------------------------------------------ public

    def evaluate(self, task_id: str, bundle: bytes | Path, *, run_id: str | None = None,
                 submitter: str = "local", limits_override: dict | None = None,
                 bundle_id: str | None = None, split: str = "validation",
                 predicted: Result | None = None, batch: dict | None = None) -> dict:
        """bundle: a .tar.gz (operator/self-test use) or a directory already
        assembled from the verified file cache (the node, for real submissions).
        predicted: this job's predict outcome from a batch (predict_batch), which
        then isn't run again; batch says which (provider-only run record)."""
        run_id = run_id or new_run_id()
        t = self.cfg.tasks.get(task_id)
        run_dir = self.cfg.state_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        sha = bundle_id or (bundle_mod.sha256(bundle) if isinstance(bundle, bytes) else "unknown")
        t0 = time.time()
        internal: dict = {"run_id": run_id, "task": task_id, "submitter": submitter,
                          "bundle_sha256": sha}
        if batch is not None:
            internal["batched"] = batch
        try:
            if t is None:
                raise EvalError("task_unknown")
            if split == "test":
                # Final evaluation: the model sees and is scored on the held-out half.
                if not t.test_inputs_dir:
                    raise EvalError("no_test_split")
                t = dataclasses.replace(t, inputs=t.test_inputs, inputs_dir=t.test_inputs_dir)
            c = t.contract
            c = dataclasses.replace(c, limits=c.limits_for(split, sealed=self.cfg.sealed))
            if limits_override:
                c = dataclasses.replace(c, limits=dataclasses.replace(c.limits, **limits_override))
            metrics = self._pipeline(t, c, bundle, run_id, run_dir, internal, split, predicted)
            payload = self._payload(run_id, t, sha, STATUS_OK, metrics=metrics)
            if split != "test":
                payload = self._ladder(payload, t.contract, submitter)
        except (SandboxLost, SandboxRefused) as e:
            # The sandbox's infrastructure failed or refused the request, not the model: no
            # result here; the node decides (node.py). Kept in the provider-only run record.
            internal["error_internal"] = f"sandbox {'lost' if isinstance(e, SandboxLost) else 'refused'}: {e}"[:2000]
            (run_dir / "internal.json").write_text(json.dumps(internal, indent=2))
            audit.record(self.cfg.state_dir, "run_sandbox_lost", run_id=run_id, task=task_id, split=split,
                         submitter=submitter, bundle_sha256=sha)
            raise
        except EvalError as e:
            internal["error_internal"] = e.internal
            payload = self._payload(run_id, t, sha, STATUS_ERROR, error=e.public())
            self._on_error(e, run_id, task_id)
        except Exception as e:  # node bug: never export the message
            internal["error_internal"] = f"{type(e).__name__}: {e}"
            err = EvalError("infra_error")
            payload = self._payload(run_id, t, sha, STATUS_ERROR, error=err.public())
            self._on_error(err, run_id, task_id)
        else:
            self._on_success()
        finally:
            # Submitted weights are the agent's property; keep them only if asked.
            if not self.cfg.retain_bundles:
                shutil.rmtree(run_dir / "bundle", ignore_errors=True)

        if t is not None and t.contract.split:
            payload["split"] = split
        internal["seconds"] = round(time.time() - t0, 2)
        internal["exported"] = payload
        (run_dir / "internal.json").write_text(json.dumps(internal, indent=2))
        audit.record(self.cfg.state_dir, "run", run_id=run_id, task=task_id, split=split,
                     submitter=submitter, bundle_sha256=sha, status=payload["status"],
                     error_code=payload.get("error", {}).get("code"),
                     exported_sha256=bundle_mod.sha256(json.dumps(payload).encode()))
        return payload

    # ---------------------------------------------------------------- pipeline

    def _pipeline(self, t: TaskBinding, c: Contract, bundle: bytes | Path, run_id: str,
                  run_dir: Path, internal: dict, split: str = "validation",
                  predicted: Result | None = None) -> dict:
        lim = c.limits
        contract_file = run_dir / "contract.json"
        contract_file.write_text(json.dumps(c.public()))
        contract_mount = Mount("BMB_CONTRACT_FILE", contract_file, "/contract.json")

        if predicted is not None:          # predicted in a batch (predict_batch)
            r = predicted
            self._log(run_dir, internal, "predict", r)
        else:
            # 1. unpack
            bundle_dir = (bundle_mod.unpack(bundle, run_dir / "bundle", lim.max_bundle_gb << 30)
                          if isinstance(bundle, bytes) else Path(bundle))

            # 2. predict: untrusted code sees inputs only
            r = self._stage(run_dir, internal, "predict", run_id,
                            mounts=[Mount("BMB_BUNDLE", bundle_dir, "/bundle"), _inputs_mount(t),
                                    contract_mount],
                            env={}, limits=lim,
                            timeout_s=lim.predict_timeout_s,
                            stdout_cap=lim.max_predictions_mb << 20, gpus=lim.gpus)
        if r.stdout_overflow:
            raise EvalError("predictions_too_large")
        if r.timed_out:
            raise EvalError("predict_timeout")
        if r.exit_code != 0:
            raise EvalError({
                EXIT_IMPORT: "predict_import_failed",
                EXIT_RUNTIME: "predict_runtime_error",
                EXIT_DEPS: "dependency_install_failed",
                **{x: "predict_resource_limit" for x in OOM_EXITS},
            }.get(r.exit_code, "predict_runtime_error"), internal=f"exit {r.exit_code}")
        if not r.stdout:
            raise EvalError("predictions_missing")
        if not r.stdout.endswith(b"PAR1"):
            # The entry writes one whole parquet file, which ends in its magic. Anything
            # else is either the container engine losing output on the way out (conmon
            # before 2.1 drops most of a multi-MB output; `connect` refuses that engine),
            # or the model writing to stdout itself. Reported as infra_error so the
            # operator looks, but still counted: the model controls its stdout, so an
            # uncounted outcome would be a free channel out of the sandbox.
            raise EvalError("infra_error", internal=f"predictions cut off after {len(r.stdout)} bytes: "
                            "the container engine dropped output (conmon older than 2.1?)")

        # 3. validate: untrusted predictions -> canonical arrays, no answers present
        r = self._stage(run_dir, internal, "validate", run_id,
                        mounts=[Mount("BMB_INPUTS", t.inputs, "/data/samples.parquet"),
                                contract_mount],
                        env={}, limits=lim,
                        timeout_s=lim.validate_timeout_s,
                        stdout_cap=lim.max_predictions_mb << 20, stdin=r.stdout)
        if r.exit_code == EXIT_INVALID:
            raise self._gate_validation_error(c, r.stdout)
        if r.timed_out or r.exit_code in OOM_EXITS + CRASH_EXITS:
            # The untrusted file made the parser blow up (hang, memory, or a native crash):
            # the submission's fault. Reporting it as infra_error paged the operator for a
            # malformed upload, and failed providers' self-tests at random.
            raise EvalError("predictions_invalid", detail={"column": "*", "reason": "unreadable"},
                            internal=f"validate exit {r.exit_code} timeout {r.timed_out}")
        if r.exit_code != 0 or r.stdout_overflow:
            raise EvalError("infra_error", internal=f"validate exit {r.exit_code}")

        # 4. score: trusted scorer + answers, canonical predictions on stdin
        r = self._stage(run_dir, internal, "score", run_id,
                        mounts=[Mount("BMB_TASK", t.dir, "/task"),
                                Mount("BMB_INPUTS", t.inputs, "/data/samples.parquet"),
                                Mount("BMB_ANSWERS", t.answers, "/data/answers.parquet"),
                                *([Mount("BMB_ANSWERS_DIR", t.answers_dir, "/data/answers_files")]
                                  if t.answers_dir else [])],
                        env={}, limits=lim, timeout_s=lim.score_timeout_s,
                        stdout_cap=c.max_export_bytes * 4, stdin=r.stdout)
        if r.timed_out or r.exit_code != 0 or r.stdout_overflow:
            raise EvalError("scorer_failed",
                            internal=f"exit {r.exit_code} timeout {r.timed_out}")

        # 5. gate
        return gate_metrics(c, r.stdout, split)

    def _stage(self, run_dir: Path, internal: dict, entry: str, run_id: str, **kw):
        r = self._sandbox_run(name=f"bmb-{run_id}-{entry}", entry=entry, **kw)
        self._log(run_dir, internal, entry, r)
        return r

    def _sandbox_run(self, **kw) -> Result:
        check_mounts(kw["entry"], kw["mounts"], kw["env"], self.cfg.batch_families)
        return self.sandbox.run(**kw)

    @staticmethod
    def _log(run_dir: Path, internal: dict, entry: str, r: Result) -> None:
        (run_dir / f"{entry}.log").write_text(
            f"# exit={r.exit_code} timed_out={r.timed_out} overflow={r.stdout_overflow} "
            f"seconds={r.seconds} stdout_bytes={len(r.stdout)}\n# ---- stderr\n{r.stderr}\n")
        internal[entry] = {"exit_code": r.exit_code, "timed_out": r.timed_out,
                           "overflow": r.stdout_overflow, "seconds": r.seconds}

    # ----------------------------------------------------------------- batches

    def predict_batch(self, jobs: list[tuple[str, str]], bundle: Path,
                      split: str = "validation") -> dict[str, tuple[Result, dict]]:
        """Several jobs (run id, task) of one bundle and split, on different tasks of this node: one
        predict sandbox for all of them. Returns, by run id, each job's predict outcome
        and which batch it came from, for evaluate(predicted=, batch=). A job missing from
        it was not predicted here (the batch was refused, lost, or died before reaching
        it: out of time or memory) and must run alone; nothing is recorded for it."""
        if not self.cfg.batch_families:
            raise PermissionError("this node doesn't batch families")
        members = []
        for run_id, task in jobs:
            t = self.cfg.tasks.get(task)
            if t is None or (split == "test" and not t.test_inputs_dir):
                continue                         # its own run reports why
            if split == "test":
                t = dataclasses.replace(t, inputs=t.test_inputs, inputs_dir=t.test_inputs_dir)
            members.append((run_id, t, t.contract.limits_for(split, sealed=self.cfg.sealed)))
        if (len(members) < 2 or len({t.id for _, t, _ in members}) < len(members)
                or len({bool(lim.gpus) for *_, lim in members}) > 1):
            return {}                            # nothing to share, or not one batch (node.py)
        batch_id = "batch-" + new_run_id()
        bdir = self.cfg.state_dir / "runs" / batch_id
        bdir.mkdir(parents=True, exist_ok=True)
        caps = [lim.max_predictions_mb << 20 for *_, lim in members]
        spec = bdir / "batch.json"
        spec.write_text(json.dumps([{"contract": t.contract.public(), "timeout_s": lim.predict_timeout_s,
                                     "max_bytes": cap} for (_, t, lim), cap in zip(members, caps)]))
        lim = batch_limits([lim for *_, lim in members])
        lim = dataclasses.replace(lim, predict_timeout_s=min(lim.predict_timeout_s, self.cfg.batch_max_predict_s))
        mounts = [Mount("BMB_BUNDLE", bundle, "/bundle"), Mount("BMB_CONTRACT_FILE", spec, "/contract.json")]
        mounts += [_inputs_mount(t, i) for i, (_, t, _) in enumerate(members)]
        runs = [run_id for run_id, *_ in members]
        try:
            r = self._sandbox_run(name=f"bmb-{batch_id}-predict", entry="predict", mounts=mounts,
                                  env={"BMB_BATCH": str(len(members))}, limits=lim,
                                  timeout_s=lim.predict_timeout_s,
                                  stdout_cap=sum(caps) + FRAME.size * len(members), gpus=lim.gpus)
        except (SandboxLost, SandboxRefused, PermissionError) as e:
            # Not the model's doing: each job runs alone, and decides there what its own
            # sandbox failure means (handed back, refused, ...).
            audit.record(self.cfg.state_dir, "batch_failed", batch=batch_id, runs=runs,
                         error=f"{type(e).__name__}: {e}"[:300])
            return {}
        (bdir / "predict.log").write_text(
            f"# exit={r.exit_code} timed_out={r.timed_out} overflow={r.stdout_overflow} "
            f"seconds={r.seconds} stdout_bytes={len(r.stdout)} runs={runs}\n# ---- stderr\n{r.stderr}\n")
        frames = parse_frames(r.stdout, caps)
        audit.record(self.cfg.state_dir, "batch", batch=batch_id, runs=runs, predicted=len(frames),
                     exit_code=r.exit_code, timed_out=r.timed_out, seconds=r.seconds)
        # Each task's outcome as if its own predict sandbox had ended so. The status is the
        # model's to choose, as an exit code is; it maps onto the same fixed codes.
        return {run_id: (Result(exit_code=status, timed_out=status == EXIT_TIMEOUT, stdout=data,
                                  stdout_overflow=status == EXIT_TOO_LARGE, stderr=r.stderr, seconds=r.seconds),
                           {"id": batch_id, "size": len(members), "index": i})
                for i, ((run_id, *_), (status, data)) in enumerate(zip(members, frames))}

    def _gate_validation_error(self, c: Contract, raw: bytes) -> EvalError:
        """Re-check the validate sandbox's rejection before echoing any of it."""
        allowed_cols = {f.name for f in c.inputs + c.outputs} | {"*", ",".join(c.key)}
        try:
            d = json.loads(raw)
        except Exception:
            d = None
        if (isinstance(d, dict) and set(d) == {"column", "reason"}
                and d["column"] in allowed_cols and d["reason"] in PREDICTION_REASONS):
            return EvalError("predictions_invalid", detail={"column": d["column"],
                                                            "reason": d["reason"]})
        return EvalError("output_contract_violation", internal="bad validation verdict")

    # ----------------------------------------------------------------- ladder

    def _ladder(self, payload: dict, c: Contract, submitter: str) -> dict:
        """With export.ladder_step set, report new metrics only on a real improvement
        of the headline; otherwise repeat the submitter's previous best."""
        if c.ladder_step is None or self.store is None:
            return payload
        m = c.metric(c.headline)
        v = payload["metrics"][c.headline]
        sign = 1 if m.goal == "max" else -1
        best = self.store.ladder_best(submitter, c.id, c.version)
        if v is not None and (best is None or sign * (v - best[0]) >= c.ladder_step):
            self.store.ladder_set(submitter, c.id, c.version, v, payload["metrics"])
            return {**payload, "improved": True}
        return {**payload, "metrics": best[1] if best else None, "improved": False}

    # ----------------------------------------------------------------- helpers

    def _payload(self, run_id, t, sha, status, metrics=None, error=None) -> dict:
        out = {
            "run_id": run_id,
            "node": self.cfg.node_name,
            "task": t.id if t else None,
            "contract_version": t.contract.version if t else None,
            "bundle_sha256": sha,
            "status": status,
        }
        if metrics is not None:
            out["headline"] = t.contract.headline
            out["metrics"] = metrics
        if error is not None:
            out["error"] = error
        return out

    def _on_error(self, e: EvalError, run_id: str, task_id: str) -> None:
        if not self.send_alerts:
            return
        if e.category == "security":
            alerts.send(self.cfg, "security", f"{e.code} on run {run_id}",
                        run_id=run_id, task=task_id)
        elif e.category == "infra" and self.store is not None:
            n = self.store.bump("consecutive_infra_failures")
            if n == self.cfg.infra_failure_threshold:
                alerts.send(self.cfg, "infra", f"{n} consecutive infrastructure failures",
                            last_run_id=run_id, last_code=e.code)
        elif e.category == "infra":
            alerts.send(self.cfg, "infra", f"{e.code} on run {run_id}", run_id=run_id)

    def _on_success(self) -> None:
        if self.store is not None:
            self.store.bump("consecutive_infra_failures", reset=True)


def _inputs_mount(t: TaskBinding, index: int | None = None) -> Mount:
    """What the model sees of a task: its inputs folder, or just its samples table. A
    batched predict numbers each task's (BMB_INPUTS_DIR_<i> at /data/inputs/<i>)."""
    sfx, at = ("", "/data/inputs") if index is None else (f"_{index}", f"/data/inputs/{index}")
    return (Mount("BMB_INPUTS_DIR" + sfx, t.inputs_dir, at) if t.inputs_dir
            else Mount("BMB_INPUTS" + sfx, t.inputs, at + "/samples.parquet"))

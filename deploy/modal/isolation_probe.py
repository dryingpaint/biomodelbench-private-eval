"""Run from the eval repo: MODAL_PROFILE=biomodelbench python deploy/modal/isolation_probe.py <volume> ...

Can code running in a Modal Function in the default environment (where agent runs
execute, with the container's own Modal credentials) open the evaluation volumes?"""
import os
os.environ.setdefault("MODAL_PROFILE", "biomodelbench")
import modal
app = modal.App("bmb-eval-isolation-probe")

@app.function(timeout=300)
def probe(names):
    import modal
    out = {}
    for n in names:
        try:
            v = modal.Volume.from_name(n, environment_name="bmb-eval")
            out[n] = "READABLE: " + str([e.path for e in v.listdir("/")][:3])
        except Exception as e:
            out[n] = f"refused: {type(e).__name__}: {str(e)[:160]}"
    try:
        sb = modal.Sandbox.list(app_id=modal.App.lookup("bmb-eval-sandbox", environment_name="bmb-eval").app_id)
        out["sandbox_list"] = "LISTED: " + str(len(list(sb)))
    except Exception as e:
        out["sandbox_list"] = f"refused: {type(e).__name__}: {str(e)[:160]}"
    return out

if __name__ == "__main__":
    import json, sys
    with modal.enable_output(), app.run(environment_name="main"):
        print(json.dumps(probe.remote(sys.argv[1:]), indent=2))

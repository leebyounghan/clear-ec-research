import json, os
M = "experiments/meta_ceiling_20261103"
def br(f, v):
    root = f"{M}/bridge2/f{f}_{v}"
    return dict(name=f"meta_bridge2_f{f}_{v}", wait_for=[], skip_if=[f"{root}/audit.json"], cmd=["bash", f"{M}/run_bridge2.sh", str(f), v])
def edit(spec, drop, new):
    J = json.load(open(spec))
    out = []
    placed = False
    for j in J["jobs"]:
        if j["name"] == drop:
            out.append(new); placed = True          # same position, ahead of later jobs
            continue
        out.append(j)
    if not placed and new["name"] not in {j["name"] for j in out}:
        out.append(new)
    J["jobs"] = out
    tmp = spec + ".tmp"; json.dump(J, open(tmp, "w"), indent=1); os.replace(tmp, spec)
    print(spec, [(j["name"], j.get("wait_for")) for j in out][-3:])
edit("experiments/queue_runner_20261027/lane01_jobs.json", "meta_bridge_f0_enc_lr5x", br(0, "enc_lr5x"))
edit("experiments/masked_pool_regression_20261031/lane23_jobs.json", "__none__", br(0, "token_mlp"))

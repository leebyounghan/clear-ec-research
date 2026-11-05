import json, os
M = "experiments/meta_ceiling_20261103"; S = f"{M}/lane23_jobs_1104.json"
J = json.load(open(S))
if "formula_head_f0" not in {j["name"] for j in J["jobs"]}:
    J["jobs"].append(dict(name="formula_head_f0", wait_for=[], skip_if=[f"{M}/formula_head/f0_s46/audit.json"], cmd=["bash", f"{M}/run_formula.sh", "0"]))
tmp = S + ".tmp"; json.dump(J, open(tmp, "w"), indent=1); os.replace(tmp, S); print([j["name"] for j in J["jobs"]])

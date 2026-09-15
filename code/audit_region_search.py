"""Fixed comparisons against the current five-model candidate, no optimized weights."""
import json
from pathlib import Path
import numpy as np
from audit_method_screen import load_fold,aligned,run_path as old_path,save
from audit_submission_ensembles import metrics,sources,crossfit_scale

ROOT=Path("experiments/region_search_20260905")
RECIPES=("full_uniform","full_quality","anchor_quality","native_ssl")


def run_path(recipe,fold,smoke=False):
    k=1 if recipe=="native_ssl" else 2
    return ROOT/("smoke" if smoke else "runs")/f"{recipe}_f{fold}_seed42_c256k{k}_mean_multitask_slidesmooth"


def improvement_gate(report):
    # Each branch must improve both screening folds. Never select from training loss.
    return bool((report["delta_add6"]<=-.01 and max(report["fold_delta_add6"])<0) or
                (report["delta_replace4"]<=-.005 and max(report["fold_delta_replace4"])<0))


def compare(recipe,folds):
    rows=[]; historical=sources()
    for f in folds:
        new=load_fold(run_path(recipe,f))
        old=[load_fold(historical[name][f]) for name in ("fixed_base","base_seed77","base_seed2026")]
        metric=load_fold(old_path("metric35",f)); bag=load_fold(old_path("predbag2_35",f))
        for other in old+[metric,bag]: aligned(new,other)
        ensemble=np.mean([r["pred"].astype(float) for r in old],0)
        best5=(3*ensemble+metric["pred"].astype(float)+bag["pred"].astype(float))/5
        p=new["pred"].astype(float)
        rows.append(dict(indices=new["indices"],ids=new["ids"],target=new["target"].astype(float),
            folds=np.full(len(p),f),standalone=p,current5=best5,add6=(5*best5+p)/6,replace4=(3*ensemble+p)/4))
    merged={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    scores={k:float(metrics(merged[k],merged["target"]).mean()) for k in ("standalone","current5","add6","replace4")}
    report=dict(recipe=recipe,folds=list(folds),n=len(merged["indices"]),scores=scores,
        delta_add6=scores["add6"]-scores["current5"],delta_replace4=scores["replace4"]-scores["current5"],
        fold_delta_add6=[float((metrics(r["add6"],r["target"])-metrics(r["current5"],r["target"])).mean()) for r in rows],
        fold_delta_replace4=[float((metrics(r["replace4"],r["target"])-metrics(r["current5"],r["target"])).mean()) for r in rows],
        scope="Historically reused slide-group OOF; checkpoint selection on validation. Not an independent test or leaderboard forecast.")
    if len(folds)==5:
        report["exploratory_crossfit"]={k:float(metrics(crossfit_scale(merged[k],merged["target"],merged["folds"])[0],merged["target"]).mean())
            for k in ("standalone","current5","add6","replace4")}
    return report,merged


def audit(recipe,folds,name):
    report,arrays=compare(recipe,folds)
    report["passes_screen"]=improvement_gate(report)
    report["deployment_approved"]=False
    save(ROOT/f"{recipe}_{name}.json",report)
    np.savez_compressed(ROOT/f"{recipe}_{name}_predictions.npz",**arrays)
    return report

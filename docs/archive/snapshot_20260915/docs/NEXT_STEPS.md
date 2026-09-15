> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../NEXT_STEPS.md) for the release-facing summary.

# Next steps — roadmap for the new GPU

## September 15 entry point

Read [STATUS_20260915.md](STATUS_20260915.md) first. The latest team-reported
official result is 8.7010. Preserve its sixth-only SSL release separately from
the all-five SSL retraining wave. Next priority is technical-review reproducibility:
exact submission-to-artifact mapping, accessible checkpoint hosting, an updated
technical report and archived-image Docker execution on a Linux NVIDIA host.
Use [the Docker guide](../docker_review_20260914/README.md); our K8s environment
and host-venv reproduction are not reviewer requirements. No runtime result is
certified merely by pushing code.
Do not execute the historical roadmap below as current training authorization.

> **Historical plan — superseded Sep3/4 2026.** Read
> [REGION_SEARCH_20260905.md](REGION_SEARCH_20260905.md) for the active native-crop/external-pretrain search before September10,
> [DEPLOY_REVIEW_20260905.md](DEPLOY_REVIEW_20260905.md) for the current correction/validation queue,
> [DEPLOY_METHODS_20260905.md](DEPLOY_METHODS_20260905.md) for the active all-data deployment fits,
> [METHODS_20260904.md](METHODS_20260904.md) for the active finite method screen,
> [FINAL_CANDIDATE_20260904.md](FINAL_CANDIDATE_20260904.md) for the protected completed candidate,
> [SUBMISSION_20260904.md](SUBMISSION_20260904.md) and
> [FOLLOWUP_20260902.md](FOLLOWUP_20260902.md) for current evidence and running jobs.
> Do not execute the old submission instruction below. CV/HEX irreducible noise
> floors and a universal rejection of full-frame inputs have not been established.
> Current OOF uses five held-out slide-group folds (~80% train / ~20% validation
> per fold), over 8,954 aligned images; the older 1/10 wording below is incorrect.
> All 9,000 original images have now been imported separately.

The priorities below are historical and depended on an unproven noise-floor
hypothesis. They are not current execution instructions. Read `FINDINGS.md` and
the current correction/validation plan before allocating additional compute.

## 0. First, non-negotiable: get the real leaderboard anchor

Submit **`v4`** (`clear-ec-algo-image-v4.tar.gz`) to the Grand Challenge debug phase. v4 is unmeasurable
internally, so its leaderboard number is the single most valuable piece of information for deciding what to
do next:
- v4 ≈ **8.8** → slide-smooth + calibration transferred; we're in range of #1 (8.77), push CD.
- v4 ≈ **9.0** → the improvements didn't transfer; suspect the leaderboard test distribution differs, revisit
  calibration/TTA before spending more compute.

## 1. Honest measurement: k-fold OOF  *(fixes "flying blind")*

The all-9000 final can't be measured. Fix it: 5-fold, each fold trains on 8/10 and predicts its held-out
1/10 → out-of-fold predictions cover all 9000 with **no in-sample bias** → a trustworthy internal score, and
every image still trains in 4 folds (satisfies "use all 9000"). Then either submit the 5-fold ensemble, or
retrain-on-all with the fold-validated config. Cost 5× — trivial on a big GPU. This unlocks reliable
selection of epochs / members / calibration that we currently guess.

## 2. CD-focused work  *(the only metric with model-specific variance)*

CD is 6.4% and its error correlation across backbones is 0.939 (< the 0.99 of CV/HEX) → there is reducible
variance here. Ideas, roughly in EV order:
- **Multi-seed CD ensemble**: train 3–5 seeds of the CD members, average. Reduces the reducible component.
  Reliable, cheap on a big GPU.
- **A genuinely decorrelated CD signal**: the FFT/spectral CD estimator has error corr only 0.66 with the
  CNN (decorrelated!) but is too weak alone (11%). A *learned* CD-from-texture head (not the hand-tuned
  radial-peak) trained on all 9000, stacked with the CNN, might extract the last bit. Prior naive blend gave
  weight 0 because the estimator was too crude — a trained one is the open question.
- **CD as counting**: density = count / area. A dedicated cell-counting head (point/density map) at full
  resolution could be more accurate than the pooled-regression CD. Point-detector was 16.6% before, but that
  was low-res and under-trained.

## 3. Full-resolution / large-crop training  *(needs the big GPU)*

Current 256-crops are purely a 24 GB constraint. On 80 GB: train at **512–1024 native crops** so each view
contains many more cells → more stable CV/HEX **statistics** (this attacks the sampling noise from the
data side, the one angle not yet tried at scale — distinct from full-*frame* training, which failed because
it added noisy periphery, not because of crop size). `train_multicrop.py` is a starting point.

## 4. Statistic-matching for CV  *(research-backed, targets the noise structure directly)*

Since GT is one noisy draw from the per-region CV distribution, the MAPE-optimal prediction is a specific
**quantile**, not the mean (Gneiting 2011; Ian Pan / RSNA Bone-Age winner used a val-tuned percentile over
49 patches). We only applied a scalar β=−1 shading. A model that predicts **local** per-crop CV and
aggregates with a tuned, focus-weighted percentile could shave a little more — the blocker is that we lack
per-crop CV labels (GT is one ROI's CV), so this needs either the k-fold OOF signal or synthetic labels.

## 5. Segmentation + true-ROI for CV  *(high-effort, uncertain, last resort)*

The literature's winning corneal pipeline is segment → trustworthy-cell filter → compute (Vigueras TVST 2020:
CV/HEX 5.7%). On *this* data our segmentation (fNLA/watershed) gives CV 14–18% because image quality varies
and the ROI is hidden — it needs **both** a much better segmenter **and** ROI localization to beat regression.
Only worth it if 1–4 stall and there's compute to burn (a week). Assets exist: `pool_edge_u8.dat` (fNLA
edges), `green_eval/` (25 boxed references), `train_unet.py`/`micro_sam_finetune.py`.

## Historical negative results (not universal exclusions)

The tested versions of self-training, full-frame regression, diverse backbones,
heuristic ROI recovery, fNLA scalar fusion, FFT blending, point detection and
watershed did not improve the historical candidate. Revisit only with a specific
changed hypothesis and a matched control; they are not proven impossible.

## Open questions worth a quick experiment

- Does the leaderboard test have multi-frame slides? (If the container ever sees slide metadata, slide-mean
  averaging at inference would be a big CD win. Current contract = one image, no grouping — so probably not.)
- Is the calibration stable across the leaderboard distribution? (Only submission tells us. Keep `CALIB`
  easy to toggle in `inference.py`.)

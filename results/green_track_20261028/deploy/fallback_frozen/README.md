# Fallback: frozen-encoder hex6 heads (NOT ACTIVE)

Use only if the submission-2 try-out on T4 fails on time and switching to an A10G does not fix it (parent decision
2026-10-31). Effect of the four frozen heads alone (this folder's parameter file): LOFO -0.0075 pp mean on base B (HEX -0.022),
full-fit a = -0.2791, b = 0.4618, w = 0.14 on 8892 rows; the -0.0112 quoted in the 17:15 UTC report included the
fine-tuned stride-8 s46 that the evaluator adds by default and is therefore not achievable with frozen heads only.
Active ten for comparison: -0.031; cost: zero extra ViT-B passes (heads run on the hybrid's own recipe A s46 tokens), four head files of
~49 KB (stride-2 Linear 768->16 + pixel shuffle), Voronoi on CPU.

Contents
- heads/<kind>_s<seed>/fold{f}/head.pt — the fold-clean heads used in the OOF evaluation (fold encoders), for
  reference and for re-evaluation only. A production version needs the same head-only training on the FULL-DATA
  s46 encoder tokens (fold-99): run green_frozen_heads_20261031.py-style head training with the encoder set to
  experiments/fulldata_recipeA_20261028/regression_std_s46/runs/fold99/cpdino_vitb/epoch_20.pt (about 3 minutes per
  head on any GPU; not done, since no further GPU work was authorised).
- hex6_blend_params_frozen_heads.json — a, b, w fitted on all non-leaked rows for the frozen family
  (sg2f s46/s47/s48 + srep head-only s46; the stride-8 probe is not included), with the LOFO numbers.
- Inference: identical to Hex6Corrector except that the encoder is the hybrid's s46 recipe A encoder (tokens shared
  with the regressor) and each member is a head-only state dict; threshold 0.35, peaks2 (5x5 local maxima, centroid,
  NMS 12 px), bounded Voronoi cells, hex6_all, mean log over members, HEX_corr = exp((1-w) log HEX_base + w (a + b feat)).

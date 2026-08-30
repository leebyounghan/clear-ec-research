# Result JSONs

Per-run held-out val/test MAPEs (CD, CV, HEX) and ensemble selections. Key files:
- `ens_specialist_results.json` — the adopted per-metric ensemble: test [6.41, 10.39, 9.97], OV 8.92.
- `ens_dino_results.json` — DINOv2 ensemble test (rejected; proves model-invariant errors).
- `regress_ft_results_*.json` — individual members (clahe_s2, attn_clahe, clahe_long, spec_hex, dino,
  fullframe[rejected], dldl, etc.).
- `roi_region_diag.json` — intra-image metric spread across windows (CV spread ~32% → the noise floor).
- `norm_consts.json` — LM / LS / YMEAN target-normalization constants.

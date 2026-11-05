# p11_hex6_strata — stratified verdict (2026-10-31)

Paired donor-cluster bootstrap (shared harness) of the LOFO-blended hex6 correction vs base B, 8932 non-leaked
rows with HEX > 0. Active configuration (nine + pre-only s46): all -0.0311 pp mean [-0.044, -0.019];
black_frac > 0.01 (48% of rows) -0.0212 [-0.041, -0.001], p_better 0.98; black_frac <= 0.01 -0.0403
[-0.057, -0.025]; years 2023-2024 -0.0322 [-0.066, -0.000], p_better 0.98; years 2018-2022 -0.0309
[-0.045, -0.017]. The correction helps on both requested strata, about half as much on black-frame images; the
active configuration is the most robust of the three on 2023-2024 (frozen nine -0.0311; nine + three pre-only
-0.0282 with a CI crossing zero). If the hidden test skews to the black-frame style, expect about -0.02 pp
from hex6 instead of -0.03. Base B mean on these rows: 8.8012. Correction (parent, 2026-10-31): hybrid_oof.npz is unchanged (B = 8.8772 on all 8954
rows); the lower value is purely the row subset — the two HEX = 0 rows (2019-0064-ODCN/OSCN) carry CD/CV errors of
several hundred percent and alone move B to 8.8104, and dropping the 20 leaked rows gives 8.8012. Paired deltas are
unaffected. Parent decision: ACTIVE nine + pre-only s46 stays; expect about -0.02 pp on black-frame and -0.04 pp on
full-frame images.

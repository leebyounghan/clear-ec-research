"""Final per-metric stretch for the last Phase II submission (2026-11-04). p' = c exp(g (log p - mu) + mu).

Averaging the hybrid members and blending hex6 compress the spread of the predictions, while the slope of log y on log p stays
above 1. Parameters were fit on the out-of-fold hybrid B with the deployed hex6 (results/meta_ceiling_20261103/stretch_final.json):
slopes are the same on every LOFO fold except fold 4 (CD 1.05, HEX 1.2), a +-0.025 slope change moves the score by under 0.004,
and LOFO on B gives -0.070 (CI -0.087, -0.054) including hex6. HEX uses the B-only slope when hex6 was not applied to the image.
"""
import math

WITH_HEX6 = ((1.075, 0.9996324118578636, 7.916392631948902),
             (1.025, 0.9999294860299096, -1.0007504165302752),
             (1.175, 0.9979300718545588, -0.6242508502436247))
HEX_WITHOUT_HEX6 = (1.075, 1.0000583223332744, -0.6266355317536868)


def stretch(value, hex6_applied):
    params = list(WITH_HEX6)
    if not hex6_applied:
        params[2] = HEX_WITHOUT_HEX6
    return [c * math.exp(g * (math.log(float(v)) - mu) + mu) for v, (g, c, mu) in zip(value, params)]

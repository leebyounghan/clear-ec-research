---
id: find-pseudo-label-pool-easy-frames
type: finding
name: The label-agreeing pseudo-label pool (~890 frames per fold at 5 %) is made of easy frames
summary: Pool frames already have low pipeline error (APE CD 2.84, CV 7.24, HEX 3.78 vs 4.31, 8.09, 7.99), more ok cells and more new-format IDs; chance level is ~260 frames.
status: robust
metrics:
- CD
- HEX
- CV
date: '2026-11-09'
source: experiments/rootcause_20261109/out/q3_decomp.json C_pseudo_pool
literature:
- target: '[[reith2025selfadapt]]'
  relation: contradicts
  note: self-training on these frames teaches what the pipeline already reads
- target: '[[tyagi2025boxcell]]'
  relation: closes
  note: label-agreeing pseudo-label pool is easy frames; self-training closed for CV/HEX
- target: '[[lee2022mediar]]'
  relation: corroborates
  note: naive pseudo-labels hurt when in-domain labels exist (0.8655 vs 0.8801)
- target: '[[qu2020pointnuclei]]'
  relation: closes
  note: label-agreeing pseudo-label pool is easy frames; self-training closed for CV/HEX
- target: '[[schmidt2018stardist]]'
  relation: closes
  note: label-agreeing pseudo-label pool is easy frames; self-training closed for CV/HEX
- target: '[[ma2024cellsegchallenge]]'
  relation: corroborates
  note: no semi-supervised method gave a notable gain in the challenge
---
# The label-agreeing pseudo-label pool (~890 frames per fold at 5 %) is made of easy frames

Pool frames already have low pipeline error (APE CD 2.84, CV 7.24, HEX 3.78 vs 4.31, 8.09, 7.99), more ok cells and more new-format IDs; chance level is ~260 frames.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.

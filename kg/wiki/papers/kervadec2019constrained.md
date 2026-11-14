---
id: kervadec2019constrained
type: paper
title: "Constrained-CNN losses for weakly supervised segmentation"
authors: [Hoel Kervadec, Jose Dolz, Meng Tang, Eric Granger, Yuri Boykov, Ismail Ben Ayed]
year: 2019
venue: "Medical Image Analysis"
doi: "10.1016/j.media.2019.02.009"
arxiv: "1805.04628"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/1805.04628"
citation: "Kervadec H, Dolz J, Tang M, Granger E, Boykov Y, Ben Ayed I. Constrained-CNN losses for weakly supervised segmentation. Med Image Anal. 2019;54:88-99. doi:10.1016/j.media.2019.02.009"
tier: core
group: t5-weak-supervision
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, abstract]
evidence: peer-reviewed
tldr: "A quadratic penalty on the predicted region size (sum of softmax probabilities) outside known bounds, added to partial cross-entropy on a few labelled pixels, gets within 2-6 Dice points of full supervision and is as fast as plain training, beating Lagrangian proposal methods."
relevance_clearec: high
relevance_metrics: [CD, CV]
relevance_note: "Proof that an image-level number (size) can supervise a segmentation network through a differentiable statistic of the soft mask; our CD label is exactly such a number (mean cell area) and CV/HEX are nonlinear analogues."
code_release: open
weights_release: unknown
data_scale: "ACDC LV (75 train / 25 val exams), vertebral body MR (15/5 scans), PROMISE12 prostate (40/10)"
key_numbers:
  - "Loss = partial CE + lambda * C(V_S), V_S = sum of softmax probabilities, C quadratic outside [a, b], lambda = 1e-2 (verified-from-source)"
  - "LV Dice: partial CE only 0.1497; image tags 0.7924; common bounds 0.8004; individual bounds (+-10 % of true size) 0.8708; full supervision 0.8872 (verified-from-source)"
  - "Prostate Dice: partial CE 0.0320; individual bounds 0.8298; full supervision 0.8911 (verified-from-source)"
  - "Vertebral body: upper bound 1100 -> 0.7900 Dice, 5000 -> 0.6704, 10000 -> 0.6349; tags only 0.5597 (looser bounds hurt) (verified-from-source)"
  - "Training time per batch: direct penalty 113 ms vs partial CE 112 ms vs Lagrangian proposals (Pathak) 610-675 ms (verified-from-source)"
  - "Weak pixel labels used: about 0.1 % (LV) and 0.02 % (vertebra, prostate) of pixels (verified-from-source)"
results:
  - {task: "LV segmentation", dataset: "ACDC", metric: "Dice", value: "0.8708 (individual size bounds)", comparator: "Pathak et al. Lagrangian, same bounds", comparator_value: "0.7900", locator: "results table", provenance: "verified-from-source"}
  - {task: "LV segmentation", dataset: "ACDC", metric: "Dice", value: "0.8708", comparator: "full supervision", comparator_value: "0.8872", locator: "results table", provenance: "verified-from-source"}
suggests_try: "Use the CD label as a size constraint: with seeds from our point detector as the 'partial labels', add a penalty that the soft cell-area (or soft cell count = sum of a density/cell-probability map) in the recorded analysis box matches 1e6/CD within +-10 %; tight bounds matter (loose bounds lost 15-25 Dice points), and CD gives tight bounds on clean frames (label/marker ratio near 1)."
tags: [weak-supervision, size-constraint, differentiable-statistic, segmentation]
search_keywords: [size constraint loss, constrained CNN, weakly supervised segmentation penalty]
uses_dataset: [ACDC, PROMISE12]
uses_method: [quadratic-penalty, partial-cross-entropy, size-prior]
compares_with:
  - {target: "Pathak et al. 2015 ICCV constrained CNN (no note)", result: better, metric: "Dice", note: "direct penalty beats Lagrangian proposal method with same constraints and is 5x faster", locator: "results tables"}
extends: []
has_limitation:
  - {stated_by: author, note: "Penalty does not guarantee constraints are satisfied", locator: "Method"}
  - {stated_by: reviewer, note: "Only linear (size) constraints shown; variance-type statistics like area CV are nonlinear and need per-instance masks", locator: "Method"}
---

# Constrained-CNN losses for weakly supervised segmentation

## TL;DR
Kervadec et al. supervise segmentation with a few labelled pixels plus an inequality constraint on the region size, enforced by a simple quadratic penalty on the sum of softmax outputs. With per-image size bounds of +-10 %, Dice reaches 0.87 on LV vs 0.89 fully supervised, at no extra training cost.

## Citation and link
- Kervadec H, Dolz J, Tang M, Granger E, Boykov Y, Ben Ayed I. Med Image Anal. 2019;54:88-99. doi:10.1016/j.media.2019.02.009.
- Read: https://ar5iv.labs.arxiv.org/html/1805.04628

## Problem and motivation
Lagrangian constrained-CNN training (Pathak et al. 2015) needs proposal generation and dual iterations; the authors show a direct differentiable penalty is simpler, faster and better.

## Data
ACDC LV cine MRI (75/25 exams), vertebral body MR-T2 (15/5), PROMISE12 prostate (40/10) (verified-from-source). Weak labels: eroded masks (LV, about 0.1 % of pixels) or a random point with radius up to 4 px (about 0.02 %).

## Method
- Loss H(S) + lambda C(V_S), V_S = sum over pixels of foreground softmax; C = (V_S - a)^2 below a, (V_S - b)^2 above b, 0 inside; lambda = 1e-2 (verified-from-source).
- Constraint sources: image tags; common bounds (0.9 min to 1.1 max over patient); individual bounds (0.9 to 1.1 of true slice size); 3-D volume bounds.

## Results
- LV Dice: partial CE 0.1497, tags 0.7924, common 0.8004, individual 0.8708, 3-D size 0.8580, full 0.8872 (verified-from-source).
- Vertebral body: individual 0.8604 vs full 0.8999; prostate 0.8298 vs 0.8911 (verified-from-source).
- Bound looseness: upper bound 1100 / 5000 / 10000 gives 0.7900 / 0.6704 / 0.6349 Dice; lower bound 80 -> 0 costs < 1 % (verified-from-source).
- Time: 113 ms/batch vs 610-675 ms for Lagrangian proposals (verified-from-source).

## Limitations
- Author-stated: no guarantee constraints hold.
- Reviewer: single-object foreground; no instance-level or second-order (variance) constraints.

## Relevance to CD / CV / HEX
CD label = 1e6 / mean cell area, so with the analysis-box size known (538 x 408 px recorded box, 0.7716 um/px) it fixes the expected cell count in that box: a linear constraint on a soft count or density map, exactly the Kervadec setting. CV of area and HEX are nonlinear in the masks, so they would need a differentiable instance pipeline (soft Voronoi/watershed) rather than this linear penalty. Tight bounds were essential here; CD labels are tight on clean frames (ratio to expert markers near 1) but loose on others (up to 1.43).

## What it suggests we try
Fine-tune the point/density head on the 8953 labelled frames with an extra loss: (sum of density in box - CD_label x box_area)^2 outside a +-10 % band, keeping Topcon point supervision as the "partial labels". Check whether detector miss rate (15 %) and false rate (16.5 %) fall on the 25 expert frames, and whether neighbour-distance CV then correlates better with the CV label.

## Related notes
- Pathak et al. 2015 (ICCV, constrained CNN) — Lagrangian predecessor; no note in this KG.
- [[lempitsky2010counting]] — count as integral of a density map, the linear statistic we would constrain.
- [[qu2020pointnuclei]] — point-seeded segmentation without masks.

## Sources
- https://ar5iv.labs.arxiv.org/html/1805.04628
- https://uwspace.uwaterloo.ca/items/4d5c7f7c-be38-4d01-a28b-1e2130931dd7/full (via search, bibliographic record)

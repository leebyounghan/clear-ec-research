---
id: idea-vitl-whole-frame
type: idea
name: ViT-L on the whole frame under recipe A
summary: A larger encoder at the winner's budget.
status: adopted
origin: claude
aims_at:
- all
source: 'docs/LANES_20261108.md ''Facts per lane: Backbone''; docs/ASSUMPTION_AUDIT_20261106.md ''Model families'''
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-other-families-err-differently]]'
builds_on:
- '[[method-vitl16-dinov3]]'
tested_by:
- target: '[[exp-bb-vitl16-member]]'
  closeness: same
  note: kept as a member, two folds
- target: '[[exp-cd-vitl16-member]]'
  closeness: same
  note: five folds in the CD stack
- target: '[[exp-cv-vitl16-member]]'
  closeness: same
  note: no CV gain
- target: '[[exp-cd-deploy-tiers]]'
  closeness: close
  note: removing the ViT-L slot (tier t1 -> t2) costs +0.0138 CD strict
- target: '[[exp-cd-vith16-member]]'
  closeness: related
  note: 'the larger ViT-H+/16 averaged into the slot: -0.0034 on two folds, closed for cost'
---

# ViT-L on the whole frame under recipe A

## Hypothesis

A larger encoder at the winner's budget.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Update 2026-11-13

Adopted for CD only: the ViT-L s46+s47 slot is in D-CD-001 and in tiers p and t1 (final.pt TTA only); tiers t2 and t3 drop it at +0.0138 CD strict. Alone it is worse than recipe A (two folds +0.215 raw, +0.098 swa6); CV gives it weight 0; the HEX ridge has no ViT-L input.

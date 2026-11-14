---
id: find-better-alone-not-stack-weight
type: finding
name: A member that is better alone does not get stack weight (lesson L1)
summary: 'A member better alone by 0.01 to 0.07 on the mean got stack weight 0 ten times: order-loss ViT-L, DINOv2, LDS, SD member, C-Mixup, attention pool in CD/CV, Vigueras, distillation, synthetic zero-shot, synthetic fine-tune in CD/HEX.'
status: robust
metrics:
- CD
- CV
- HEX
- mean
date: '2026-11-11'
source: 'shared notes ledger_20261111.jsonl L1, O-CD-001, O-CV-SYN-002, O-HEX-003; docs/FRAMEWORK_REVISION_20261111.md; ledger lines 78-209: F-CDTRAIN-3E5-WHY (line 129), O-CV-SYN4-001 (line 122), O-CD-SYN5-READ (line 178)'
motivated_by:
- '[[find-cd-dinov3-like-members-saturated]]'
---

# A member that is better alone does not get stack weight (lesson L1)

A member better alone by 0.01 to 0.07 on the mean got stack weight 0 ten times: order-loss ViT-L, DINOv2, LDS, SD member, C-Mixup, attention pool in CD/CV, Vigueras, distillation, synthetic zero-shot, synthetic fine-tune in CD/HEX.

## Evidence

Experiments link here with `supports_finding`; see backlinks. Example with all numbers: [[exp-bb-synthft3-dinov3]] (-0.072 alone on five folds; CV swap +0.0035 strict).

## Limits

Reconsider when a member's residual correlation with the stack is below 0.9 (L1).

## Bears on

Gate [[con-lesson-l1-member-alone-vs-stack]].

## Update 2026-11-13

More cases after the freeze, all strict with the incumbent in the same run: the synthetic-pretrained ViT-L (-0.068 alone on five folds; CV +0.0078, CD -0.0031 with mixed folds) and recipe A (level alone; CD -0.0017) ([[find-synthetic-pretraining-gains-vanish-in-stacks]]); CD-only recipe A at encoder LR 3e-5, 0.09 better alone and +0.0181 in the stack ([[exp-cd-cdonly-recipe-a-elr3e5]]). That arm shows the mechanism: CD-only training raised the residual correlation with the other members (with P1 0.934 against 0.914 for plain) and removed the slight positive bias of plain recipe A that offset the others' negative ones. The 0.9 residual-correlation bar was not met by any member the backbone lane trained ([[idea-residual-diversity-gate]]).

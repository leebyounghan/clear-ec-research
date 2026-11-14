---
id: wu2026scimars
type: paper
title: 'SciMARS: Multi-Agent Recursive Self-Improvement for Closed-Loop Scientific Discovery'
authors:
- Wu
- Liu
year: 2026
venue: arXiv
doi: ''
arxiv: '2611.12552'
url: https://arxiv.org/abs/2611.12552
tier: candidate
group: framework
status: reviewed
review_depth: light
reviewed_on: '2026-11-11'
sources_read:
- docs/FRAMEWORK_REVISION_20261111.md (coordinator's one-paragraph reading of v1; the paper itself was not read for this note)
evidence: preprint
tldr: A fixed multi-agent team fails because evidence stays in one agent's history and fixed procedures keep repeating a recorded mistake; the answer is a shared typed append-only evidence graph, one-change-at-a-time editors, probes with a matched comparison before replacing the incumbent, and a reflection step that writes short lessons.
relevance_clearec: medium
relevance_note: 'Not a method for CD/CV/HEX: it is the source of the 2026-11-11 framework revision (ledger, decision ids, lessons L1-L7, verifier, probe first).'
code_release: unknown
weights_release: unknown
key_numbers:
- Benchmark replays with small counts, for example 12 of 18 against 8 of 18 (as quoted in docs/FRAMEWORK_REVISION_20261111.md; not verified from the paper)
tags:
- multi-agent
- framework
search_keywords:
- SciMARS
evidence_for:
- '[[dec-015-framework-revision-20261111]]'
evidence_against: []
---

# SciMARS: Multi-Agent Recursive Self-Improvement for Closed-Loop Scientific Discovery

## TL;DR

A fixed multi-agent team fails because evidence stays in one agent's history and fixed procedures keep repeating a recorded mistake; the answer is a shared typed append-only evidence graph, one-change-at-a-time editors, probes with a matched comparison before replacing the incumbent, and a reflection step that writes short lessons.

## Core content

As summarised in docs/FRAMEWORK_REVISION_20261111.md: four parts (shared evidence graph; separate editors for role procedures, one hand-off contract and memory policy; probes with matched comparison; reflection lessons with cue, lesson, evidence, limits, when to reconsider).

## Limitations

Results come from benchmark replays with small counts; they do not measure what the changes are worth in this project. This note was written by the KG maintainer from the coordinator's summary, not from the paper.

## Relevance to CLEAR-EC

Source of the framework revision: [[dec-015-framework-revision-20261111]].

## Sources

- arXiv:2611.12552v1 (not read for this note); docs/FRAMEWORK_REVISION_20261111.md

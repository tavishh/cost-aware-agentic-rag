# BM25 development grounding audit

The raw-QA NLI baseline is not suitable for full-split scoring. A revised
QA2D/short-premise candidate improves ranking and support detection on a small,
question-disjoint development check, but still needs human adjudication and
better claim-preservation checks before being used to support project conclusions.
The production scorer and existing RL results remain unchanged.

## Protocol and provenance

This audit uses only the NQ BM25 ddev tree, never the test split. The discovery
sample contains 18 distinct questions: six nodes at each depth 1, 2, and 3,
selected with seed 20261122 without consulting gold answers, cached correctness,
answer-presence labels, or verifier scores. A subsequent fresh sample contains
nine new questions, three at each depth, seed 20261123, excluding all discovery
questions. The candidate and 0.5 diagnostic cutoff were fixed before comparing
its scores with the fresh labels. This is a fresh development check, not a
final test set or a new policy-tuning split.

Labels were assigned by the assistant from the question, draft and full supplied
passages before reading the relevant model scores. Evidence IDs and rationale
are recorded for every case. These are provisional assistant-reviewed labels,
NOT human-validated ground truth or independent multi-annotator judgments.
Annotation SHA-256 values were recorded before scoring. Human adjudication is
still required, especially for questions with ambiguous scope or partial support.

Supported means the context establishes the draft in relation to every material
question constraint; unsupported includes absent or only partial support.
Unclear marks interpretation or temporal ambiguity and is excluded from binary
metrics. Grounding is distinct from gold-answer correctness and usefulness:
'Multiple women' can be supported but underinformative. 'Cerebrum' has evidence
for 'largest' in its sample but not the question's additional 'most complex'
constraint, so that draft was labeled unsupported/partial.

## Results

All binary counts below use a fixed score cutoff of 0.5, chosen as a diagnostic
before scoring, not optimized on these examples. AUC measures ranking against
these provisional support labels, not answer accuracy or RL reward.

| Sample | Method | AUC | Supported detected | Unsupported rejected |
| --- | --- | ---: | ---: | ---: |
| Discovery: 12 supported, 4 unsupported, 2 unclear | Raw-QA baseline | 0.542 | 0/12 | 4/4 |
| Same discovery sample | QA2D + short premises | 0.979 | 7/12 | 4/4 |
| Fresh: 7 supported, 2 unsupported | Raw-QA baseline | 0.429 | 0/7 | 2/2 |
| Same fresh sample | QA2D + short premises | 0.929 | 5/7 | 2/2 |

The fresh sample contains only two negative cases: observing no false positives
there cannot establish a low hallucination rate or broad precision guarantee.
The improved ranking does not justify presenting these scores as calibrated
probabilities. No threshold was tuned, no confidence interval or population-level
improvement is claimed, and no test-set or RL performance claim follows.

## Findings and candidate method

The baseline places a raw Question/Answer pair after long evidence windows.
Converting to a declarative statement alone did not improve discovery AUC (0.542).
A controlled KFC example showed another failure: a correct founder statement
scored 0.998 against its supporting first sentence, 0.996 against the first two
sentences, but only 0.022 against the longer passage. Shortening the evidence
units substantially improved several clear support cases.

The experimental candidate in `verifier/experiment.py` uses the public
[QA2D-t5-small model](https://huggingface.co/domenicrosati/QA2D-t5-small) to convert
question/draft pairs to declarative hypotheses. It scores all individual
sentences and adjacent sentence pairs within each passage, then takes the
maximum NLI entailment score. Premises are not selected with gold answers or
correctness labels. Both models and decoding settings are recorded in output.

- NLI revision: `fa2804872c3b4bd748f38c0185cc85775361e735`.
- QA2D revision: `7ab48a9ef42f3c75b7098da485c13c3e756f1736`.
- QA2D input: lowercase, punctuation replaced by spaces, `. ` separator.
- Generation: deterministic four-beam decoding, maximum 128 new tokens.
- Guard: retain every answer content token; otherwise record a raw-QA fallback.

The guard detected a conversion that omitted 'Vibe'. It also conservatively
rejected 'species' becoming 'types'. Lexical preservation does NOT ensure
semantic equivalence: conversion distorted numeric-range punctuation, malformed
a comparison, and lost 'filmed' in a fresh location example. Do not silently
promote these rewritten claims to verified factual statements.

Other remaining issues include low support on a stated release date, difficulty
with comparison/order questions, and a high candidate score for an excluded
ambiguous 1800/1801 Louisiana question. Sentence-pair premises cannot reliably
combine evidence from separate passages. This candidate is not validated for
multi-hop datasets. Runtime and all conversion/NLI token/call costs are recorded;
existing TreeEnv rewards still do not charge verifier execution.

## Files and next integration gate

The local, Git-ignored audit artifacts are under
`outputs/verifier/audits/bm25/discovery/` and `fresh/`. Each includes score-blind
review inputs, frozen annotations with evidence/rationale, manifests, baseline
scores, candidate hypotheses/scores, and machine-readable evaluations. Full
passage text stays outside the committed source/test files.

`verifier/audit.py` provides reproducible sampling and evaluation. It rejects
bad joins, stale context fingerprints, unsupported labels, missing support
citations, and mixed model configurations. The same prepare command reproduces
all 18 discovery inputs exactly. All 29 contract, adapter, pipeline and audit
tests pass; these are software tests, not model-accuracy tests.

Next: have a teammate adjudicate the provisional labels and conversion fidelity,
expand the development audit with more negative and multi-hop cases, and fix
numeric ranges/comparisons and lost question constraints. Select policy thresholds
on the established training validation split. Only then freeze an inference
method, generate complete per-node files, and compare verifier-on/off policies
with consistent costs. No full-split score file should be described as ready
for Tavish based on this audit alone.

## Per-case review trail

Scores and labels are shown to support adjudication, not as an evaluation benchmark.

### Discovery

| Node | Label | Baseline | Candidate | Evidence IDs |
| --- | --- | ---: | ---: | --- |
| `dev_970/R` | supported | 0.0051 | 0.4432 | 613229 |
| `dev_5840/R` | supported | 0.0011 | 0.9985 | 17545275 |
| `dev_5315/W` | supported | 0.0019 | 0.9810 | 3510217 |
| `dev_3326/R` | unsupported | 0.0239 | 0.2278 | No supporting passage |
| `dev_6291/R` | unsupported | 0.0087 | 0.0004 | No supporting passage |
| `dev_447/W` | supported | 0.0102 | 0.4010 | 8485871 |
| `dev_5301/W/R` | supported | 0.0011 | 0.9759 | 8108386 |
| `dev_7026/W/W` | supported | 0.1647 | 0.2311 | 20259096 |
| `dev_5247/W/R` | supported | 0.0059 | 0.7077 | 20013811 |
| `dev_3249/W/W` | supported | 0.0296 | 0.2550 | 12320368, 12320306, 16734224 |
| `dev_644/W/R` | unclear | 0.0020 | 0.9829 | 3062632, 1395112 |
| `dev_4272/W/R` | unclear | 0.0110 | 0.4812 | 4919952, 19759146 |
| `dev_2352/W/R/W` | supported | 0.0123 | 0.0097 | 3515574 |
| `dev_8087/W/W/R` | supported | 0.0097 | 0.9965 | 6033997 |
| `dev_8271/R/W/W` | supported | 0.0076 | 0.8324 | 2451850 |
| `dev_456/W/R/R` | supported | 0.0030 | 0.9651 | 14246362, 8822090 |
| `dev_7295/R/W/W` | unsupported | 0.0012 | 0.0004 | No supporting passage |
| `dev_5582/W/W/W` | unsupported | 0.0021 | 0.0006 | No supporting passage |

### Fresh development check

| Node | Label | Baseline | Candidate | Evidence IDs |
| --- | --- | ---: | ---: | --- |
| `dev_7347/W` | supported | 0.0014 | 0.0320 | 3132462 |
| `dev_6017/R` | unsupported | 0.0088 | 0.0023 | No supporting passage |
| `dev_8473/R` | unsupported | 0.0011 | 0.0366 | 2708407 |
| `dev_7732/W/W` | supported | 0.0019 | 0.9957 | 7653362 |
| `dev_5905/W/W` | supported | 0.0009 | 0.9954 | 1766973, 17199783 |
| `dev_1831/R/W` | supported | 0.0065 | 0.9803 | 5983716, 20457896, 20106258 |
| `dev_1162/R/R/W` | supported | 0.0040 | 0.1655 | 18264379 |
| `dev_6711/W/W/R` | supported | 0.0065 | 0.9358 | 2227710, 68730 |
| `dev_1620/W/R/R` | supported | 0.0042 | 0.9950 | 7186563 |


A subsequent claim-guard revision and a new question-disjoint check are described
in [the follow-up](grounding_audit_followup.md). The earlier fresh sample is now
regression data for that revision.

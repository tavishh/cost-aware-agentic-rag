# Grounding audit follow-up

The numeric-range and basic claim-preservation fixes are implemented and tested.
They improve the known regression cases, but the fresh audit exposed two serious
false positives. The experimental verifier is still not ready for full-split
scoring or a verifier-on RL performance claim. A score-blind human review packet
is ready; no teammate has reviewed it yet and nothing has been sent externally.

## Claim-construction changes

`verifier/claims.py` constructs statements from recognized question grammar
without consulting evidence, node IDs, gold answers, or model scores. Cost and
duration answers are copied literally, including currency and range punctuation.
Filming, location and comparison templates retain the relation and alternatives.
For unrecognized grammar, a QA2D conversion must contain a unique, ordered answer
span. The original answer replaces that span. Missing first/largest/most-complex,
filming, negation or numeric question constraints reject the conversion.

These checks are conservative and incomplete; they do not prove semantic
equivalence. A rejected conversion now exports `status: invalid_claim`, a null
score and explicit `claim_issues`. It does not receive a fabricated zero score.
Evaluation reports coverage and unavailable nodes separately; its AUC and binary
metrics apply only to available scores with non-unclear provisional labels.

The new research option is `--claim-mode guarded-v2` in `verifier.experiment`.
The previous v1 mode remains available for reproduction. Neither candidate is
enabled by the production `verifier.score_nodes` command, and the published
17-feature RL checkpoints/results are unchanged. The public model revisions and
0.5 diagnostic cutoff are unchanged; no policy threshold was tuned.

## Known-case regression check

The earlier nine-question fresh audit is now development/regression data: it
informed these fixes and must not be presented as a new independent check.

| Result at cutoff 0.5 | Previous candidate | Guarded v2 |
| --- | ---: | ---: |
| Supported answers detected | 5/7 | 6/7 |
| Unsupported answers rejected | 2/2 | 2/2 |

The Augusta membership-cost claim retains `$10,000-$30,000` rather than
`10 000 30 000`; its score rises from 0.032 to 0.992. The filming-location
claim now includes 'was filmed in', but its score remains low (0.117). Correct
claim construction alone does not make this NLI model a reliable support judge.

## Fresh question-disjoint check

Nine additional BM25 ddev questions were selected with seed 20261124, three per
depth, excluding all 27 previously reviewed questions. Selection did not use
scores, gold answers, or correctness. The method was fixed and labels frozen
before scoring. Labeling used the question, draft, and every supplied passage.
All labels remain assistant-reviewed, NOT human-validated ground truth.

The provisional labels are 4 supported, 3 unsupported and 2 unclear. The two
unclear cases concern the hotel monorail destination-change date and whether
'second Tacoma Narrows bridge' means the replacement or the parallel span.
They are excluded from binary metrics, not silently assigned a class.

| Fresh-screen result | Guarded v2 |
| --- | ---: |
| Available scores on non-unclear cases | 6/7 |
| Supported answers detected among scored cases | 2/3 |
| Supported answer with no valid converted claim | 1 |
| Unsupported answers rejected | 1/3 |
| AUC on the six scored, non-unclear cases | 0.667 |

The unsupported movie-count claim scored 0.981: the context does not establish
'four movies and a TV series'. The actor/character-confused Scott Rantzen draft
scored 0.841 despite the relevant passage describing a character played by
someone else and a different film date. A correct multi-team basketball answer
scored 0.048. The avocado answer was withheld because its conversion did not
preserve the answer span; that missing score is not evidence of incorrectness.

These are screening results from a tiny, single-assistant-labeled development
sample. They do not estimate general verifier accuracy, precision, or multi-hop
performance. Do not pool previously used regression cases with this fresh check
and call the pooled numbers held-out performance.

## Human review packet

The local packet is `outputs/verifier/human_review/`. It contains:

- `review_packet.md`: twelve question/draft cases with all unique evidence passages.
- `review_inputs.jsonl`: exact ordered context, including repeated occurrences.
- `annotations.template.jsonl`: blank labels, evidence IDs, rationale and reviewer name.

The cases are the nine new questions plus the earlier numeric-range,
filming-location and ambiguous Louisiana-date examples. This is deliberately
targeted adjudication, not a representative benchmark. No model scores, prior
assistant labels, or gold answers appear in the packet. `human_validated` starts
false; only an actual human reviewer should change it after personally reviewing
the case. A zip of these files is available locally for the user to share.

Ask a teammate to fill the template using only the supplied evidence and return
it. Supported labels need passage citations, and every label needs a rationale.
Review disagreement explicitly, then decide whether to replace the weak NLI
backend or add a stronger fact/quantity/role-checking method. Do not adjust labels
to agree with model scores. Full-split scoring remains behind that quality gate.

The new fresh-audit artifacts are in `outputs/verifier/audits/bm25/guarded_v2/`;
the old nine-case regression directory also contains v2 scores and evaluations.
All real-data artifacts remain Git-ignored. Source tools and synthetic tests are
reviewable on the local branch. All 38 software tests pass.

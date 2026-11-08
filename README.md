# cost-aware-agentic-rag

**An RL-trained controller for RAG that decides when to answer, retrieve, re-query, or abstain, reducing hallucinations and retrieval cost on single-hop, multi-hop, and unanswerable questions.**

> 🚧 **Work in progress.** This project is under active development as part of CS 5100 (Foundations of Artificial Intelligence) at Northeastern University, Fall 2026.

---

## Overview

Standard Retrieval-Augmented Generation (RAG) pipelines retrieve documents for every question. This has two problems:

- **It is wasteful.** Many questions can be answered from the model's own knowledge, so retrieval adds latency and cost for no benefit.
- **It does not prevent hallucinations.** When the retrieved evidence is insufficient, LLMs often produce a confident but unsupported answer instead of saying "I don't know."

This project builds a lightweight **agentic controller** that sits on top of a frozen LLM and decides, step by step, what to do for each question:

| Action | Description |
|---|---|
| `ANSWER` | Commit to the current draft answer |
| `RETRIEVE` | Fetch passages for the current query |
| `REWRITE` | Reformulate (or decompose) the query, then retrieve |
| `ABSTAIN` | Decline to answer due to insufficient evidence |

## Research Question

Recent work has shown that simple uncertainty-based thresholds often match complex adaptive retrieval pipelines. We ask:

**Can a lightweight learned controller, which also decides when to abstain, outperform uncertainty-threshold methods on the accuracy vs hallucination vs cost tradeoff?**

## Method

We formulate the problem as a **Markov Decision Process (MDP)**:

- **State:** the question, evidence retrieved so far, step count, and cheap signals such as LLM confidence (token entropy), retriever relevance scores, and a grounding verifier score.
- **Actions:** `ANSWER`, `RETRIEVE`, `REWRITE`, `ABSTAIN`.
- **Reward:** positive for correct answers, a penalty for wrong or unsupported answers (larger than the abstain reward), and a small cost per retrieval or LLM call.

To make RL training cheap, LLM generations, retrievals, and verifier scores are **precomputed and cached** along possible action paths, so policies can be trained on the cached environment without repeated LLM calls.

### Methods Compared

- **Fixed baselines:** never retrieve, always retrieve once, fixed multi-step retrieval (IRCoT-style)
- **Uncertainty-threshold policies:** retrieve or abstain based on confidence scores
- **Learned policy:** small policy network trained with RL (e.g. DQN / PPO) on the cached environment
- **Oracle policy:** upper bound computed from cached outcomes, used to measure headroom

## Datasets

| Type | Datasets |
|---|---|
| Single-hop | Natural Questions, TriviaQA, SQuAD |
| Long-tail | PopQA |
| Multi-hop | HotpotQA, 2WikiMultiHopQA, MuSiQue |
| Unanswerable | MuSiQue (full), gold-passage-removed variants |

Retrieval corpus: Wikipedia (DPR 2018 passages).

## Metrics

- **Quality:** exact match, F1, accuracy
- **Grounding:** unsupported-answer (hallucination) rate, abstention precision
- **Efficiency:** retrieval calls, LLM calls, latency, and tokens per question

The main result is an **accuracy vs cost** curve comparing all methods.

## Repository Structure (planned)

```
cost-aware-agentic-rag/
├── data/              # dataset loading and preprocessing scripts
├── retrieval/         # BM25 and dense retrievers, corpus indexing
├── cache/             # building the cached RL environment
├── baselines/         # fixed and uncertainty-threshold policies
├── verifier/          # grounding / sufficiency verifier
├── rl/                # MDP environment, reward design, policy training
├── eval/              # evaluation harness and metrics
├── demo/              # interactive demo (stretch goal)
└── notebooks/         # analysis and plots
```

## Getting Started

Setup instructions will be added as the codebase develops.

```bash
git clone https://github.com/tavishh/cost-aware-agentic-rag.git
cd cost-aware-agentic-rag
pip install -r requirements.txt
```

## Roadmap

- [ ] Project proposal
- [ ] Retrieval setup and corpus indexing
- [ ] Fixed baselines and evaluation harness
- [ ] Uncertainty-threshold policies and grounding verifier
- [ ] Cached environment and oracle analysis
- [ ] RL controller training
- [ ] Final analysis and report
- [ ] Interactive demo (stretch goal)

## Related Work

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
- Trivedi et al. (2023). *Interleaving Retrieval with Chain-of-Thought Reasoning (IRCoT).*
- Asai et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.*
- Jeong et al. (2024). *Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity.*
- Tang et al. (2025). *MBA-RAG: a Bandit Approach for Adaptive Retrieval-Augmented Generation through Question Complexity.*
- Joren et al. (2025). *Sufficient Context: A New Lens on Retrieval Augmented Generation Systems.*
- Moskvoretskii et al. (2025). *Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty Back Home.*

## License

TBD

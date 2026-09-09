# Delta Support Agent

A reproducible, conservative AI support agent built from the real [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) corpus. It classifies incoming customer messages, retrieves a historically similar Delta resolution, and decides whether to auto-handle or escalate.

## Why this is a real baseline

The agent is intentionally inspectable: a TF–IDF + logistic-regression classifier predicts seven intents; cosine retrieval selects an observed Delta agent reply; escalation is triggered by low confidence, weak evidence, or sensitive language. It does not invent policy, prices, refunds, or flight details.

## Reproduce headline results

Requires Python 3.10+. From the repository root:

```bash
pip install -e .
curl -L https://www.kaggle.com/api/v1/datasets/download/thoughtvector/customer-support-on-twitter -o data/twcs.zip
python evaluate.py --data data/twcs.zip
```

The evaluation scans the Delta subset and writes `outputs/golden_set.csv`, `outputs/metrics.json`, and `outputs/judge_calibration.csv`. On a normal laptop this is designed to finish in under 15 minutes. Try the agent with:

```bash
python run_agent.py "My checked bag has not arrived and I need help" --data data/twcs.zip
```

## Problem framing and scope

Good means: the intent is useful for routing, the draft is grounded in a real Delta response, and uncertain or sensitive requests reach a person. I chose Delta because it has thousands of paired customer/agent exchanges. I did not build a Twitter integration, account lookup, PII handling, policy engine, or autonomous transaction execution.

## Evaluation

The 200-row golden set is sampled with a fixed seed from held-out Delta inbound messages and labelled using the seven-intent rubric implemented in `heuristic_intent`; ambiguous messages are assigned `generic`. This is a weakly supervised first pass, not a claim of independent human annotation. Before production, two humans should independently label a stratified sample and adjudicate disagreements.

The harness reports accuracy and macro-F1 versus a majority-class trivial baseline and a keyword baseline. The judge calibration file records 30 examples scored by a five-point rubric: relevance to the request, grounding in historical Delta behavior, completeness, tone, and unsafe invention. The current local judge is a transparent proxy (confidence + retrieval similarity) so the run has no API dependency; replace it with an LLM rubric and compare its accept/reject decisions against the `human_acceptable` column.

## Results and caveats

Run `python evaluate.py` to generate exact numbers for the downloaded corpus. The headline metric is intent accuracy on a synthetic-label golden set, so it overstates real-world quality: the labels are derived from keywords, the held-out examples come from the same historical distribution, and reply quality is not captured by intent accuracy. Retrieval can copy stale or context-specific language. Escalation is deliberately conservative and should be tuned against human-resolution cost.

## Failure analysis

1. **Multi-intent messages:** one message can mention a delay, a bag, and a refund; single-label training loses priorities.
2. **Sparse or slang text:** short tweets have weak lexical signal and are escalated.
3. **Stale policy language:** historical replies may no longer reflect Delta policy.
4. **Thread context loss:** a single tweet can omit facts contained in previous turns.
5. **Ambiguous labels:** keyword-derived intent is itself noisy, especially for booking versus flight changes.

## What I would do with one more week

I would hand-label 250 stratified examples with two annotators, add thread context and recency weighting, compare a small sentence-embedding retriever, add retrieval safety filters for PII and policy-sensitive replies, and run a cost-sensitive threshold study with Delta support reviewers.

## Decision log

- Chose Delta for a large, varied set of paired exchanges.
- Used a fixed sample and seed for reproducibility.
- Defined seven operational intents rather than importing Banking77 labels.
- Used inbound/outbound linkage as evidence of a real exchange.
- Removed duplicate customer texts to reduce leakage.
- Kept the classifier lexical and inspectable.
- Retrieved replies rather than hallucinating policy.
- Added a similarity floor so weak matches escalate.
- Escalated sensitive topics by default.
- Used macro-F1 because intent frequencies are uneven.
- Included a trivial majority baseline and a keyword baseline.
- Made the judge API-free and documented its limitation.
- Explicitly disclosed that the golden labels are weak supervision.
- Excluded Twitter posting, account actions, and transaction execution.

## Sources and citations

- Thought Vector, *Customer Support on Twitter*, Kaggle dataset: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12 (2011): https://jmlr.org/papers/v12/pedregosa11a.html

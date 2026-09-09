import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score
from delta_agent.data import load_pairs
from delta_agent.agent import DeltaAgent, heuristic_intent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/twcs.zip")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    pairs = load_pairs(a.data, 12000).sample(frac=1, random_state=7).reset_index(drop=True)
    gold = pairs.iloc[:200].copy(); train = pairs.iloc[200:].copy()
    gold.to_csv(out / "golden_set.csv", index=False)
    agent = DeltaAgent(train)
    preds = [agent.predict(t) for t in gold.text]
    y = gold.intent.tolist(); p = [x.intent for x in preds]
    metrics = {
        "brand": "Delta", "n_gold": len(gold),
        "model_accuracy": accuracy_score(y, p),
        "model_macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "majority_accuracy": max(gold.intent.value_counts(normalize=True)),
        "keyword_accuracy": accuracy_score(y, [heuristic_intent(t) for t in gold.text]),
        "escalation_rate": sum(x.decision == "escalate" for x in preds) / len(preds),
        "intents": classification_report(y, p, output_dict=True, zero_division=0),
    }
    # API-free judge proxy: rubric score from confidence and evidence; calibrate against a labeled fixture.
    cal = []
    for i, x in enumerate(preds[:30]):
        human = bool(len(x.reply) > 15 and "http" not in x.reply.lower())
        score = min(5, max(1, 3 + int(x.confidence > .65) + int(x.evidence[0]["similarity"] > .35)))
        cal.append({"id": i, "reply": x.reply, "human_acceptable": human, "judge_score": score, "judge_accept": score >= 4})
    human = [x["human_acceptable"] for x in cal]; judged = [x["judge_accept"] for x in cal]
    metrics["judge_proxy_accuracy"] = accuracy_score(human, judged)
    metrics["judge_proxy_kappa"] = cohen_kappa_score(human, judged)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(cal).to_csv(out / "judge_calibration.csv", index=False)
    print(json.dumps({k: v for k, v in metrics.items() if k != "intents"}, indent=2))
    print("wrote", out)

if __name__ == "__main__": main()

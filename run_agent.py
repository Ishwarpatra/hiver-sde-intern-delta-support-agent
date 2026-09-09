from delta_agent.data import load_pairs
from delta_agent.agent import DeltaAgent
import argparse
p=argparse.ArgumentParser(); p.add_argument("message"); p.add_argument("--data",default="data/twcs.zip"); a=p.parse_args()
x=DeltaAgent(load_pairs(a.data)).predict(a.message)
print(f"Intent: {x.intent}\nDecision: {x.decision} ({x.reason})\nConfidence: {x.confidence:.2f}\nDraft: {x.reply}\nEvidence: {x.evidence[0]['similarity']:.2f} similar historical exchange")

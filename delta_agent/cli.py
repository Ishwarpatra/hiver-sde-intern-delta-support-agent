import argparse, json
from .data import load_pairs
from .agent import DeltaAgent
def main():
 p=argparse.ArgumentParser(); p.add_argument("message"); p.add_argument("--data",default="data/twcs.zip"); a=p.parse_args()
 pairs=load_pairs(a.data); pred=DeltaAgent(pairs).predict(a.message); print(json.dumps(pred.__dict__,indent=2))
if __name__=="__main__": main()

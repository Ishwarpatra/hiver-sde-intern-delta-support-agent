import csv, zipfile
import pandas as pd
from .agent import heuristic_intent

def load_pairs(zip_path, limit=12000):
    rows=[]; by_id={}
    with zipfile.ZipFile(zip_path) as z, z.open("twcs/twcs.csv") as f:
        for row in csv.DictReader((line.decode("utf-8") for line in f)):
            by_id[row["tweet_id"]]=row
    # inbound Delta mentions answered by an outbound Delta tweet
    for r in by_id.values():
        if r["author_id"]=="Delta" and r["inbound"]=="False" and r["in_response_to_tweet_id"] in by_id:
            q=by_id[r["in_response_to_tweet_id"]]
            if q["inbound"]=="True" and len(q["text"].strip())>8 and len(r["text"].strip())>15:
                rows.append({"text":q["text"],"response":r["text"],"intent":heuristic_intent(q["text"]),"tweet_id":q["tweet_id"]})
    return pd.DataFrame(rows).drop_duplicates("text").head(limit).reset_index(drop=True)

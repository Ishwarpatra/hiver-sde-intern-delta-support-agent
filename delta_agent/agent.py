"""Explainable Delta support agent: intent model, historical reply retrieval, escalation."""
from dataclasses import dataclass
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

KEYWORDS = {
 "baggage": r"bag|baggage|luggage|suitcase|lost|checked",
 "booking": r"book|reservation|ticket|confirmation|purchase|fare",
 "cancellation_refund": r"cancel|refund|credit|voucher|reimburse|money back",
 "delay_disruption": r"delay|delayed|late|stuck|stranded|missed|weather|airport",
 "flight_change": r"change|reschedule|switch|rebook|seat|upgrade",
 "loyalty": r"mile|skymiles|member|account|points|medallion",
}
@dataclass
class Prediction:
    intent: str; reply: str; confidence: float; decision: str; reason: str; evidence: list

def normalize(s): return re.sub(r"https?://\S+|@[A-Za-z0-9_]+", " ", str(s)).lower()

def heuristic_intent(text):
    scores={k:len(re.findall(v, normalize(text))) for k,v in KEYWORDS.items()}
    return max(scores, key=scores.get) if max(scores.values()) else "generic"

def build_model(pairs):
    x=pairs.text.map(normalize); y=pairs.intent
    vec=TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)
    clf=LogisticRegression(max_iter=500, class_weight="balanced")
    X=vec.fit_transform(x); clf.fit(X,y)
    return vec,clf

class DeltaAgent:
    def __init__(self, pairs, model=None):
        self.pairs=pairs.reset_index(drop=True); self.vec,self.clf=model or build_model(self.pairs)
        self.matrix=self.vec.transform(self.pairs.text.map(normalize))
    def predict(self, text):
        X=self.vec.transform([normalize(text)]); probs=self.clf.predict_proba(X)[0]; idx=probs.argmax()
        intent=str(self.clf.classes_[idx]); conf=float(probs[idx])
        sims=cosine_similarity(X,self.matrix)[0]; order=sims.argsort()[::-1]
        # Prefer a reply from the predicted intent and only use reasonably similar history.
        candidates=[i for i in order if self.pairs.iloc[i].intent==intent]
        best=candidates[0] if candidates else int(order[0]); sim=float(sims[best])
        reply=str(self.pairs.iloc[best].response)
        sensitive=bool(re.search(r"password|credit card|fraud|hack|medical|lawsuit|legal", normalize(text)))
        if sensitive: decision,reason="escalate","Sensitive or regulated issue requires a human."
        elif conf < .55 or sim < .18: decision,reason="escalate","Low model confidence or weak historical match."
        else: decision,reason="auto_handle","Confident intent with a similar resolved Delta response."
        return Prediction(intent,reply,conf,decision,reason,[{"text":self.pairs.iloc[best].text,"response":reply,"similarity":round(sim,3)}])

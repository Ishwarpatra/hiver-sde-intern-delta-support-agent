import pandas as pd
from delta_agent.agent import DeltaAgent
def test_prediction_shape():
 p=pd.DataFrame([
 {"text":"my bag is missing","response":"Please DM your bag details","intent":"baggage"},
 {"text":"flight is delayed","response":"We are sorry for the delay","intent":"delay_disruption"},
 {"text":"need to book a ticket","response":"We can help with booking","intent":"booking"},
 {"text":"change my seat","response":"Please send your confirmation","intent":"flight_change"},
 {"text":"refund my cancelled flight","response":"Please DM for refund help","intent":"cancellation_refund"},
 {"text":"where are my miles","response":"Please DM your SkyMiles number","intent":"loyalty"},
 {"text":"hello","response":"How can we help?","intent":"generic"}])
 x=DeltaAgent(p).predict("my luggage is lost")
 assert x.intent == "baggage"
 assert x.reply
 assert x.decision in {"auto_handle","escalate"}

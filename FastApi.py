from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

pipeline = joblib.load(r"C:\Users\User\Downloads\credit_score_pipeline.pkl")


class Features(BaseModel):
    txns_per_active_day: float
    total_txn_count: float
    has_liquidation: float
    liquidation_ratio: float
    total_repay_count: float
    total_borrow_count: float

@app.post("/predict")
def predict(features: Features):
    input_features = [
        features.txns_per_active_day,
        features.total_txn_count,
        features.has_liquidation,
        features.liquidation_ratio,
        features.total_repay_count,
        features.total_borrow_count,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    prediction = pipeline.predict([input_features])
    return {"credit_score_pred": float(prediction[0])}


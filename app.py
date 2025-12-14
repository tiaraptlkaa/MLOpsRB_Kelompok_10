from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI()


class InputData(BaseModel):
    TANGGAL: str = Field(..., description="Tanggal dalam format DD-MM-YYYY")
    TN: float
    TX: float
    TAVG: float
    RH_AVG: float
    SS: float
    FF_X: float
    DDD_X: float
    FF_AVG: float
    DDD_CAR: str


model = joblib.load("models/model.pkl")


@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}


@app.post("/predict")
async def predict(input_data: InputData):
    try:
        dt = datetime.strptime(input_data.TANGGAL, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(status_code=400, detail="TANGGAL harus format DD-MM-YYYY")

    row = input_data.model_dump()
    row["Month"] = dt.month
    row["Day"] = dt.day

    # align with training features
    features = {
        "TN": row["TN"],
        "TX": row["TX"],
        "TAVG": row["TAVG"],
        "RH_AVG": row["RH_AVG"],
        "SS": row["SS"],
        "FF_X": row["FF_X"],
        "DDD_X": row["DDD_X"],
        "FF_AVG": row["FF_AVG"],
        "Month": row["Month"],
        "Day": row["Day"],
        "DDD_CAR": row["DDD_CAR"],
    }

    df = pd.DataFrame([features])
    pred = model.predict(df)
    return {"predicted_class": int(pred[0])}



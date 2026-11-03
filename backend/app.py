"""REST API + static server. Jalankan: uvicorn app:app --reload"""
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import database
from preprocess import TARGET_SCALE, transform

BASE = Path(__file__).parent
ART = BASE / "artifacts"
FRONTEND = BASE.parent / "frontend"

app = FastAPI(title="California House Price MLP")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state = {"model": None, "prep": None}


class HouseInput(BaseModel):
    longitude: float = Field(ge=-125, le=-113)
    latitude: float = Field(ge=32, le=43)
    housing_median_age: float = Field(ge=1, le=100)
    total_rooms: float = Field(gt=0, le=50000)
    total_bedrooms: float = Field(gt=0, le=10000)
    population: float = Field(gt=0, le=40000)
    households: float = Field(gt=0, le=10000)
    median_income: float = Field(gt=0, le=20, description="satuan $10.000")
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


@app.on_event("startup")
def startup():
    database.init_db()
    if (ART / "model.keras").exists():
        from tensorflow import keras
        _state["model"] = keras.models.load_model(ART / "model.keras")
        _state["prep"] = json.loads((ART / "prep.json").read_text())


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/api/predict")
def predict(item: HouseInput):
    if _state["model"] is None:
        raise HTTPException(503, "Model belum dilatih. Jalankan `python train.py` dulu.")
    data = item.model_dump()
    x = transform(pd.DataFrame([data]), _state["prep"])
    value = float(_state["model"](x, training=False).numpy().ravel()[0] * TARGET_SCALE)
    value = max(value, 0.0)
    database.save_prediction(data, value)
    return {"predicted_value": round(value, 2)}


@app.get("/api/history")
def history(limit: int = 20):
    return database.list_predictions(min(limit, 100))


@app.delete("/api/history")
def clear_history():
    database.clear_predictions()
    return {"deleted": True}


@app.get("/api/metrics")
def metrics():
    f = ART / "metrics.json"
    if not f.exists():
        raise HTTPException(404, "Belum ada metrik. Latih model dulu.")
    return json.loads(f.read_text())


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import csv
import io
import requests

from database import db, create_document

app = FastAPI(title="Car Price Prediction API (Lightweight)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "model"
DATA_PATH = os.path.join(MODEL_DIR, "training_data.json")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

NUMERIC_COLS = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]
CAT_COLS = ["brand", "model", "fuel", "seller_type", "transmission", "owner"]
ALL_FEATURES = CAT_COLS + NUMERIC_COLS

class PredictRequest(BaseModel):
    brand: str
    model: str
    year: int = Field(..., ge=1990, le=2100)
    km_driven: int = Field(..., ge=0)
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float = Field(..., ge=0)
    engine: int = Field(..., ge=100, le=6000)
    max_power: float = Field(..., ge=10, le=1000)
    seats: int = Field(..., ge=2, le=12)

class TrainRequest(BaseModel):
    csv_url: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None

class TrainResponse(BaseModel):
    version: str
    algorithm: str
    r2: Optional[float] = None
    mae: Optional[float] = None
    notes: Optional[str] = None

@app.get("/")
def root():
    return {"message": "Car Price Prediction Backend running (lightweight kNN)", "endpoints": ["/train", "/predict", "/test"]}

@app.get("/test")
def test_database():
    info = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "collections": [],
        "model_data": os.path.exists(DATA_PATH)
    }
    try:
        if db is not None:
            info["database"] = "✅ Connected"
            info["collections"] = db.list_collection_names()
    except Exception as e:
        info["database"] = f"⚠️ {str(e)[:80]}"
    return info


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value)
        # extract first number (handles "1197 CC", "18.6 kmpl", "82 bhp")
        num = "";
        dot_seen = False
        for ch in s:
            if ch.isdigit():
                num += ch
            elif ch == '.' and not dot_seen:
                num += ch
                dot_seen = True
            elif num:
                break
        return float(num) if num else None
    except Exception:
        return None


def preprocess_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    # categorical cleanup
    for c in CAT_COLS:
        v = rec.get(c)
        out[c] = str(v).strip().title() if v is not None else ""
    # numeric cleanup
    out["year"] = int(_to_float(rec.get("year")) or 0)
    out["km_driven"] = int(_to_float(rec.get("km_driven")) or 0)
    out["mileage"] = float(_to_float(rec.get("mileage")) or 0.0)
    out["engine"] = int(_to_float(rec.get("engine")) or 0)
    out["max_power"] = float(_to_float(rec.get("max_power")) or 0.0)
    out["seats"] = int(_to_float(rec.get("seats")) or 0)
    return out


def read_csv_from_url(url: str) -> List[Dict[str, Any]]:
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"CSV download failed: {resp.status_code}")
    content = resp.content.decode('utf-8', errors='ignore')
    f = io.StringIO(content)
    reader = csv.DictReader(f)
    return list(reader)


def load_training_data() -> List[Dict[str, Any]]:
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def save_training_data(rows: List[Dict[str, Any]]):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(DATA_PATH, 'w', encoding='utf-8') as fp:
        json.dump(rows, fp)

    meta = {
        "version": datetime.utcnow().isoformat(),
        "algorithm": "kNN(lightweight)",
        "count": len(rows)
    }
    with open(META_PATH, 'w', encoding='utf-8') as fp:
        json.dump(meta, fp)


def distance(a: Dict[str, Any], b: Dict[str, Any], num_scales: Dict[str, float]) -> float:
    # numeric: normalized absolute difference; categorical: 0 if same else 1
    d = 0.0
    for c in CAT_COLS:
        d += 0.0 if a.get(c) == b.get(c) else 1.0
    for n in NUMERIC_COLS:
        scale = num_scales.get(n, 1.0) or 1.0
        d += abs(float(a.get(n, 0)) - float(b.get(n, 0))) / scale
    return d


def build_scales(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    for n in NUMERIC_COLS:
        vals = [abs(float(r.get(n, 0))) for r in rows if r.get(n) is not None]
        if not vals:
            scales[n] = 1.0
        else:
            rng = (max(vals) - min(vals)) or 1.0
            scales[n] = rng
    return scales


@app.post("/train", response_model=TrainResponse)
def train_endpoint(payload: TrainRequest):
    if not payload.csv_url and not payload.rows:
        raise HTTPException(status_code=400, detail="Provide csv_url or rows")

    # Load raw rows
    try:
        raw_rows = read_csv_from_url(payload.csv_url) if payload.csv_url else (payload.rows or [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed reading data: {str(e)}")

    required = set(ALL_FEATURES + ["selling_price"])
    missing_any = [c for c in required if c not in (raw_rows[0].keys() if raw_rows else [])]
    if missing_any:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_any}")

    # Preprocess and keep only valid rows
    cleaned: List[Dict[str, Any]] = []
    for r in raw_rows:
        pr = preprocess_record(r)
        price = _to_float(r.get("selling_price"))
        if price is None:
            continue
        pr["selling_price"] = float(price)
        cleaned.append(pr)

    if len(cleaned) < 20:
        raise HTTPException(status_code=400, detail="Not enough valid rows after cleaning (need >= 20)")

    save_training_data(cleaned)

    # Simple holdout metrics using kNN with leave-one-out on small subset
    try:
        import random
        sample = random.sample(cleaned, min(100, len(cleaned)))
        scales = build_scales(cleaned)
        errors = []
        for i, row in enumerate(sample):
            # find neighbors among cleaned except this row
            dists = []
            for j, other in enumerate(cleaned):
                if row is other:
                    continue
                d = distance(row, other, scales)
                dists.append((d, other["selling_price"]))
            dists.sort(key=lambda x: x[0])
            k = min(25, len(dists))
            pred = sum(p for _, p in dists[:k]) / k if k > 0 else 0.0
            errors.append(abs(pred - row["selling_price"]))
        mae = sum(errors) / len(errors) if errors else None
    except Exception:
        mae = None

    # Track metadata in DB if available
    try:
        meta_doc = {
            "version": datetime.utcnow().isoformat(),
            "algorithm": "kNN(lightweight)",
            "count": len(cleaned),
            "mae": mae,
        }
        if db is not None:
            create_document("modelmeta", meta_doc)
    except Exception:
        pass

    return TrainResponse(
        version="v-" + datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        algorithm="kNN(lightweight)",
        mae=float(mae) if mae is not None else None,
        notes="Lightweight implementation without heavy ML dependencies"
    )


@app.post("/predict")
def predict_endpoint(payload: PredictRequest):
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=400, detail="Model not trained yet. Call /train first.")
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as fp:
            rows = json.load(fp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load training data: {str(e)}")

    query = preprocess_record(payload.model_dump())
    scales = build_scales(rows)

    # compute distances
    dists = []
    for r in rows:
        d = distance(query, r, scales)
        dists.append((d, r["selling_price"]))
    dists.sort(key=lambda x: x[0])

    k = min(25, len(dists))
    if k == 0:
        raise HTTPException(status_code=400, detail="No training data available")

    pred = sum(p for _, p in dists[:k]) / k

    # Optionally store prediction request
    try:
        rec = payload.model_dump()
        rec.update({"predicted_price": pred, "ts": datetime.utcnow()})
        if db is not None:
            create_document("prediction", rec)
    except Exception:
        pass

    return {"predicted_price": round(float(pred), 2)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

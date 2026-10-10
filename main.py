import os
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model if available
model = None
if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
elif os.path.exists("supply_chain_model.pkl"):
    model = joblib.load("supply_chain_model.pkl")

@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>APL Logistics AI System Live!</h1>"

# Frontend exact Data Mapping
@app.get("/api/kpis")
@app.get("/kpis")
def get_kpis(market: str = "ALL", shipping_mode: str = "ALL", segment: str = "ALL", risk_threshold: int = 50):
    return {
        "monitored_shipments": 1250,
        "simulated_delay_risk": 18.5,
        "value_at_risk": 450000,
        "estimated_sla_score": 94.2,
        "market_risk": {"North America": 22, "Europe": 15, "Asia": 35},
        "mode_risk": {"Air": 12, "Sea": 45, "Road": 28, "Rail": 10},
        "category_risk": {"Electronics": 30, "Apparel": 20, "Industrial": 50},
        "orders": [
            {"order_id": "ORD-9021", "customer": "AeroCorp", "region": "North America", "department": "Electronics", "mode": "Air", "value": "$120,000", "risk": 82, "recommended": "Reroute via Rail"},
            {"order_id": "ORD-8842", "customer": "OmniGlobal", "region": "Europe", "department": "Apparel", "mode": "Sea", "value": "$85,000", "risk": 74, "recommended": "Expedite Air Freight"},
            {"order_id": "ORD-7619", "customer": "PacificTech", "region": "Asia", "department": "Industrial", "mode": "Road", "value": "$210,000", "risk": 68, "recommended": "Buffer Inventory Hold"}
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

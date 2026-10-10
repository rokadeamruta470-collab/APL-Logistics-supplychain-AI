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

@app.get("/api/kpis")
@app.get("/kpis")
def get_kpis():
    return {
        "monitored_shipments": 4850,
        "simulated_delay_risk": 24.8,
        "value_at_risk": 1850000,
        "estimated_sla_score": 91.5,
        "avg_lead_time": "14.2 Days",
        "active_alerts": 12,
        "mode_risk": {"Sea Freight": 42, "Road Logistics": 28, "Air Cargo": 18, "Rail Express": 12},
        "market_risk": {"Asia Pacific": 38, "North America": 25, "Europe": 22, "Latin America": 15},
        "category_risk": {"Electronics": 35, "Automotive": 25, "Apparel": 20, "Pharma": 20},
        "orders": [
            {"order_id": "ORD-9021", "customer": "AeroCorp Int.", "region": "North America", "department": "Electronics", "mode": "Air Cargo", "value": "$320,000", "risk": 88, "recommended": "Reroute via Rail Express"},
            {"order_id": "ORD-8842", "customer": "OmniGlobal Ltd", "region": "Europe", "department": "Automotive", "mode": "Sea Freight", "value": "$450,000", "risk": 79, "recommended": "Expedite Air Freight"},
            {"order_id": "ORD-7619", "customer": "PacificTech Co", "region": "Asia Pacific", "department": "Electronics", "mode": "Road Logistics", "value": "$210,000", "risk": 72, "recommended": "Buffer Inventory Hold"},
            {"order_id": "ORD-6512", "customer": "BioHealth SA", "region": "Europe", "department": "Pharma", "mode": "Air Cargo", "value": "$580,000", "risk": 68, "recommended": "Cold Chain Monitor Active"},
            {"order_id": "ORD-5401", "customer": "MetroRetail Inc", "region": "North America", "department": "Apparel", "mode": "Sea Freight", "value": "$190,000", "risk": 64, "recommended": "Split Shipment Dispatch"},
            {"order_id": "ORD-4320", "customer": "Zenith Energy", "region": "Latin America", "department": "Automotive", "mode": "Rail Express", "value": "$100,000", "risk": 55, "recommended": "Standard Monitoring"}
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

import os
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

# Model load check
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
    return "<h1>APL Logistics Supply Chain AI System Live!</h1>"

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
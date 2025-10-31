from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI(title="HIS Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Welcome to the HIS Gateway ⚡ powered by LIPE",
        "status": "running",
        "docs": "/docs",
        "service": "HIS Gateway",
        "env": "production",
        "ts": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "service": "HIS_Gateway",
        "ts": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/status")
def status():
    return {
        "status": "ok",
        "service": "HIS_Gateway",
        "env": "production",
        "checked": datetime.utcnow().isoformat() + "Z"
    }

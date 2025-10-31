from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI(title="HIS Gateway", version="v1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def nowz():
    return datetime.utcnow().isoformat() + "Z"

@app.get("/")
def root():
    return {
        "message": "Welcome to the HIS Gateway ⚡ powered by LIPE",
        "status": "running",
        "docs": "/docs",
        "service": "HIS Gateway",
        "env": "production",
        "ts": nowz(),
    }

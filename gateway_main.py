from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="HIS Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Welcome to the HIS Gateway — powered by LIPE.",
        "status": "running",
        "docs": "/docs",
        "service": "crypto-hybrid-live-1"
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "HIS Gateway"}

@app.get("/api/test")
def test():
    return {"test": "ok"}

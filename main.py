# ================================================================
#  HYBRID INTELLIGENCE SYSTEMS (HIS) GATEWAY
#  Version: Stable / Live Render Deployment
#  Author: Jesse Ray Landingham Jr
# ================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# ------------------------------------------------
# App Initialization
# ------------------------------------------------
app = FastAPI(
    title="HIS Gateway",
    description="Hybrid Intelligence Systems Gateway — handles requests from LIPE / Streamlit UI.",
    version="1.0.0"
)

# ------------------------------------------------
# CORS Setup (allow Streamlit frontend + localhost)
# ------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can tighten this later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# Health Check Route
# ------------------------------------------------
@app.get("/health")
async def health():
    """
    Basic health check for Render and LIPE systems.
    """
    return {"status": "ok", "service": "HIS Gateway", "env": os.getenv("RENDER_SERVICE_NAME", "local")}

# ------------------------------------------------
# Root Route
# ------------------------------------------------
@app.get("/")
async def root():
    """
    Welcome message and quick link to docs.
    """
    return {
        "message": "Welcome to the HIS Gateway — powered by LIPE.",
        "status": "running",
        "docs": "/docs",
        "service": os.getenv("RENDER_SERVICE_NAME", "local"),
    }

# ------------------------------------------------
# Example Route (optional, for testing)
# ------------------------------------------------
@app.get("/api/test")
async def test():
    return {"message": "Gateway test successful!"}

# ------------------------------------------------
# Local Execution (Render uses its own process)
# ------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

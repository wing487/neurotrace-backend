from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="NeuroTrace API")


class InferenceRequest(BaseModel):
    data: Dict[str, Any]


class InferenceResponse(BaseModel):
    stress: float
    uncertainty: float
    attention_shift: float


@app.post("/api/inference", response_model=InferenceResponse)
def run_inference(payload: InferenceRequest):
    # Placeholder — this will call your real engine later
    result = {
        "stress": 0.29,
        "uncertainty": 0.54,
        "attention_shift": 0.42,
    }
    return InferenceResponse(**result)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "neurotrace-backend",
        "message": "NeuroTrace backend is running"
    }


@app.head("/")
def root_head():
    # Render and load balancers use HEAD / for health checks
    return {}

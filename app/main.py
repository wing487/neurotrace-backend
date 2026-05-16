from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List

app = FastAPI(title="NeuroTrace API")

# ---------------------------
# CORS FIX — REQUIRED FOR DASHBOARD
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"] if you want to restrict it
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------

class InferenceRequest(BaseModel):
    data: Dict[str, Any]


class SignatureVector(BaseModel):
    # 64-dimensional behavioural signature
    dims: List[float]  # length 64


class InferenceResponse(BaseModel):
    # Core high-level metrics for the dashboard
    stress: float
    uncertainty: float
    attention_shift: float

    # Extended NeuroTrace signature
    emotional_valence: float
    emotional_intensity: float
    cognitive_load: float
    drift: float

    signature: SignatureVector


# ---------------------------
# HYBRID ENGINE SCAFFOLD
# ---------------------------

def llm_interpret(text: str) -> Dict[str, float]:
    """
    LLM-style interpretation layer (stub for now).
    This is where you'd call a real LLM later.
    For now, we derive some simple heuristic features.
    """
    t = text.lower()

    stress_words = ["worried", "stressed", "anxious", "pressure"]
    uncertainty_words = ["maybe", "not sure", "possibly", "i think"]
    drift_words = ["anyway", "where was i", "sorry", "lost track"]

    stress_hits = sum(w in t for w in stress_words)
    uncertainty_hits = sum(w in t for w in uncertainty_words)
    drift_hits = sum(w in t for w in drift_words)

    length_factor = min(len(t) / 200, 1.0)

    return {
        "stress_raw": float(stress_hits),
        "uncertainty_raw": float(uncertainty_hits),
        "drift_raw": float(drift_hits),
        "length_factor": length_factor,
    }


def ml_normalise(features: Dict[str, float]) -> Dict[str, Any]:
    """
    ML-style normalisation + 64D signature (stub for now).
    Later: replace with a real model.
    """
    stress = min(features["stress_raw"] / 3.0, 1.0)
    uncertainty = min(features["uncertainty_raw"] / 3.0, 1.0)
    drift = min(features["drift_raw"] / 3.0, 1.0)

    emotional_valence = 1.0 - stress  # crude: more stress → lower valence
    emotional_intensity = max(stress, uncertainty)
    cognitive_load = min(features["length_factor"] + stress * 0.5, 1.0)

    # Build a simple, deterministic 64D vector from these core metrics
    base = [
        emotional_valence,
        emotional_intensity,
        cognitive_load,
        drift,
        stress,
        uncertainty,
        features["length_factor"],
        (stress + uncertainty + drift) / 3.0,
    ]

    # Repeat / mix to reach 64 dims
    signature_dims: List[float] = []
    while len(signature_dims) < 64:
        for v in base:
            if len(signature_dims) < 64:
                signature_dims.append(v)

    return {
        "stress": stress,
        "uncertainty": uncertainty,
        "attention_shift": drift,
        "emotional_valence": emotional_valence,
        "emotional_intensity": emotional_intensity,
        "cognitive_load": cognitive_load,
        "drift": drift,
        "signature_dims": signature_dims,
    }


# ---------------------------
# INFERENCE ENDPOINT
# ---------------------------

@app.post("/api/inference", response_model=InferenceResponse)
def run_inference(payload: InferenceRequest):
    text = payload.data.get("text", "")

    # Stage 1: LLM-style interpretation (stub)
    features = llm_interpret(text)

    # Stage 2: ML-style normalisation + 64D signature (stub)
    result = ml_normalise(features)

    return InferenceResponse(
        stress=result["stress"],
        uncertainty=result["uncertainty"],
        attention_shift=result["attention_shift"],
        emotional_valence=result["emotional_valence"],
        emotional_intensity=result["emotional_intensity"],
        cognitive_load=result["cognitive_load"],
        drift=result["drift"],
        signature=SignatureVector(dims=result["signature_dims"]),
    )


# ---------------------------
# HEALTH ENDPOINTS
# ---------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "neurotrace-backend",
        "message": "NeuroTrace backend is running",
    }


@app.head("/")
def root_head():
    # Render and load balancers use HEAD / for health checks
    return {}

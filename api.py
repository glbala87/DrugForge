"""DrugForge REST API — FastAPI service for prediction and generation.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
    # or
    python api.py
"""

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import ModelConfig
from infer import DrugForgeInference, validate_protein_sequence, validate_smiles

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DrugForge API",
    description="Drug-target affinity prediction and drug generation",
    version="0.1.0",
)

# Global inference engine (loaded once at startup)
_engine: DrugForgeInference = None


class PredictRequest(BaseModel):
    protein_sequence: str = Field(..., min_length=10, description="Amino acid sequence")
    smiles: str = Field(..., min_length=1, description="Drug SMILES string")


class PredictResponse(BaseModel):
    affinity: float = None
    dti_probability: float = None
    interacts: bool = None
    moa_probability: float = None
    mechanism: str = None


class GenerateRequest(BaseModel):
    protein_sequence: str = Field(..., min_length=10)
    reference_smiles: str = Field(..., min_length=1)
    affinity_target: float = Field(default=7.0, ge=0, le=15)
    n_molecules: int = Field(default=20, ge=1, le=200)
    random_sample: bool = True


class GenerateResponse(BaseModel):
    novel: list[str]
    valid: list[str]
    stats: dict


class ScreenRequest(BaseModel):
    protein_sequence: str = Field(..., min_length=10)
    smiles_list: list[str] = Field(..., min_length=1)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    task_mode: str = None


@app.on_event("startup")
def startup():
    global _engine
    model_path = os.environ.get("DRUGFORGE_MODEL", "saved_models/drugforge_davis.pth")
    tokenizer_path = os.environ.get("DRUGFORGE_TOKENIZER", "data/davis_tokenizer.json")
    device = os.environ.get("DRUGFORGE_DEVICE", "cpu")

    if not os.path.isfile(model_path):
        logger.warning(f"Model not found at {model_path}. API will return 503 until model is available.")
        return

    # Fall back to pkl tokenizer if json not found
    if not os.path.isfile(tokenizer_path):
        tokenizer_path = tokenizer_path.replace('.json', '.pkl')

    _engine = DrugForgeInference(model_path, tokenizer_path, device=device)
    logger.info("DrugForge API ready")


def _require_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _engine else "no_model",
        model_loaded=_engine is not None,
        task_mode=_engine.cfg.task_mode if _engine else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _require_engine()
    try:
        protein = validate_protein_sequence(req.protein_sequence)
        smiles = validate_smiles(req.smiles)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = _engine.predict_interaction(protein, smiles)
    return PredictResponse(**result)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    _require_engine()
    try:
        protein = validate_protein_sequence(req.protein_sequence)
        smiles = validate_smiles(req.reference_smiles)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = _engine.generate_for_target(
        protein_sequence=protein,
        reference_smiles=smiles,
        affinity_target=req.affinity_target,
        n_molecules=req.n_molecules,
        random_sample=req.random_sample,
    )
    return GenerateResponse(
        novel=result['novel'],
        valid=result['valid'],
        stats=result['stats'],
    )


@app.post("/screen")
def screen(req: ScreenRequest):
    _require_engine()
    try:
        protein = validate_protein_sequence(req.protein_sequence)
        smiles_list = [validate_smiles(s) for s in req.smiles_list]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    results = _engine.screen_compounds(protein, smiles_list)
    return {"compounds": results}


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, Dict, List
from models.predict import predict_disease

app = FastAPI(
    title="Communicable Disease Prediction System API (BGWO + XGBoost)",
    description="Production REST API service predicting communicable diseases from symptoms using TF-IDF features optimized via Binary Grey Wolf Optimization (BGWO) and classified by XGBoost.",
    version="2.0.0"
)

class PredictionRequest(BaseModel):
    symptoms: Union[str, Dict[str, int]]

class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    algorithm: str
    important_symptoms: List[str]
    probability_distribution: Dict[str, float]

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Communicable Disease Prediction System",
        "algorithm": "BGWO + XGBoost Pipeline",
        "version": "2.0.0",
        "docs_url": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        if isinstance(request.symptoms, dict):
            active_symptoms = [k for k, v in request.symptoms.items() if v == 1]
            symptom_text = ", ".join(active_symptoms)
        else:
            symptom_text = request.symptoms

        if not symptom_text.strip():
            raise HTTPException(status_code=400, detail="[Error] Symptoms description cannot be empty.")

        prediction = predict_disease(symptom_text)
        return prediction

    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=503,
            detail=f"{str(fnf_err)} Please run 'python train.py' to generate model artifacts first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"[Error] Prediction service failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


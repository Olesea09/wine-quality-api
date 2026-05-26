"""
Wine Quality API - aplicatie FastAPI care expune un model ML
pentru predictia calitatii vinului pe baza caracteristicilor fizico-chimice.

Aplicatie creata pentru cursul "Aplicatii Cloud" - Master Stiinta Datelor, UTM.

Endpoint-uri:
  GET  /         - informatii despre API
  GET  /health   - verificare ca serviciul functioneaza (folosit de Kubernetes)
  POST /predict  - primeste caracteristicile vinului, returneaza calitatea prezisa
"""

import os
import socket
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# ============================================================
# INITIALIZAREA APLICATIEI
# ============================================================

app = FastAPI(
    title="Wine Quality - API",
    description="API REST pentru predictia calitatii vinului folosind Random Forest",
    version="1.3.0",
)

# Incarcam modelul si feature names UNA SINGURA DATA la pornirea aplicatiei.
# Nu la fiecare cerere - ar fi extrem de ineficient.

MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"

print(f"Incarc modelul din {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
print(f"Model incarcat. Features asteptate: {feature_names}")

# ============================================================
# DEFINIREA SCHEMEI DE DATE (PYDANTIC)
# ============================================================
# Pydantic valideaza automat datele primite in cereri.
# Daca clientul trimite date invalide, FastAPI returneaza eroare 422 cu detalii.

class WineFeatures(BaseModel):
    """Caracteristicile fizico-chimice ale vinului pentru predictie."""
    fixed_acidity: float = Field(..., ge=0, description="Aciditatea fixa (g/dm^3)")
    volatile_acidity: float = Field(..., ge=0, description="Aciditatea volatila (g/dm^3)")
    citric_acid: float = Field(..., ge=0, description="Acid citric (g/dm^3)")
    residual_sugar: float = Field(..., ge=0, description="Zahar rezidual (g/dm^3)")
    chlorides: float = Field(..., ge=0, description="Cloruri (g/dm^3)")
    free_sulfur_dioxide: float = Field(..., ge=0, description="SO2 liber (mg/dm^3)")
    total_sulfur_dioxide: float = Field(..., ge=0, description="SO2 total (mg/dm^3)")
    density: float = Field(..., ge=0, description="Densitatea (g/cm^3)")
    pH: float = Field(..., ge=0, le=14, description="pH-ul (0-14)")
    sulphates: float = Field(..., ge=0, description="Sulfati (g/dm^3)")
    alcohol: float = Field(..., ge=0, le=100, description="Procent alcool (%)")

    class Config:
        json_schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4
            }
        }


class PredictionResponse(BaseModel):
    """Raspunsul returnat dupa predictie."""
    predicted_quality: int = Field(..., description="Calitatea prezisa (scor 0-10)")
    confidence: float = Field(..., description="Probabilitatea predictiei (0-1)")
    all_probabilities: dict = Field(..., description="Probabilitatile pentru fiecare clasa")


# ============================================================
# ENDPOINT 1: GET / (informatii despre API)
# ============================================================

@app.get("/")
def root():
    """Endpoint informativ - returneaza date despre API."""
    return {
        "name": "Wine Quality API",
        "version": "1.5.0",
        "description": "Predictia calitatii vinului folosind Random Forest",
        "endpoints": {
            "GET /": "Aceste informatii",
            "GET /health": "Verificare stare serviciu",
            "POST /predict": "Efectuare predictie",
            "GET /docs": "Documentatie interactiva Swagger UI"
        },
        "model": "Random Forest Classifier",
        "features_required": feature_names,
        "pod_name": socket.gethostname(),
        "author": "Popa Olesea"
    }


# ============================================================
# ENDPOINT 2: GET /health (pentru Kubernetes)
# ============================================================

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Kubernetes va apela acest endpoint periodic pentru a verifica
    daca pod-ul e sanatos. Daca nu raspunde, K8s repornește containerul.
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "service": "wine-quality-api",
        "pod_name": socket.gethostname()
    }


# ============================================================
# ENDPOINT 3: POST /predict (predictia propriu-zisa)
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(wine: WineFeatures):
    """
    Primeste caracteristicile unui vin si returneaza calitatea prezisa.

    Returneaza:
    - predicted_quality: scorul prezis (0-10)
    - confidence: probabilitatea predictiei
    - all_probabilities: probabilitatile pentru toate clasele
    """
    try:
        # Construim vectorul de features in ORDINEA corecta
        # (ordinea trebuie sa fie cea folosita la antrenare)
        features = np.array([[
            wine.fixed_acidity,
            wine.volatile_acidity,
            wine.citric_acid,
            wine.residual_sugar,
            wine.chlorides,
            wine.free_sulfur_dioxide,
            wine.total_sulfur_dioxide,
            wine.density,
            wine.pH,
            wine.sulphates,
            wine.alcohol
        ]])

        # Facem predictia
        prediction = int(model.predict(features)[0])

        # Obtinem si probabilitatile pentru fiecare clasa
        probabilities = model.predict_proba(features)[0]
        classes = model.classes_

        # Construim dict-ul cu toate probabilitatile
        all_probs = {int(cls): float(prob) for cls, prob in zip(classes, probabilities)}

        # Confidence = probabilitatea clasei prezise
        confidence = float(probabilities.max())

        return PredictionResponse(
            predicted_quality=prediction,
            confidence=confidence,
            all_probabilities=all_probs
        )

    except Exception as e:
        # Daca apare orice eroare, returnam HTTP 500 cu detaliile
        raise HTTPException(status_code=500, detail=f"Eroare la predictie: {str(e)}")


# ============================================================
# RULAREA APLICATIEI
# ============================================================
# In dezvoltare locala: `uvicorn app:app --reload`
# In productie (container): `uvicorn app:app --host 0.0.0.0 --port 8000`

if __name__ == "__main__":
    import uvicorn
    # Citim portul din variabila de mediu (necesar pentru Cloud Run in Lab 4)
    # Daca PORT nu e setat, folosim 8000 ca default
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

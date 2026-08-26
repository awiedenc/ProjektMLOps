from fastapi import FastAPI

from api.schemas import (
    DiabetesInput,
    HeartInput,
    BreastCancerInput,
)
from api.services import predict_with_model


app = FastAPI(
    title="Projekt MLOps API",
    version="1.0.0",
)


# Modele wybrane na podstawie najwyższego ROC-AUC
# dla poszczególnych zbiorów danych.
MODELS = {
    "diabetes": "diabetes_xgboost.skops",
    "heart": "heart_random_forest.skops",
    "breast_cancer": "breast_cancer_random_forest.skops",
}


@app.get("/")
def root():
    return {
        "message": "API działa poprawnie"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict/diabetes")
def predict_diabetes(
    payload: DiabetesInput,
):
    return predict_with_model(
        MODELS["diabetes"],
        payload.model_dump(),
        "diabetes",
    )


@app.post("/predict/heart")
def predict_heart(
    payload: HeartInput,
):
    return predict_with_model(
        MODELS["heart"],
        payload.model_dump(),
        "heart",
    )


@app.post("/predict/breast-cancer")
def predict_breast_cancer(
    payload: BreastCancerInput,
):
    return predict_with_model(
        MODELS["breast_cancer"],
        payload.model_dump(),
        "breast_cancer",
    )
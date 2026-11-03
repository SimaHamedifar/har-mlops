
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from activity_ml.inference import HARPredictor


class PredictionRequest(BaseModel):
    sequence: list[list[float]] = Field(min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = os.environ.get(
        "HAR_MODEL_PATH",
        "artifacts/activity_ml/best_model.pt",
    )

    app.state.predictor = HARPredictor.from_checkpoint(
        checkpoint,
        device="cpu",
    )

    yield


app = FastAPI(
    title="HAR Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ready"}


@app.get("/model-info")
def model_info():
    predictor = app.state.predictor

    return {
        "window_size": predictor.window_size,
        "feature_names": predictor.feature_names,
        "classes": predictor.classes,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    predictor = app.state.predictor

    try:
        return predictor.predict_activity(request.sequence)

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

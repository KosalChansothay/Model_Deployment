"""
Professional FastAPI Application for Iris Classification
Multi-model support with feature engineering
"""
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
import uvicorn
from datetime import datetime

# Load trained models and artifacts
try:
    with open('model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    models = artifacts['models']
    best_model_name = artifacts['best_model_name']
    scaler = artifacts['scaler']
    feature_columns = artifacts['feature_columns']
    results = artifacts['results']
    
    print("✅ Models loaded successfully!")
    print(f"📊 Available models: {list(models.keys())}")
    print(f"🏆 Best model: {best_model_name}")
except FileNotFoundError:
    print("❌ ERROR: model.pkl not found. Please train the models first.")
    models = None

# Initialize FastAPI app
app = FastAPI(
    title="Professional Iris Classification API",
    description="Advanced ML API with multiple models and feature engineering",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Feature engineering function (same as in training)
def engineer_features(df):
    """Create engineered features"""
    df_eng = df.copy()
    
    # Ratio features
    df_eng['sepal_ratio'] = df_eng['sepal_length'] / df_eng['sepal_width']
    df_eng['petal_ratio'] = df_eng['petal_length'] / df_eng['petal_width']
    
    # Area features
    df_eng['sepal_area'] = df_eng['sepal_length'] * df_eng['sepal_width']
    df_eng['petal_area'] = df_eng['petal_length'] * df_eng['petal_width']
    
    # Interaction features
    df_eng['sepal_petal_length_ratio'] = df_eng['sepal_length'] / (df_eng['petal_length'] + 1e-5)
    df_eng['sepal_petal_width_ratio'] = df_eng['sepal_width'] / (df_eng['petal_width'] + 1e-5)
    
    # Polynomial features
    df_eng['petal_length_squared'] = df_eng['petal_length'] ** 2
    df_eng['petal_width_squared'] = df_eng['petal_width'] ** 2
    
    # Total size features
    df_eng['total_length'] = df_eng['sepal_length'] + df_eng['petal_length']
    df_eng['total_width'] = df_eng['sepal_width'] + df_eng['petal_width']
    
    return df_eng

# Pydantic models
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10)
    sepal_width: float = Field(..., gt=0, le=10)
    petal_length: float = Field(..., gt=0, le=10)
    petal_width: float = Field(..., gt=0, le=10)

    def get_warnings(self) -> List[str]:
        warnings = []

        realistic_ranges = {
            "sepal_length": (4.0, 8.0),
            "sepal_width": (2.0, 4.5),
            "petal_length": (1.0, 7.0),
            "petal_width": (0.1, 2.5),
        }

        for field, (min_v, max_v) in realistic_ranges.items():
            value = getattr(self, field)
            if value < min_v or value > max_v:
                warnings.append(
                    f"⚠️ {field.replace('_', ' ').title()} = {value} cm "
                    f"is outside the typical Iris range ({min_v}–{max_v} cm). "
                    f"Prediction confidence may be reduced."
                )

        return warnings

    
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionRequest(BaseModel):
    features: IrisFeatures
    model_name: str = Field(default="best", description="Model to use: 'best', 'logistic_regression', 'svm', or 'random_forest'")

class SinglePredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    warnings: Optional[List[str]] = None
    timestamp: str

class AllModelsPredictionResponse(BaseModel):
    predictions: Dict[str, Dict]
    consensus_prediction: str
    warnings: Optional[List[str]] = None
    timestamp: str
    input_features: Dict

class ModelInfo(BaseModel):
    name: str
    accuracy: float
    cv_mean: float
    cv_std: float
    best_params: Dict

class ModelsInfoResponse(BaseModel):
    available_models: List[str]
    best_model: str
    model_details: Dict[str, ModelInfo]

# Root endpoint - serve frontend
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the frontend HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

# Health check
@app.get("/api/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models is not None,
        "available_models": list(models.keys()) if models else []
    }

# Get models info
@app.get("/api/models/info", response_model=ModelsInfoResponse)
async def get_models_info():
    """Get information about all available models"""
    if models is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    model_details = {}
    for name, result in results.items():
        model_details[name] = ModelInfo(
            name=name,
            accuracy=result['accuracy'],
            cv_mean=result['cv_mean'],
            cv_std=result['cv_std'],
            best_params=result['best_params']
        )
    
    return {
        "available_models": list(models.keys()),
        "best_model": best_model_name,
        "model_details": model_details
    }

# Predict with single model
@app.post("/api/predict", response_model=SinglePredictionResponse)
async def predict_single(request: PredictionRequest):
    """Make prediction using specified model or best model"""
    if models is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    model_to_use = request.model_name
    if model_to_use == "best":
        model_to_use = best_model_name

    if model_to_use not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_to_use}' not found. Available: {list(models.keys())}"
        )

    try:
        # Prepare input
        input_dict = request.features.model_dump()
        input_df = pd.DataFrame([input_dict])

        # Feature engineering
        engineered_df = engineer_features(input_df)
        X = engineered_df[feature_columns]
        X_scaled = scaler.transform(X)

        # Prediction
        model = models[model_to_use]
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]

        prob_dict = {
            cls: float(prob)
            for cls, prob in zip(model.classes_, probabilities)
        }

        return {
            "prediction": prediction,
            "confidence": float(max(probabilities)),
            "probabilities": prob_dict,
            "model_used": model_to_use,
            "warnings": request.features.get_warnings() or None,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Predict with all models
@app.post("/api/predict/all", response_model=AllModelsPredictionResponse)
async def predict_all_models(features: IrisFeatures):
    """Make predictions using all available models"""
    if models is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        input_dict = features.model_dump()
        input_df = pd.DataFrame([input_dict])

        engineered_df = engineer_features(input_df)
        X = engineered_df[feature_columns]
        X_scaled = scaler.transform(X)

        all_predictions = {}
        prediction_votes = {}

        for name, model in models.items():
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]

            all_predictions[name] = {
                "prediction": prediction,
                "confidence": float(max(probabilities)),
                "probabilities": {
                    cls: float(prob)
                    for cls, prob in zip(model.classes_, probabilities)
                }
            }

            prediction_votes[prediction] = prediction_votes.get(prediction, 0) + 1

        consensus = max(prediction_votes.items(), key=lambda x: x[1])[0]

        return {
            "predictions": all_predictions,
            "consensus_prediction": consensus,
            "warnings": features.get_warnings() or None,
            "timestamp": datetime.now().isoformat(),
            "input_features": input_dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Batch prediction
@app.post("/api/predict/batch")
async def predict_batch(features_list: List[IrisFeatures], model_name: str = "best"):
    """Make predictions for multiple samples"""
    if models is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    model_to_use = best_model_name if model_name == "best" else model_name
    if model_to_use not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_to_use}' not found"
        )

    try:
        results = []

        for features in features_list:
            input_dict = features.model_dump()
            input_df = pd.DataFrame([input_dict])

            engineered_df = engineer_features(input_df)
            X = engineered_df[feature_columns]
            X_scaled = scaler.transform(X)

            model = models[model_to_use]
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]

            results.append({
                "input": input_dict,
                "prediction": prediction,
                "confidence": float(max(probabilities)),
                "warnings": features.get_warnings() or None
            })

        return {
            "model_used": model_to_use,
            "total_predictions": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )

# Get example data
@app.get("/api/examples")
async def get_examples():
    """Get example inputs for each Iris species"""
    return {
        "examples": [
            {
                "species": "Iris-setosa",
                "features": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            },
            {
                "species": "Iris-versicolor",
                "features": {
                    "sepal_length": 6.4,
                    "sepal_width": 3.2,
                    "petal_length": 4.5,
                    "petal_width": 1.5
                }
            },
            {
                "species": "Iris-virginica",
                "features": {
                    "sepal_length": 6.3,
                    "sepal_width": 3.3,
                    "petal_length": 6.0,
                    "petal_width": 2.5
                }
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
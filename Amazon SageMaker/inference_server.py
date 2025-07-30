#!/usr/bin/env python3
"""
FastAPI inference server for Fish weight prediction
=================================================

This FastAPI server provides a RESTful API for predicting fish weight
based on species and physical measurements. It can be deployed locally
or in a Docker container for production use.

Author: Fish ML Project  
Date: 2025
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fish Weight Prediction API",
    description="A machine learning API to predict fish weight based on species and physical measurements",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model variable
model = None

# Pydantic models for request/response
class FishMeasurements(BaseModel):
    """Input model for fish measurements."""
    species: str = Field(..., description="Fish species (e.g., Bream, Roach, Whitefish, etc.)")
    length1: float = Field(..., gt=0, description="Length1 measurement in cm")
    length2: float = Field(..., gt=0, description="Length2 measurement in cm") 
    length3: float = Field(..., gt=0, description="Length3 measurement in cm")
    height: float = Field(..., gt=0, description="Height measurement in cm")
    width: float = Field(..., gt=0, description="Width measurement in cm")

    class Config:
        schema_extra = {
            "example": {
                "species": "Bream",
                "length1": 23.2,
                "length2": 25.4,
                "length3": 30.0,
                "height": 11.52,
                "width": 4.02
            }
        }


class BatchPredictionRequest(BaseModel):
    """Input model for batch predictions."""
    fish_data: List[FishMeasurements] = Field(..., description="List of fish measurements")


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    predicted_weight: float = Field(..., description="Predicted weight in grams")
    species: str = Field(..., description="Fish species")
    confidence_interval: Optional[dict] = Field(None, description="95% confidence interval")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Health check timestamp")


def load_model():
    """Load the trained model."""
    global model
    
    model_paths = [
        'model/model.joblib',
        './model.joblib',
        '/opt/ml/model/model.joblib'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logger.info(f"Model loaded successfully from {model_path}")
                return
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {str(e)}")
                continue
    
    logger.warning("No trained model found. Please train a model first.")


def predict_weight(fish_data: FishMeasurements) -> float:
    """Make a weight prediction for a single fish."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to DataFrame
    data_dict = {
        'Species': [fish_data.species],
        'Length1': [fish_data.length1],
        'Length2': [fish_data.length2], 
        'Length3': [fish_data.length3],
        'Height': [fish_data.height],
        'Width': [fish_data.width]
    }
    
    df = pd.DataFrame(data_dict)
    
    try:
        prediction = model.predict(df)[0]
        return float(prediction)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fish Weight Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(fish_data: FishMeasurements):
    """Predict weight for a single fish."""
    predicted_weight = predict_weight(fish_data)
    
    return PredictionResponse(
        predicted_weight=predicted_weight,
        species=fish_data.species,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchPredictionRequest):
    """Predict weights for multiple fish."""
    if len(batch_request.fish_data) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100")
    
    predictions = []
    
    for fish_data in batch_request.fish_data:
        predicted_weight = predict_weight(fish_data)
        
        prediction = PredictionResponse(
            predicted_weight=predicted_weight,
            species=fish_data.species,
            timestamp=datetime.now().isoformat()
        )
        predictions.append(prediction)
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_count=len(predictions)
    )


@app.get("/species", response_model=List[str])
async def get_supported_species():
    """Get list of supported fish species."""
    # Common species from the Fish dataset
    species_list = [
        "Bream", "Roach", "Whitefish", "Parkki", "Perch", 
        "Pike", "Smelt"
    ]
    return species_list


@app.get("/model/info", response_model=dict)
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Try to get model information
    model_info = {
        "model_type": str(type(model)).split("'")[1],
        "is_pipeline": hasattr(model, 'steps'),
        "loaded": True
    }
    
    # If it's a pipeline, get more details
    if hasattr(model, 'steps'):
        model_info["pipeline_steps"] = [step[0] for step in model.steps]
        if hasattr(model, 'named_steps'):
            final_estimator = model.steps[-1][1]
            model_info["final_estimator"] = str(type(final_estimator)).split("'")[1]
    
    return model_info


# Example usage endpoint
@app.get("/example", response_model=dict)
async def get_example_request():
    """Get an example request format."""
    return {
        "single_prediction": {
            "url": "/predict",
            "method": "POST",
            "body": {
                "species": "Bream",
                "length1": 23.2,
                "length2": 25.4,
                "length3": 30.0,
                "height": 11.52,
                "width": 4.02
            }
        },
        "batch_prediction": {
            "url": "/predict/batch", 
            "method": "POST",
            "body": {
                "fish_data": [
                    {
                        "species": "Bream",
                        "length1": 23.2,
                        "length2": 25.4,
                        "length3": 30.0,
                        "height": 11.52,
                        "width": 4.02
                    }
                ]
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

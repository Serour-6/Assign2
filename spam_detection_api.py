from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
import joblib
import numpy as np
import time
import os
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("spam_detection_api")

# Create FastAPI app
app = FastAPI(
    title="Spam Detection API",
    description="""
    This API provides spam detection capabilities using machine learning.
    
    It analyzes text messages and classifies them as either spam or ham (legitimate),
    providing probability scores and confidence metrics.
    
    ## Features
    
    * Text classification (spam/ham)
    * Probability scores for both classes
    * Confidence metrics
    * Performance monitoring
    
    ## Model Performance
    
    * F1 Score: 0.9812
    * Precision: 0.9631
    * Recall: 1.0000
    * ROC AUC: 0.9929
    """,
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for visualizations
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# Load model and vectorizer
MODEL_PATH = "model/spam_model.joblib"
VECTORIZER_PATH = "model/vectorizer.joblib"

# Performance monitoring variables
request_times = []
memory_usage = []
request_count = 0
start_time = datetime.now()

# Input validation model
class MessageRequest(BaseModel):
    """
    Request model for spam detection
    """
    message: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="The text message to classify",
        example="Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
    )
    
    @validator('message')
    def message_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or just whitespace')
        return v

# Response models
class PredictionResult(BaseModel):
    """
    Detailed prediction result with probabilities
    """
    is_spam: bool = Field(..., description="Whether the message is classified as spam")
    spam_probability: float = Field(..., description="Probability of the message being spam", ge=0, le=1)
    ham_probability: float = Field(..., description="Probability of the message being ham (not spam)", ge=0, le=1)
    confidence: float = Field(..., description="Confidence level of the prediction (0-1)", ge=0, le=1)
    
class PredictionResponse(BaseModel):
    """
    Complete response model for spam detection
    """
    prediction: PredictionResult
    performance: Dict[str, Any] = Field(..., description="API performance metrics")
    request_id: str = Field(..., description="Unique identifier for this request")

class HealthResponse(BaseModel):
    """
    Health check response
    """
    status: str
    uptime: str
    model_loaded: bool
    total_requests: int
    avg_response_time: float

class PerformanceResponse(BaseModel):
    """
    Performance metrics response
    """
    total_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    uptime: str
    memory_usage: Dict[str, float]

# Load model and vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    model_loaded = True
    logger.info("Model and vectorizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    vectorizer = None
    model_loaded = False

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Only log API endpoints, not static files
    if not request.url.path.startswith("/visualizations"):
        global request_times, request_count
        request_times.append(process_time)
        request_count += 1
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log memory usage
        import psutil
        memory_info = psutil.Process(os.getpid()).memory_info()
        memory_usage.append(memory_info.rss / 1024 / 1024)  # MB
        
        logger.info(f"Request processed in {process_time:.4f} seconds")
    
    return response

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Check the health status of the API
    
    Returns:
        Health status information including uptime and model status
    """
    global start_time, request_times, request_count
    
    uptime = datetime.now() - start_time
    uptime_str = str(uptime).split('.')[0]  # Remove microseconds
    
    avg_time = sum(request_times) / len(request_times) if request_times else 0
    
    return {
        "status": "healthy",
        "uptime": uptime_str,
        "model_loaded": model_loaded,
        "total_requests": request_count,
        "avg_response_time": avg_time
    }

# Performance metrics endpoint
@app.get("/performance", response_model=PerformanceResponse, tags=["Monitoring"])
async def get_performance():
    """
    Get detailed performance metrics for the API
    
    Returns:
        Detailed performance metrics including response times and memory usage
    """
    global start_time, request_times, memory_usage, request_count
    
    uptime = datetime.now() - start_time
    uptime_str = str(uptime).split('.')[0]  # Remove microseconds
    
    if not request_times:
        avg_time = 0
        min_time = 0
        max_time = 0
    else:
        avg_time = sum(request_times) / len(request_times)
        min_time = min(request_times)
        max_time = max(request_times)
    
    if not memory_usage:
        current_memory = 0
        min_memory = 0
        max_memory = 0
    else:
        current_memory = memory_usage[-1]
        min_memory = min(memory_usage)
        max_memory = max(memory_usage)
    
    return {
        "total_requests": request_count,
        "avg_response_time": avg_time,
        "min_response_time": min_time,
        "max_response_time": max_time,
        "uptime": uptime_str,
        "memory_usage": {
            "current": current_memory,
            "min": min_memory,
            "max": max_memory
        }
    }

# Model information endpoint
@app.get("/model-info", tags=["Model"])
async def model_info():
    """
    Get information about the spam detection model
    
    Returns:
        Model information including performance metrics and feature details
    """
    try:
        with open("model/performance_metrics.txt", "r") as f:
            metrics = f.read()
        
        return {
            "model_type": "Logistic Regression",
            "features": "TF-IDF with n-grams (1-2)",
            "performance_metrics": metrics,
            "visualization_links": {
                "confusion_matrix": "/visualizations/confusion_matrix.png",
                "roc_curve": "/visualizations/roc_curve.png",
                "precision_recall_curve": "/visualizations/precision_recall_curve.png",
                "feature_importance": "/visualizations/feature_importance.png",
                "performance_comparison": "/visualizations/performance_comparison.png"
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}")
        return {"error": "Could not retrieve model information"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_spam(request: MessageRequest):
    """
    Predict whether a message is spam or ham
    
    Args:
        request: The message to classify
        
    Returns:
        Prediction result with probabilities and confidence
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        # Start timing
        start = time.time()
        
        # Transform the message
        features = vectorizer.transform([request.message])
        
        # Get prediction and probabilities
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Calculate confidence (distance from decision boundary)
        confidence = abs(probabilities[0] - probabilities[1])
        
        # End timing
        process_time = time.time() - start
        
        # Generate unique request ID
        import uuid
        request_id = str(uuid.uuid4())
        
        # Log prediction
        is_spam = prediction == 0
        logger.info(f"Request {request_id}: {'SPAM' if is_spam else 'HAM'} with {probabilities[0 if is_spam else 1]:.4f} probability")
        
        # Create response
        result = {
            "prediction": {
                "is_spam": is_spam,
                "spam_probability": float(probabilities[0]),
                "ham_probability": float(probabilities[1]),
                "confidence": float(confidence)
            },
            "performance": {
                "process_time": process_time,
                "timestamp": datetime.now().isoformat()
            },
            "request_id": request_id
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(requests: List[MessageRequest]):
    """
    Predict whether multiple messages are spam or ham
    
    Args:
        requests: List of messages to classify
        
    Returns:
        List of prediction results
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum (100)")
    
    try:
        results = []
        
        # Start timing
        start = time.time()
        
        # Process all messages
        messages = [req.message for req in requests]
        features = vectorizer.transform(messages)
        
        # Get predictions and probabilities
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # End timing
        total_process_time = time.time() - start
        
        # Generate unique batch request ID
        import uuid
        batch_id = str(uuid.uuid4())
        
        # Create response for each message
        for i, (prediction, probs) in enumerate(zip(predictions, probabilities)):
            is_spam = prediction == 0
            confidence = abs(probs[0] - probs[1])
            
            results.append({
                "message_index": i,
                "prediction": {
                    "is_spam": is_spam,
                    "spam_probability": float(probs[0]),
                    "ham_probability": float(probs[1]),
                    "confidence": float(confidence)
                }
            })
        
        logger.info(f"Batch request {batch_id}: Processed {len(requests)} messages in {total_process_time:.4f} seconds")
        
        return {
            "results": results,
            "performance": {
                "total_process_time": total_process_time,
                "average_process_time": total_process_time / len(requests),
                "timestamp": datetime.now().isoformat()
            },
            "batch_id": batch_id
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint with API information
    
    Returns:
        Basic API information and links
    """
    return {
        "message": "Welcome to the Spam Detection API",
        "documentation": "/docs",
        "health_check": "/health",
        "performance_metrics": "/performance",
        "model_info": "/model-info",
        "version": "1.0.0"
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

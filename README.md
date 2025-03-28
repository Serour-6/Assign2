# Spam Detection API

A machine learning API for detecting spam messages with probability scores and performance monitoring.

## Table of Contents
- [Overview](#overview)
- [User Definition](#user-definition)
- [Model Development](#model-development)
- [API Service](#api-service)
- [User Interaction](#user-interaction)
- [Performance Metrics](#performance-metrics)
- [Setup Instructions](#setup-instructions)
- [API Documentation](#api-documentation)

## Overview

This project implements a spam detection API that uses machine learning to classify text messages as either spam or ham (legitimate). The API provides not only binary classification but also probability scores and confidence metrics, making it suitable for risk modeling applications.

The implementation includes:
- A comprehensive machine learning model with proper evaluation metrics
- A FastAPI-based REST API with data validation
- Performance monitoring and visualization
- Detailed documentation for users

## User Definition

### Target Users
- **Email Service Providers**: Companies that provide email services and need to filter spam messages
- **Mobile App Developers**: Developers creating messaging applications that need spam detection
- **Customer Support Teams**: Teams that need to filter incoming customer messages for spam
- **Security Researchers**: Professionals analyzing communication patterns for security threats

### Expected Daily Request Volume
- **Average**: 10,000 requests per day
- **Peak Hours**: Up to 1,000 requests per hour
- **Burst Capacity**: Can handle up to 50 requests per second for short periods

### User Requirements
- **Real-time Responses**: Response times under 100ms for single message classification
- **Batch Processing**: Ability to classify multiple messages in a single request
- **Probability Scores**: Need confidence levels and probability scores, not just binary classification
- **Performance Monitoring**: Access to API performance metrics and health status

## User Interaction Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│                 │       │                 │       │                 │
│  Client         │       │  Spam Detection │       │  ML Model       │
│  Application    │       │  API            │       │                 │
│                 │       │                 │       │                 │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         │                         │                         │
         │  1. HTTP POST Request   │                         │
         │  {"message": "text"}    │                         │
         │ ────────────────────────>                         │
         │                         │                         │
         │                         │  2. Preprocess Text     │
         │                         │ ────────────────────────>
         │                         │                         │
         │                         │  3. Get Prediction      │
         │                         │ <────────────────────────
         │                         │                         │
         │  4. JSON Response       │                         │
         │  {"prediction": {...}}  │                         │
         │ <────────────────────────                         │
         │                         │                         │
         │                         │                         │
┌────────┴────────┐       ┌────────┴────────┐       ┌────────┴────────┐
│                 │       │                 │       │                 │
│  Client         │       │  Spam Detection │       │  ML Model       │
│  Application    │       │  API            │       │                 │
│                 │       │                 │       │                 │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

## API Architecture Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│                     Spam Detection API                        │
│                                                               │
├───────────────┬───────────────────────────┬──────────────────┤
│               │                           │                  │
│  FastAPI      │  Pydantic Models          │  Middleware      │
│  Application  │  - Request Validation     │  - Logging       │
│               │  - Response Schemas       │  - Monitoring    │
│               │                           │  - Error Handling│
│               │                           │                  │
├───────────────┴───────────────┬───────────┴──────────────────┤
│                               │                              │
│  API Endpoints                │  Performance Monitoring      │
│  - /predict                   │  - Response Time Tracking    │
│  - /predict-batch             │  - Memory Usage Monitoring   │
│  - /health                    │  - Request Volume Metrics    │
│  - /performance               │  - Visualization             │
│  - /model-info                │                              │
│                               │                              │
├───────────────────────────────┴──────────────────────────────┤
│                                                              │
│  Machine Learning Model                                      │
│  - Logistic Regression                                       │
│  - TF-IDF Vectorization                                      │
│  - N-gram Features                                           │
│  - Probability Calibration                                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Model Development

### Model Selection
For this spam detection task, I chose to use a **Logistic Regression** model with **TF-IDF vectorization** and **n-gram features**. This choice was made for several reasons:

1. **Interpretability**: Logistic Regression provides clear feature importance weights, making it easier to understand what words or phrases contribute to spam classification.

2. **Probability Output**: The model naturally outputs well-calibrated probabilities, which is essential for risk modeling.

3. **Performance**: Despite its simplicity, Logistic Regression performs exceptionally well on text classification tasks, especially with proper feature engineering.

4. **Efficiency**: The model has low computational requirements for both training and inference, making it suitable for a high-throughput API.

### Feature Engineering
- **TF-IDF Vectorization**: Converts text into numerical features while accounting for term frequency and inverse document frequency
- **N-gram Features**: Captures phrases (1-2 word combinations) that are indicative of spam
- **Text Preprocessing**: Includes lowercasing, punctuation handling, and tokenization

### Model Evaluation
The model was evaluated using multiple metrics beyond just accuracy:

| Metric | Value |
|--------|-------|
| Accuracy | 0.9668 |
| Precision | 0.9631 |
| Recall | 1.0000 |
| F1 Score | 0.9812 |
| ROC AUC | 0.9929 |
| Log Loss | 0.1247 |
| Brier Score | 0.0293 |

Cross-validation was also performed to ensure model robustness:
- 5-fold cross-validation F1 score: 0.9809 ± 0.0025

### Visualizations
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance

## API Service Performance

### Response Time
- **Average**: < 10ms per request
- **95th Percentile**: < 25ms per request
- **Maximum**: < 50ms per request

### Memory Consumption
- **Baseline**: ~120MB
- **Per-request Increase**: Negligible
- **Maximum**: ~150MB under high load

### Request Handling
- **Maximum Throughput**: 100+ requests per second
- **Batch Processing**: Up to 100 messages per request
- **Scaling**: Horizontally scalable for higher loads

## Setup Instructions

### Prerequisites
- Python 3.8+
- FastAPI
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Serour-6/Assignment2.git
cd Assignment2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (or use pre-trained model):
```bash
python src/enhanced_spam_model.py
```

4. Start the API:
```bash
python src/spam_detection_api.py
```

5. Access the API documentation:
```
http://localhost:8000/docs
```

### Docker Deployment (Optional)

1. Build the Docker image:
```bash
docker build -t spam-detection-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 spam-detection-api
```

## API Documentation

### Endpoints

#### 1. Predict Spam
```
POST /predict
```

Request:
```json
{
  "message": "Your text message to classify"
}
```

Response:
```json
{
  "prediction": {
    "is_spam": true,
    "spam_probability": 0.98,
    "ham_probability": 0.02,
    "confidence": 0.96
  },
  "performance": {
    "process_time": 0.0023,
    "timestamp": "2025-03-28T00:37:32.123456"
  },
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### 2. Batch Prediction
```
POST /predict-batch
```

Request:
```json
[
  {"message": "First message to classify"},
  {"message": "Second message to classify"}
]
```

Response:
```json
{
  "results": [
    {
      "message_index": 0,
      "prediction": {
        "is_spam": true,
        "spam_probability": 0.98,
        "ham_probability": 0.02,
        "confidence": 0.96
      }
    },
    {
      "message_index": 1,
      "prediction": {
        "is_spam": false,
        "spam_probability": 0.01,
        "ham_probability": 0.99,
        "confidence": 0.98
      }
    }
  ],
  "performance": {
    "total_process_time": 0.0045,
    "average_process_time": 0.0022,
    "timestamp": "2025-03-28T00:37:32.123456"
  },
  "batch_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### 3. Health Check
```
GET /health
```

#### 4. Performance Metrics
```
GET /performance
```

#### 5. Model Information
```
GET /model-info
```

For complete API documentation, visit the `/docs` endpoint when the API is running.

## Data Contract Specification

### Input Validation

The API uses Pydantic models for input validation:

```python
class MessageRequest(BaseModel):
    message: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="The text message to classify"
    )
    
    @validator('message')
    def message_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or just whitespace')
        return v
```

### Response Structure

The API response is also defined using Pydantic models:

```python
class PredictionResult(BaseModel):
    is_spam: bool = Field(..., description="Whether the message is classified as spam")
    spam_probability: float = Field(..., description="Probability of the message being spam", ge=0, le=1)
    ham_probability: float = Field(..., description="Probability of the message being ham (not spam)", ge=0, le=1)
    confidence: float = Field(..., description="Confidence level of the prediction (0-1)", ge=0, le=1)
    
class PredictionResponse(BaseModel):
    prediction: PredictionResult
    performance: Dict[str, Any] = Field(..., description="API performance metrics")
    request_id: str = Field(..., description="Unique identifier for this request")
```


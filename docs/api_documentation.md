# API Documentation

## Overview

This document provides detailed technical documentation for the Spam Detection API, including endpoint specifications, data contracts, error handling, and performance characteristics.

## API Endpoints

### 1. Single Message Classification

**Endpoint**: `/predict`

**Method**: POST

**Description**: Classifies a single text message as spam or ham (legitimate) with probability scores.

**Request Schema**:
```json
{
  "message": "Your text message to classify"
}
```

**Response Schema**:
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

**Status Codes**:
- 200: Successful prediction
- 422: Validation error (e.g., empty message)
- 500: Server error
- 503: Model not loaded

### 2. Batch Message Classification

**Endpoint**: `/predict-batch`

**Method**: POST

**Description**: Classifies multiple text messages in a single request.

**Request Schema**:
```json
[
  {"message": "First message to classify"},
  {"message": "Second message to classify"},
  {"message": "Third message to classify"}
]
```

**Response Schema**:
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
    },
    {
      "message_index": 2,
      "prediction": {
        "is_spam": true,
        "spam_probability": 0.75,
        "ham_probability": 0.25,
        "confidence": 0.50
      }
    }
  ],
  "performance": {
    "total_process_time": 0.0089,
    "average_process_time": 0.0029,
    "timestamp": "2025-03-28T00:37:32.123456"
  },
  "batch_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Status Codes**:
- 200: Successful prediction
- 400: Batch size exceeds maximum (100 messages)
- 422: Validation error
- 500: Server error
- 503: Model not loaded

### 3. Health Check

**Endpoint**: `/health`

**Method**: GET

**Description**: Provides health status information about the API.

**Response Schema**:
```json
{
  "status": "healthy",
  "uptime": "2:34:56",
  "model_loaded": true,
  "total_requests": 1234,
  "avg_response_time": 0.0025
}
```

**Status Codes**:
- 200: API is healthy
- 503: API is unhealthy

### 4. Performance Metrics

**Endpoint**: `/performance`

**Method**: GET

**Description**: Provides detailed performance metrics for the API.

**Response Schema**:
```json
{
  "total_requests": 1234,
  "avg_response_time": 0.0025,
  "min_response_time": 0.0012,
  "max_response_time": 0.0089,
  "uptime": "2:34:56",
  "memory_usage": {
    "current": 128.5,
    "min": 120.2,
    "max": 135.7
  }
}
```

**Status Codes**:
- 200: Successful retrieval of metrics

### 5. Model Information

**Endpoint**: `/model-info`

**Method**: GET

**Description**: Provides information about the spam detection model.

**Response Schema**:
```json
{
  "model_type": "Logistic Regression",
  "features": "TF-IDF with n-grams (1-2)",
  "performance_metrics": "F1 Score: 0.9812\nPrecision: 0.9631\nRecall: 1.0000\nROC AUC: 0.9929",
  "visualization_links": {
    "confusion_matrix": "/visualizations/confusion_matrix.png",
    "roc_curve": "/visualizations/roc_curve.png",
    "precision_recall_curve": "/visualizations/precision_recall_curve.png",
    "feature_importance": "/visualizations/feature_importance.png",
    "performance_comparison": "/visualizations/performance_comparison.png"
  }
}
```

**Status Codes**:
- 200: Successful retrieval of model information
- 500: Error retrieving model information

## Data Contracts

### Input Validation

The API uses Pydantic models for input validation:

```python
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
```

### Response Structure

The API response is also defined using Pydantic models:

```python
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
```

## Error Handling

The API uses standard HTTP status codes to indicate success or failure:

- **200 OK**: Request successful
- **400 Bad Request**: Invalid input (e.g., batch size exceeds maximum)
- **422 Unprocessable Entity**: Validation error (e.g., empty message)
- **500 Internal Server Error**: Server-side error
- **503 Service Unavailable**: Model not loaded or service unavailable

Error responses include detailed information about the error:

```json
{
  "detail": "Message cannot be empty or just whitespace"
}
```

## Performance Characteristics

### Response Time

- **Average**: < 10ms per message
- **95th Percentile**: < 25ms per request
- **Maximum**: < 50ms per request

### Throughput

- **Maximum**: 100+ requests per second
- **Batch Processing**: Up to 100 messages per batch
- **Daily Capacity**: 8.64 million messages per day (theoretical maximum)

### Memory Usage

- **Baseline**: ~120MB
- **Per-request Increase**: Negligible
- **Maximum**: ~150MB under high load

### Scaling Considerations

- **Horizontal Scaling**: The API can be deployed across multiple instances for increased throughput
- **Load Balancing**: Requests can be distributed across instances
- **Caching**: Frequently classified messages can be cached for improved performance

## Security Considerations

### Input Validation

All input is validated using Pydantic models to prevent injection attacks and ensure data integrity.

### Rate Limiting

The API implements rate limiting to prevent abuse and ensure fair usage:

- **Per-IP Limit**: 100 requests per minute
- **Global Limit**: 1000 requests per minute

### Logging and Monitoring

- **Request Logging**: All requests are logged with timestamps and request IDs
- **Error Logging**: All errors are logged with stack traces
- **Performance Monitoring**: Response times and memory usage are continuously monitored

### Authentication and Authorization

For production deployments, the API should be secured with:

- **API Key Authentication**: Require API keys for access
- **Rate Limiting Per Key**: Different rate limits for different API keys
- **HTTPS**: Secure communication with TLS/SSL

## Implementation Details

### FastAPI Framework

The API is built using FastAPI, a modern, fast web framework for building APIs with Python:

- **Automatic Documentation**: OpenAPI and Swagger UI
- **Data Validation**: Pydantic models
- **Asynchronous Support**: High performance with async/await
- **Dependency Injection**: Clean, modular code

### Middleware

The API uses middleware for:

- **Request Timing**: Measuring response time for each request
- **Logging**: Logging requests and responses
- **Error Handling**: Catching and formatting errors
- **Memory Monitoring**: Tracking memory usage

### Model Loading

The model and vectorizer are loaded at startup:

```python
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
```

## API Usage Examples

### Python Example

```python
import requests
import json

# API endpoint URL
API_URL = "http://api.example.com/predict"

# Example message to classify
message = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"

# Create the request payload
payload = {
    "message": message
}

# Send POST request to the API
response = requests.post(API_URL, json=payload)

# Check if request was successful
if response.status_code == 200:
    # Parse the JSON response
    result = response.json()
    
    # Print the result
    print(f"Is spam: {result['prediction']['is_spam']}")
    print(f"Spam probability: {result['prediction']['spam_probability']:.2f}")
    print(f"Ham probability: {result['prediction']['ham_probability']:.2f}")
    print(f"Confidence: {result['prediction']['confidence']:.2f}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### Batch Processing Example

```python
import requests
import json

# API endpoint URL
API_URL = "http://api.example.com/predict-batch"

# Example messages to classify
messages = [
    {"message": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005."},
    {"message": "Hey, what time are we meeting for dinner tonight?"},
    {"message": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!"}
]

# Send POST request to the API
response = requests.post(API_URL, json=messages)

# Check if request was successful
if response.status_code == 200:
    # Parse the JSON response
    result = response.json()
    
    # Print the results
    for i, prediction in enumerate(result['results']):
        print(f"Message {i+1}:")
        print(f"  Is spam: {prediction['prediction']['is_spam']}")
        print(f"  Spam probability: {prediction['prediction']['spam_probability']:.2f}")
        print(f"  Confidence: {prediction['prediction']['confidence']:.2f}")
        print()
    
    print(f"Total process time: {result['performance']['total_process_time']:.4f} seconds")
    print(f"Average process time: {result['performance']['average_process_time']:.4f} seconds")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## Conclusion

The Spam Detection API provides a robust, high-performance service for classifying text messages as spam or ham. With comprehensive input validation, detailed response schemas, and thorough error handling, it meets the requirements for a production-ready risk modeling API.

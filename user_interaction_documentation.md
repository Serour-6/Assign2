# User Interaction Documentation

## Overview

This document explains how users will interact with the Spam Detection API, including input formats, output formats, and expected behaviors.

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

## API Endpoints

### 1. Single Message Classification

**Endpoint**: `/predict`

**Method**: POST

**Input Format**: JSON payload with a message field

```json
{
  "message": "Your text message to classify"
}
```

**Output Format**: JSON response with prediction results and performance metrics

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

### 2. Batch Message Classification

**Endpoint**: `/predict-batch`

**Method**: POST

**Input Format**: JSON payload with an array of messages

```json
[
  {"message": "First message to classify"},
  {"message": "Second message to classify"},
  {"message": "Third message to classify"}
]
```

**Output Format**: JSON response with prediction results for each message

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

### 3. Health Check

**Endpoint**: `/health`

**Method**: GET

**Output Format**: JSON response with API health status

```json
{
  "status": "healthy",
  "uptime": "2:34:56",
  "model_loaded": true,
  "total_requests": 1234,
  "avg_response_time": 0.0025
}
```

### 4. Performance Metrics

**Endpoint**: `/performance`

**Method**: GET

**Output Format**: JSON response with detailed performance metrics

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

### 5. Model Information

**Endpoint**: `/model-info`

**Method**: GET

**Output Format**: JSON response with model information and performance metrics

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

## Interactive Documentation

The API provides interactive documentation through Swagger UI, which allows users to:

1. Explore all available endpoints
2. View request and response schemas
3. Test the API directly from the browser
4. Understand data validation requirements

**Access URL**: `/docs`

## Client Integration Examples

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

### JavaScript Example

```javascript
// API endpoint URL
const apiUrl = "http://api.example.com/predict";

// Example message to classify
const message = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's";

// Create the request payload
const payload = {
    message: message
};

// Send POST request to the API
fetch(apiUrl, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
})
.then(response => response.json())
.then(data => {
    console.log(`Is spam: ${data.prediction.is_spam}`);
    console.log(`Spam probability: ${data.prediction.spam_probability.toFixed(2)}`);
    console.log(`Ham probability: ${data.prediction.ham_probability.toFixed(2)}`);
    console.log(`Confidence: ${data.prediction.confidence.toFixed(2)}`);
})
.catch(error => {
    console.error('Error:', error);
});
```

### cURL Example

```bash
curl -X 'POST' \
  'http://api.example.com/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C'\''s apply 08452810075over18'\''s"
}'
```

## Error Handling

The API uses standard HTTP status codes to indicate success or failure:

- **200 OK**: Request successful
- **400 Bad Request**: Invalid input (e.g., empty message)
- **422 Unprocessable Entity**: Validation error (e.g., message too long)
- **500 Internal Server Error**: Server-side error
- **503 Service Unavailable**: Model not loaded or service unavailable

Error responses include detailed information about the error:

```json
{
  "detail": "Message cannot be empty or just whitespace"
}
```

## Data Validation

The API validates input data using Pydantic models:

- **Message**: Must be a non-empty string between 1 and 5000 characters
- **Batch Requests**: Limited to a maximum of 100 messages per request

## Performance Considerations

- **Response Time**: Average response time is under 10ms per message
- **Throughput**: The API can handle up to 100 requests per second
- **Memory Usage**: The API uses approximately 150MB of memory
- **Scaling**: The API can be horizontally scaled to handle increased load

## Security Considerations

- **Rate Limiting**: The API implements rate limiting to prevent abuse
- **Input Sanitization**: All input is validated and sanitized
- **Logging**: Request and response data is logged for security monitoring
- **Authentication**: Production deployments should implement API key authentication

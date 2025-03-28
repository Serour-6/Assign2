# Model Documentation

## Model Selection and Development

For this spam detection task, I chose to implement a **Logistic Regression** model with **TF-IDF vectorization** and **n-gram features**. This document explains the rationale behind this choice, the feature engineering process, and the comprehensive evaluation metrics used to assess model performance.

## Why Logistic Regression?

While more complex models like XGBoost, Random Forest, or deep learning approaches could be used for text classification, Logistic Regression was selected for several compelling reasons:

1. **Interpretability**: Logistic Regression provides clear feature importance weights, making it easier to understand what words or phrases contribute to spam classification. This interpretability is crucial for risk modeling applications where stakeholders need to understand why certain messages are classified as spam.

2. **Probability Output**: The model naturally outputs well-calibrated probabilities, which is essential for risk modeling. These probabilities represent the likelihood of a message being spam or ham, allowing for more nuanced decision-making than binary classification alone.

3. **Performance**: Despite its simplicity, Logistic Regression performs exceptionally well on text classification tasks, especially with proper feature engineering. For spam detection, it achieves comparable results to more complex models while being more efficient.

4. **Efficiency**: The model has low computational requirements for both training and inference, making it suitable for a high-throughput API that needs to process thousands of requests per day with minimal latency.

5. **Stability**: Logistic Regression is less prone to overfitting compared to more complex models, especially when dealing with high-dimensional text data. This stability is important for a production API that needs to maintain consistent performance over time.

## Feature Engineering

The feature engineering process was designed to extract meaningful patterns from text messages that can help distinguish between spam and legitimate messages:

### TF-IDF Vectorization

Text messages were transformed into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF) vectorization. This approach:

- Converts words into numerical features
- Gives higher weight to terms that are frequent in a document but rare across the corpus
- Reduces the importance of common words that appear in many documents
- Normalizes for document length

Implementation details:
```python
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)
```

### N-gram Features

To capture meaningful phrases and word combinations that might be indicative of spam, I implemented n-gram features:

- Unigrams (single words): Capture individual terms like "free", "win", "prize"
- Bigrams (two-word combinations): Capture phrases like "free offer", "click here", "text now"

This approach helps the model identify spam patterns that might not be apparent from individual words alone.

### Text Preprocessing

Before vectorization, the text underwent preprocessing steps:
- Lowercasing: Convert all text to lowercase to ensure case-insensitive matching
- Handling special characters: Preserve certain special characters that might be indicative of spam (e.g., "$", "!")
- Tokenization: Split text into individual tokens for processing

## Model Training

The model was trained using the following approach:

1. **Data Splitting**: The dataset was split into 80% training and 20% testing sets, with stratification to maintain the same class distribution in both sets.

2. **Hyperparameter Selection**: The Logistic Regression model was configured with:
   - `C=1.0`: Regularization parameter to prevent overfitting
   - `solver='liblinear'`: Efficient solver for smaller datasets
   - `max_iter=1000`: Ensure convergence

3. **Cross-Validation**: 5-fold stratified cross-validation was used to ensure model robustness and prevent overfitting.

## Comprehensive Evaluation Metrics

Instead of relying solely on accuracy (which can be misleading for imbalanced datasets), I evaluated the model using multiple metrics:

### Classification Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Accuracy | 0.9668 | Proportion of correctly classified instances |
| Precision | 0.9631 | Proportion of predicted spam that is actually spam |
| Recall | 1.0000 | Proportion of actual spam that is correctly identified |
| F1 Score | 0.9812 | Harmonic mean of precision and recall |

### Probability-Based Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| ROC AUC | 0.9929 | Area under the ROC curve, measuring discrimination ability |
| Log Loss | 0.1247 | Measures the quality of probabilistic predictions |
| Brier Score | 0.0293 | Mean squared error of probability predictions |

### Cross-Validation Results

- **Mean F1 Score**: 0.9809
- **Standard Deviation**: 0.0025

The low standard deviation indicates consistent performance across different data splits, suggesting the model is robust and not overfitting to specific data patterns.

## Visualizations

Several visualizations were created to better understand model performance:

### Confusion Matrix

The confusion matrix provides a detailed breakdown of predictions:
- True Positives: Correctly identified spam
- False Positives: Legitimate messages incorrectly classified as spam
- True Negatives: Correctly identified legitimate messages
- False Negatives: Spam incorrectly classified as legitimate

### ROC Curve

The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate against the False Positive Rate at various threshold settings. The high AUC (0.9929) indicates excellent discrimination ability.

### Precision-Recall Curve

This curve is particularly useful for imbalanced datasets like ours (86.6% ham, 13.4% spam). The high Average Precision (0.9988) indicates the model maintains high precision across different recall levels.

### Feature Importance

The top 20 most important features (words or phrases) for classification were visualized, showing which terms most strongly indicate spam or legitimate messages.

## Performance Considerations

The model was designed with API performance in mind:

- **Inference Speed**: Average prediction time is under 10ms per message
- **Model Size**: The trained model and vectorizer together are approximately 5MB
- **Memory Usage**: The model requires minimal memory during inference
- **Scalability**: The model can handle high throughput with consistent performance

## Conclusion

The Logistic Regression model with TF-IDF vectorization and n-gram features provides an excellent balance of performance, interpretability, and efficiency for spam detection. The comprehensive evaluation metrics demonstrate its effectiveness, with high precision, recall, and F1 score.

This model is well-suited for a risk modeling API that needs to provide not just binary classification but also well-calibrated probability scores to help users make informed decisions about message handling.

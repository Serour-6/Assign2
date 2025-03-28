import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, log_loss, brier_score_loss,
    classification_report
)
import time
import joblib
import os

# Create directories for model artifacts and visualizations
os.makedirs('model', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('/home/ubuntu/upload/mail_data.csv')

# Display basic dataset information
print("\nDataset Overview:")
print(f"Shape: {df.shape}")
print("\nClass Distribution:")
print(df['Category'].value_counts())
print("\nClass Distribution (%):")
print(df['Category'].value_counts(normalize=True) * 100)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values if any
df = df.dropna()

# Encode target variable
print("\nEncoding target variable...")
df.loc[df['Category'] == 'spam', 'Category'] = 0
df.loc[df['Category'] == 'ham', 'Category'] = 1
df['Category'] = df['Category'].astype('int')

# Split the dataset
print("\nSplitting dataset into train and test sets...")
X = df['Message']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Feature extraction
print("\nExtracting features using TF-IDF...")
start_time = time.time()
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)
feature_extraction_time = time.time() - start_time
print(f"Feature extraction completed in {feature_extraction_time:.2f} seconds")

# Train the model
print("\nTraining Logistic Regression model...")
start_time = time.time()
model = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000, random_state=42)
model.fit(X_train_features, y_train)
training_time = time.time() - start_time
print(f"Model training completed in {training_time:.2f} seconds")

# Model evaluation function
def evaluate_model(model, X_train_features, y_train, X_test_features, y_test):
    """
    Comprehensive model evaluation with multiple metrics and visualizations
    """
    results = {}
    
    # Make predictions
    print("\nGenerating predictions...")
    start_time = time.time()
    y_train_pred = model.predict(X_train_features)
    y_test_pred = model.predict(X_test_features)
    
    # Get probability predictions
    y_train_proba = model.predict_proba(X_train_features)
    y_test_proba = model.predict_proba(X_test_features)
    prediction_time = time.time() - start_time
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    
    # Calculate metrics
    print("\nCalculating performance metrics...")
    
    # Basic metrics
    results['accuracy_train'] = accuracy_score(y_train, y_train_pred)
    results['accuracy_test'] = accuracy_score(y_test, y_test_pred)
    
    results['precision_train'] = precision_score(y_train, y_train_pred)
    results['precision_test'] = precision_score(y_test, y_test_pred)
    
    results['recall_train'] = recall_score(y_train, y_train_pred)
    results['recall_test'] = recall_score(y_test, y_test_pred)
    
    results['f1_train'] = f1_score(y_train, y_train_pred)
    results['f1_test'] = f1_score(y_test, y_test_pred)
    
    # Probability-based metrics
    results['log_loss_train'] = log_loss(y_train, y_train_proba)
    results['log_loss_test'] = log_loss(y_test, y_test_proba)
    
    results['brier_score_train'] = brier_score_loss(y_train, y_train_proba[:, 1])
    results['brier_score_test'] = brier_score_loss(y_test, y_test_proba[:, 1])
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
    results['roc_auc'] = auc(fpr, tpr)
    
    # Average Precision (PR AUC)
    results['avg_precision'] = average_precision_score(y_test, y_test_proba[:, 1])
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_features, y_train, cv=cv, scoring='f1')
    results['cv_f1_mean'] = cv_scores.mean()
    results['cv_f1_std'] = cv_scores.std()
    
    # Print metrics
    print("\n===== Model Performance Metrics =====")
    print(f"Accuracy (Train): {results['accuracy_train']:.4f}")
    print(f"Accuracy (Test): {results['accuracy_test']:.4f}")
    print(f"Precision (Test): {results['precision_test']:.4f}")
    print(f"Recall (Test): {results['recall_test']:.4f}")
    print(f"F1 Score (Test): {results['f1_test']:.4f}")
    print(f"Log Loss (Test): {results['log_loss_test']:.4f}")
    print(f"Brier Score (Test): {results['brier_score_test']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"Average Precision (PR AUC): {results['avg_precision']:.4f}")
    print(f"Cross-validation F1 (mean): {results['cv_f1_mean']:.4f}")
    print(f"Cross-validation F1 (std): {results['cv_f1_std']:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Spam', 'Ham']))
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Spam', 'Ham'], 
                yticklabels=['Spam', 'Ham'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png')
    
    # 2. ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {results["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('visualizations/roc_curve.png')
    
    # 3. Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba[:, 1])
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {results["avg_precision"]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('visualizations/precision_recall_curve.png')
    
    # 4. Feature Importance (top 20)
    plt.figure(figsize=(12, 10))
    feature_names = vectorizer.get_feature_names_out()
    
    # For logistic regression, coefficients represent feature importance
    importance = model.coef_[0]
    
    # Get indices of top 20 features (absolute value)
    indices = np.argsort(np.abs(importance))[-20:]
    
    # Plot horizontal bar chart
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Coefficient Value')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    
    # 5. Performance Metrics Comparison
    plt.figure(figsize=(12, 8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    train_values = [results['accuracy_train'], results['precision_train'], 
                   results['recall_train'], results['f1_train']]
    test_values = [results['accuracy_test'], results['precision_test'], 
                  results['recall_test'], results['f1_test']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, train_values, width, label='Train')
    plt.bar(x + width/2, test_values, width, label='Test')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/performance_comparison.png')
    
    return results, prediction_time

# Evaluate the model
results, prediction_time = evaluate_model(model, X_train_features, y_train, X_test_features, y_test)

# Performance metrics for API documentation
print("\n===== Performance Metrics for API Documentation =====")
print(f"Average prediction time per message: {prediction_time/len(X_test):.5f} seconds")
print(f"Model size: {os.path.getsize('model/spam_model.joblib')/1024/1024:.2f} MB (after saving)")
print(f"Vectorizer size: {os.path.getsize('model/vectorizer.joblib')/1024/1024:.2f} MB (after saving)")

# Save model and vectorizer
print("\nSaving model and vectorizer...")
joblib.dump(model, 'model/spam_model.joblib')
joblib.dump(vectorizer, 'model/vectorizer.joblib')

# Save performance metrics
with open('model/performance_metrics.txt', 'w') as f:
    f.write("===== Model Performance Metrics =====\n")
    f.write(f"Accuracy (Train): {results['accuracy_train']:.4f}\n")
    f.write(f"Accuracy (Test): {results['accuracy_test']:.4f}\n")
    f.write(f"Precision (Test): {results['precision_test']:.4f}\n")
    f.write(f"Recall (Test): {results['recall_test']:.4f}\n")
    f.write(f"F1 Score (Test): {results['f1_test']:.4f}\n")
    f.write(f"Log Loss (Test): {results['log_loss_test']:.4f}\n")
    f.write(f"Brier Score (Test): {results['brier_score_test']:.4f}\n")
    f.write(f"ROC AUC: {results['roc_auc']:.4f}\n")
    f.write(f"Average Precision (PR AUC): {results['avg_precision']:.4f}\n")
    f.write(f"Cross-validation F1 (mean): {results['cv_f1_mean']:.4f}\n")
    f.write(f"Cross-validation F1 (std): {results['cv_f1_std']:.4f}\n")
    f.write(f"\nAverage prediction time per message: {prediction_time/len(X_test):.5f} seconds\n")

print("\nModel training and evaluation completed successfully!")

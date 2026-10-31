from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on test data and computes key metrics:
    - Accuracy, Precision, Recall, Weighted F1-Score
    - Confusion Matrix
    - Classification Report (dict and text string)
    - ROC-AUC Score (Multi-class One-vs-Rest strategy, if model supports predict_proba)
    """
    preds = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, preds)
    clf_report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    clf_report_text = classification_report(y_test, preds, zero_division=0)
    
    # Calculate ROC-AUC if probability estimates are available
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)
            roc_auc = float(roc_auc_score(y_test, probs, multi_class='ovr', average='weighted'))
        except Exception:
            roc_auc = None

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "classification_report_text": clf_report_text
    }
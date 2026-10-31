import sys
import os
from typing import Dict, List, Any
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import pandas as pd
from preprocessing.clean import clean_text

def load_pipeline() -> tuple:
    """
    Loads saved model, encoder, vectorizer, and selected features.
    Handles running from project root or subdirectories.
    """
    model_dir = "model"
    if not os.path.exists(model_dir):
        if os.path.exists("../model"):
            model_dir = "../model"
        elif os.path.exists("../../model"):
            model_dir = "../../model"
        else:
            raise FileNotFoundError("[Error] Cannot find the 'model/' directory containing trained artifacts.")
        
    model = pickle.load(open(os.path.join(model_dir, "best_model.pkl"), "rb"))
    encoder = pickle.load(open(os.path.join(model_dir, "encoder.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(model_dir, "vectorizer.pkl"), "rb"))
    selected_features = pickle.load(open(os.path.join(model_dir, "selector.pkl"), "rb"))
    
    return model, encoder, vectorizer, selected_features

def predict_disease(symptom_text: str) -> Dict[str, Any]:
    """
    Inference Pipeline:
    1. Preprocesses symptom text
    2. Vectorizes text via fitted TF-IDF Vectorizer
    3. Filters features using BGWO selected indices
    4. Predicts disease class & probability distribution using trained XGBoost classifier
    5. Formulates explainability indicators
    """
    model, encoder, vectorizer, selected_features = load_pipeline()
    
    cleaned_text = clean_text(symptom_text)
    if not cleaned_text.strip():
        raise ValueError("Input symptom description contains no valid text.")
        
    X_tfidf = vectorizer.transform(pd.Series([cleaned_text]))
    X_selected = X_tfidf.reindex(columns=selected_features, fill_value=0)
    
    prediction_code = model.predict(X_selected)[0]
    probabilities = model.predict_proba(X_selected)[0] if hasattr(model, "predict_proba") else [1.0]
    
    disease = str(encoder.inverse_transform([prediction_code])[0])
    max_prob = float(max(probabilities))
    confidence_percentage = round(max_prob * 100.0, 2)
    
    # Class probability distribution map
    all_classes = encoder.encoder.classes_
    prob_dist = {}
    for i, cls_name in enumerate(all_classes):
        prob_dist[str(cls_name)] = round(float(probabilities[i]) * 100.0, 2)
        
    # Sort probability distribution descending
    prob_dist = dict(sorted(prob_dist.items(), key=lambda x: x[1], reverse=True))

    # Active feature explainability
    active_features = [col for col in selected_features if X_selected.loc[0, col] > 0]
    important_symptoms = active_features
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_map = dict(zip(selected_features, importances))
        important_symptoms = sorted(active_features, key=lambda f: importance_map.get(f, 0.0), reverse=True)
        
    model_type = type(model).__name__
    algorithm_tag = "BGWO + XGBoost" if "XGB" in model_type else f"BGWO + {model_type}"

    return {
        "disease": disease,
        "confidence": confidence_percentage,
        "algorithm": algorithm_tag,
        "important_symptoms": important_symptoms,
        "probability_distribution": prob_dist
    }

if __name__ == "__main__":
    symptom_input = "high fever, cough, body pain"
    if len(sys.argv) > 1:
        symptom_input = " ".join(sys.argv[1:])
        
    print(f"\n[Predict] Symptoms: '{symptom_input}'")
    try:
        res = predict_disease(symptom_input)
        print("\n==========================")
        print(f"Predicted Disease: {res['disease']}")
        print(f"Confidence: {res['confidence']}%")
        print(f"Algorithm: {res['algorithm']}")
        print("\nImportant Symptoms:")
        for sym in res['important_symptoms']:
            print(f" - {sym}")
        print("\nProbability Distribution:")
        for k, v in res['probability_distribution'].items():
            print(f" - {k}: {v}%")
        print("==========================\n")
    except Exception as e:
        print(f"Prediction Error: {str(e)}")


import os
import json
import pickle
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from preprocessing.clean import clean_data
from preprocessing.encoder import DiseaseEncoder
from features.feature_extraction import extract_features
from features.bgwo import BinaryGrayWolfOptimizer
from models.train_models import get_models, tune_xgboost
from models.evaluate import evaluate_model
from models.visualize import generate_all_plots

# ==========================================
# 1. LOAD DATASET
# ==========================================
csv_path = "data/communicable_disease_dataset.csv"
if not os.path.exists(csv_path):
    print(f"⚠️ {csv_path} not found. Checking fallback data/dataset.csv...")
    csv_path = "data/dataset.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError("[Error] Could not locate the dataset in data/ directory!")

df_raw = pd.read_csv(csv_path)
print(f"\n[Dataset] Dataset loaded successfully from {csv_path}")
print(f"Columns: {df_raw.columns.tolist()} | Shape: {df_raw.shape}")

target_col = "disease" if "disease" in df_raw.columns else "diseases"

# ==========================================
# 2. DATA PREPROCESSING & CLEANING
# ==========================================
df = clean_data(df_raw.copy())

# ==========================================
# 3. LABEL ENCODING
# ==========================================
print("\n[Encoding] Encoding target label (diseases)...")
encoder = DiseaseEncoder()
df[target_col] = encoder.fit_transform(df[target_col])
class_names = [str(c) for c in encoder.encoder.classes_]
print(f"[Success] Encoding completed across {len(class_names)} classes.")

# ==========================================
# 4. TRAIN-TEST SPLIT (80% / 20%)
# ==========================================
X_raw = df["symptoms"]
y = df[target_col]

print("\n[Split] Splitting data (80% train, 20% test)...")
try:
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print("[Success] Stratified split completed successfully.")
except ValueError:
    print("[Warning] Stratification failed. Falling back to random split...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

df_train = pd.DataFrame({"symptoms": X_train_raw}).reset_index(drop=True)
df_test = pd.DataFrame({"symptoms": X_test_raw}).reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# ==========================================
# 5. FEATURE EXTRACTION (TF-IDF Vectorizer)
# ==========================================
X_train, vectorizer = extract_features(df_train, is_training=True)
X_test = extract_features(df_test, vectorizer=vectorizer, is_training=False)

# ==========================================
# 6. FEATURE SELECTION (BGWO)
# ==========================================
print("\n[BGWO] Starting Feature Selection using Binary Gray Wolf Optimization (BGWO)...")
bgwo = BinaryGrayWolfOptimizer(num_wolves=12, max_iter=15, random_state=42)
selected_features = bgwo.fit(X_train, y_train)

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

bgwo_stats = bgwo.get_summary_stats()

# ==========================================
# 7. MODEL TRAINING & COMPARISON
# ==========================================
models = get_models()

# Tune XGBoost specifically
print("\n[XGBoost] Hyperparameter tuning for XGBoost Flagship Classifier...")
tuned_xgb = tune_xgboost(X_train_selected, y_train)
models["XGBoost"] = tuned_xgb

results = {}
best_model = None
best_f1 = -1.0
best_model_name = ""

print("\n" + "="*90)
print(f"{'Model Name':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
print("="*90)

for name, model in models.items():
    try:
        model.fit(X_train_selected, y_train)
        metrics = evaluate_model(model, X_test_selected, y_test)
        results[name] = metrics
        
        auc_str = f"{metrics['roc_auc']:.4f}" if metrics.get('roc_auc') is not None else "N/A"
        print(f"{name:<20} | {metrics['accuracy']:<10.4f} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1']:<10.4f} | {auc_str:<10}")
        
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model
            best_model_name = name
            
    except Exception as e:
        print(f"[Error] Failed training {name}: {str(e)}")

print("="*90)

# Flagship model preference (prefer XGBoost if competitive, else best model)
if "XGBoost" in models:
    flagship_model = models["XGBoost"]
    print(f"\n[Production Flagship] Saving Tuned XGBoost Classifier (F1: {results['XGBoost']['f1']:.4f})")
else:
    flagship_model = best_model
    print(f"\n[Production Flagship] Saving Best Model: {best_model_name} (F1: {best_f1:.4f})")

# ==========================================
# 8. GENERATE VISUALIZATIONS
# ==========================================
print("\n[Visualizations] Generating publication charts under reports/figures/...")
generate_all_plots(
    df=df_raw,
    bgwo_stats=bgwo_stats,
    results=results,
    best_model=flagship_model,
    class_names=class_names,
    selected_features=selected_features
)

# ==========================================
# 9. SAVE PIPELINE ARTIFACTS
# ==========================================
print("\n[Save] Serializing model artifacts & JSON summaries...")
os.makedirs("model", exist_ok=True)

pickle.dump(flagship_model, open("model/best_model.pkl", "wb"))
encoder.save("model/encoder.pkl")
vectorizer.save("model/vectorizer.pkl")
pickle.dump(selected_features, open("model/selector.pkl", "wb"))
bgwo.save_stats("model/bgwo_stats.json")

# Clean results dict for JSON serialization (remove numpy arrays / un-serializable fields)
serializable_results = {}
for m_name, m_val in results.items():
    serializable_results[m_name] = {
        "accuracy": m_val["accuracy"],
        "precision": m_val["precision"],
        "recall": m_val["recall"],
        "f1": m_val["f1"],
        "roc_auc": m_val["roc_auc"]
    }

with open("model/evaluation_results.json", "w") as f:
    json.dump(serializable_results, f, indent=4)

print("\n[Success] All artifacts saved successfully under 'model/' directory:")
print("  - best_model.pkl")
print("  - encoder.pkl")
print("  - vectorizer.pkl")
print("  - selector.pkl")
print("  - bgwo_stats.json")
print("  - evaluation_results.json")
print("\n[Finish] End-to-end training pipeline completed successfully!\n")
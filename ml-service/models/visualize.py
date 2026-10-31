import os
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure figures output directory exists
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_dataset_distribution(df: pd.DataFrame, output_dir: str = FIGURES_DIR) -> str:
    """Plots and saves the dataset disease class distribution."""
    plt.figure(figsize=(10, 5))
    target_col = "disease" if "disease" in df.columns else "diseases"
    counts = df[target_col].value_counts()
    
    sns.barplot(x=counts.values, y=counts.index, palette="Blues_d")
    plt.title("Dataset Class Distribution across Communicable Diseases", fontsize=14, fontweight='bold')
    plt.xlabel("Sample Count", fontsize=12)
    plt.ylabel("Disease Category", fontsize=12)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "dataset_distribution.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Saved dataset distribution plot to {filepath}")
    return filepath

def plot_bgwo_feature_reduction(original_count: int, selected_count: int, output_dir: str = FIGURES_DIR) -> str:
    """Plots and saves the BGWO feature reduction comparison chart."""
    plt.figure(figsize=(7, 5))
    categories = ['Original TF-IDF Features', 'BGWO Selected Features']
    counts = [original_count, selected_count]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = plt.bar(categories, counts, color=colors, width=0.5)
    reduction_pct = ((original_count - selected_count) / original_count) * 100
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height / 2,
                 f'{int(height)}',
                 ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)
                 
    plt.title(f"BGWO Feature Reduction ({reduction_pct:.1f}% Reduced)", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Features", fontsize=12)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "bgwo_feature_reduction.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Saved BGWO feature reduction plot to {filepath}")
    return filepath

def plot_model_comparison(results: Dict[str, Dict[str, float]], output_dir: str = FIGURES_DIR) -> str:
    """Plots and saves model comparison bar charts (Accuracy, Precision, Recall, F1)."""
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in models]
    f1_scores = [results[m]["f1"] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#3498db')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score (%)', color='#2ecc71')
    
    ax.set_ylabel('Percentage Score (%)', fontsize=12)
    ax.set_title('Machine Learning Classifier Benchmark (BGWO Features)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, fontsize=11)
    ax.legend()
    ax.set_ylim([70, 105])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Saved model comparison plot to {filepath}")
    return filepath

def plot_precision_recall_f1(results: Dict[str, Dict[str, float]], output_dir: str = FIGURES_DIR) -> str:
    """Plots multi-metric (Precision, Recall, F1) comparison across models."""
    df_metrics = pd.DataFrame(results).T[['precision', 'recall', 'f1']] * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df_metrics.plot(kind='bar', ax=ax, colormap='viridis', width=0.7)
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Classifier Precision, Recall, and F1-Score Metrics', fontsize=14, fontweight='bold')
    ax.set_xticklabels(df_metrics.index, rotation=15, fontsize=11)
    ax.set_ylim([70, 105])
    plt.legend(title="Metrics")
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "precision_recall_f1_comparison.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Saved precision/recall/f1 plot to {filepath}")
    return filepath

def plot_confusion_matrix_heatmap(cm: List[List[int]], class_names: List[str], model_name: str = "XGBoost", output_dir: str = FIGURES_DIR) -> str:
    """Plots and saves the confusion matrix heatmap for a classifier."""
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix heatmap to {filepath}")
    return filepath

def plot_xgboost_feature_importance(model: Any, feature_names: List[str], top_n: int = 15, output_dir: str = FIGURES_DIR) -> str:
    """Plots and saves the top XGBoost feature importances."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        df_imp = df_imp.sort_values(by='importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(9, 6))
        sns.barplot(x='importance', y='feature', data=df_imp, palette='magma')
        plt.title(f"Top {top_n} Most Influential Features (XGBoost)", fontsize=14, fontweight='bold')
        plt.xlabel("Feature Importance Weight", fontsize=12)
        plt.ylabel("Symptom Feature", fontsize=12)
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, "xgboost_feature_importance.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved XGBoost feature importance plot to {filepath}")
        return filepath
    return ""

def generate_all_plots(df: pd.DataFrame, bgwo_stats: Dict[str, Any], results: Dict[str, Any], best_model: Any, class_names: List[str], selected_features: List[str], output_dir: str = FIGURES_DIR):
    """Generates all project visualization charts."""
    plot_dataset_distribution(df, output_dir)
    plot_bgwo_feature_reduction(bgwo_stats["original_feature_count"], bgwo_stats["selected_feature_count"], output_dir)
    plot_model_comparison(results, output_dir)
    plot_precision_recall_f1(results, output_dir)
    
    if "XGBoost" in results and "confusion_matrix" in results["XGBoost"]:
        plot_confusion_matrix_heatmap(results["XGBoost"]["confusion_matrix"], class_names, "XGBoost", output_dir)
    elif best_model and hasattr(best_model, "predict"):
        best_name = type(best_model).__name__
        cm = list(results.values())[0]["confusion_matrix"]
        plot_confusion_matrix_heatmap(cm, class_names, best_name, output_dir)
        
    if hasattr(best_model, 'feature_importances_'):
        plot_xgboost_feature_importance(best_model, selected_features, 15, output_dir)

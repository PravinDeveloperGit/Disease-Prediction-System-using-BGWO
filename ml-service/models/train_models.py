import logging
from typing import Dict, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def tune_xgboost(X_train, y_train) -> XGBClassifier:
    """
    Performs hyperparameter tuning for XGBoost using GridSearchCV.
    Tunes n_estimators, max_depth, learning_rate, and subsample.
    """
    logger.info("[XGBoost] Starting hyperparameter tuning via GridSearchCV...")
    base_xgb = XGBClassifier(
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False
    )
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    try:
        grid_search = GridSearchCV(
            estimator=base_xgb,
            param_grid=param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        logger.info(f"[XGBoost] Best Hyperparameters: {grid_search.best_params_}")
        logger.info(f"[XGBoost] Best CV Weighted F1-Score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    except Exception as e:
        logger.warning(f"[XGBoost] Grid Search failed ({str(e)}). Falling back to optimized default XGBoost.")
        default_xgb = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric='mlogloss',
            random_state=42
        )
        default_xgb.fit(X_train, y_train)
        return default_xgb

def get_models() -> Dict[str, Any]:
    """
    Returns a dictionary of machine learning model instances to train and compare.
    Includes Decision Tree, Random Forest, Naive Bayes, SVM, and XGBoost.
    """
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, kernel='rbf', random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric='mlogloss',
            random_state=42
        )
    }
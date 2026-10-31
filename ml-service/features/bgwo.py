import json
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BinaryGrayWolfOptimizer:
    """
    Binary Grey Wolf Optimization (BGWO) Metaheuristic for Feature Selection.
    Models natural hunting hierarchy (Alpha, Beta, Delta wolves) to select discriminative
    symptom features while reducing TF-IDF matrix dimensionality.
    """
    def __init__(self, num_wolves: int = 12, max_iter: int = 15, random_state: int = 42):
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.random_state = random_state
        self.best_wolf = None
        self.best_fitness = -1.0
        self.selected_features: List[str] = []
        self.original_feature_count: int = 0
        self.selected_feature_count: int = 0
        self.feature_reduction_percentage: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> List[str]:
        """
        Executes the BGWO feature selection algorithm on X_train.
        Returns the list of selected feature column names.
        """
        np.random.seed(self.random_state)
        self.original_feature_count = X_train.shape[1]
        num_features = self.original_feature_count

        logger.info(f"[BGWO] Initializing population: {self.num_wolves} wolves across {num_features} potential features...")

        # 1. Initialize wolf population randomly (binary 0 or 1)
        wolves = np.random.randint(0, 2, size=(self.num_wolves, num_features))

        # Ensure no wolf starts with 0 features
        for i in range(self.num_wolves):
            if np.sum(wolves[i]) == 0:
                wolves[i, np.random.randint(0, num_features)] = 1

        alpha_wolf = np.zeros(num_features)
        beta_wolf = np.zeros(num_features)
        delta_wolf = np.zeros(num_features)

        alpha_score = -float('inf')
        beta_score = -float('inf')
        delta_score = -float('inf')

        # 2. Optimization Loop
        for iteration in range(self.max_iter):
            a = 2.0 - 2.0 * (iteration / self.max_iter)

            for i in range(self.num_wolves):
                fitness = self._evaluate_fitness(X_train, y_train, wolves[i])

                if fitness > alpha_score:
                    delta_score = beta_score
                    delta_wolf = beta_wolf.copy()

                    beta_score = alpha_score
                    beta_wolf = alpha_wolf.copy()

                    alpha_score = fitness
                    alpha_wolf = wolves[i].copy()

                elif fitness > beta_score:
                    delta_score = beta_score
                    delta_wolf = beta_wolf.copy()

                    beta_score = fitness
                    beta_wolf = wolves[i].copy()

                elif fitness > delta_score:
                    delta_score = fitness
                    delta_wolf = wolves[i].copy()

            logger.info(f"[BGWO] Iteration {iteration+1:02d}/{self.max_iter:02d} | Best Fitness: {alpha_score:.4f}")

            # 3. Position Updates via Sigmoidal Transfer Function
            for i in range(self.num_wolves):
                for j in range(num_features):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1, C1 = 2.0 * a * r1 - a, 2.0 * r2
                    D_alpha = abs(C1 * alpha_wolf[j] - wolves[i, j])
                    X1 = alpha_wolf[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2, C2 = 2.0 * a * r1 - a, 2.0 * r2
                    D_beta = abs(C2 * beta_wolf[j] - wolves[i, j])
                    X2 = beta_wolf[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3, C3 = 2.0 * a * r1 - a, 2.0 * r2
                    D_delta = abs(C3 * delta_wolf[j] - wolves[i, j])
                    X3 = delta_wolf[j] - A3 * D_delta

                    X_avg = (X1 + X2 + X3) / 3.0
                    s_val = 1.0 / (1.0 + np.exp(-X_avg))

                    wolves[i, j] = 1 if s_val >= np.random.rand() else 0

                if np.sum(wolves[i]) == 0:
                    wolves[i, np.random.randint(0, num_features)] = 1

        self.best_wolf = alpha_wolf
        self.best_fitness = alpha_score

        selected_indices = np.where(self.best_wolf == 1)[0]
        if len(selected_indices) == 0:
            logger.warning("[BGWO] No features selected by wolves. Falling back to all features.")
            self.selected_features = X_train.columns.tolist()
        else:
            self.selected_features = X_train.columns[selected_indices].tolist()

        self.selected_feature_count = len(self.selected_features)
        self.feature_reduction_percentage = round(
            ((self.original_feature_count - self.selected_feature_count) / self.original_feature_count) * 100.0, 2
        )

        logger.info(f"[BGWO] Optimization Complete:")
        logger.info(f"  - Original Feature Count : {self.original_feature_count}")
        logger.info(f"  - Selected Feature Count : {self.selected_feature_count}")
        logger.info(f"  - Feature Reduction Rate : {self.feature_reduction_percentage}%")

        return self.selected_features

    def get_summary_stats(self) -> Dict[str, any]:
        """Returns feature selection metrics dictionary."""
        return {
            "original_feature_count": self.original_feature_count,
            "selected_feature_count": self.selected_feature_count,
            "feature_reduction_percentage": self.feature_reduction_percentage,
            "best_fitness_score": float(self.best_fitness)
        }

    def save_stats(self, filepath: str) -> None:
        """Saves BGWO statistics to JSON."""
        stats = self.get_summary_stats()
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Saved BGWO stats to {filepath}")

    def _evaluate_fitness(self, X_train: pd.DataFrame, y_train: pd.Series, wolf: np.ndarray) -> float:
        selected_indices = np.where(wolf == 1)[0]
        if len(selected_indices) == 0:
            return 0.0

        X_subset = X_train.iloc[:, selected_indices]
        clf = DecisionTreeClassifier(max_depth=5, random_state=self.random_state)

        try:
            scores = cross_val_score(clf, X_subset, y_train, cv=3, scoring='f1_weighted', error_score=0)
            f1 = np.mean(scores)
        except Exception:
            f1 = 0.0

        num_selected = len(selected_indices)
        total_features = X_train.shape[1]

        fitness = 0.7 * f1 + 0.3 * (1.0 - (num_selected / total_features))
        return fitness


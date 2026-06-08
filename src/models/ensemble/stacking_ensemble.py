import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

class StackingEnsembleMetaLearner:
    """
    Stacking Ensemble Meta-Learner combining outputs from:
    1. PortfolioGNN (spatial contagion & network dependencies)
    2. DemandTFT (probabilistic demand forecasting distress indices)
    3. CreditScorerXGB/LGBM (engineered feature credit scoring)
    
    Meta-learner: Logistic Regression or Random Forest to combine probabilities into a final PD.
    Computes Expected Loss (EL) = PD * LGD * EAD.
    """
    def __init__(self, meta_model: Optional[Any] = None):
        if meta_model is None:
            self.meta_model = LogisticRegression(C=1.0, random_state=42)
        else:
            self.meta_model = meta_model
            
    def construct_meta_features(
        self, 
        gnn_pd: np.ndarray, 
        tft_distress: np.ndarray, 
        xgb_pd: np.ndarray
    ) -> np.ndarray:
        """
        Concatenates outputs of base models into a feature matrix for the meta-learner.
        Outputs shapes: (N,) or (N, 1)
        Returns: (N, 3) matrix
        """
        # Ensure flat arrays
        gnn_pd_flat = gnn_pd.flatten()
        tft_distress_flat = tft_distress.flatten()
        xgb_pd_flat = xgb_pd.flatten()
        
        return np.column_stack((gnn_pd_flat, tft_distress_flat, xgb_pd_flat))
        
    def fit(self, meta_features: np.ndarray, y: np.ndarray):
        """
        Fits the meta-learner on the base model predictions.
        meta_features: Shape (N, 3)
        y: Shape (N,) binary target (default vs non-default)
        """
        self.meta_model.fit(meta_features, y)
        
    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        """
        Predicts final Default Probability (PD).
        Returns: (N,) probability array
        """
        return self.meta_model.predict_proba(meta_features)[:, 1]
        
    def predict(self, meta_features: np.ndarray) -> np.ndarray:
        """
        Predicts final default decision.
        Returns: (N,) binary array
        """
        return self.meta_model.predict(meta_features)
        
    def evaluate(self, meta_features: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        probs = self.predict_proba(meta_features)
        preds = self.predict(meta_features)
        
        try:
            auc = float(roc_auc_score(y, probs))
        except ValueError:
            auc = 0.50
            
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "auroc": auc  # Target is AUROC > 0.80
        }
        
    @staticmethod
    def compute_expected_loss(
        pd_value: Union[float, np.ndarray],
        lgd_value: Union[float, np.ndarray],
        ead_value: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Computes the Expected Loss (EL) for credit risk assessment.
        Formula: EL = PD * LGD * EAD
        Where:
            PD: Probability of Default (0.0 to 1.0)
            LGD: Loss Given Default (0.0 to 1.0, e.g., 0.45 represents 45% loss)
            EAD: Exposure at Default ($ amount)
        """
        return pd_value * lgd_value * ead_value
        
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.meta_model, path)
        
    def load_model(self, path: str):
        if os.path.exists(path):
            self.meta_model = joblib.load(path)
        else:
            raise FileNotFoundError(f"Stacking Meta-Learner model file not found at {path}")

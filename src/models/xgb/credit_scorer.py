import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import shap
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

class CreditScorerXGB:
    """XGBoost Classifier wrapper for credit risk scoring and probability of default estimation."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42
        }
        if params:
            self.default_params.update(params)
        self.model = xgb.XGBClassifier(**self.default_params)
        self.explainer = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set: Optional[list] = None):
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            try:
                self.explainer = shap.Explainer(self.model)
            except Exception:
                self.explainer = None
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1] # Return probability of class 1 (default)
        
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        if self.explainer is None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                try:
                    self.explainer = shap.Explainer(self.model)
                except Exception:
                    self.explainer = None
                    
        if self.explainer is not None:
            try:
                if hasattr(self.explainer, "shap_values"):
                    return self.explainer.shap_values(X)
                shap_vals = self.explainer(X)
                if hasattr(shap_vals, "values"):
                    return shap_vals.values
                return shap_vals
            except Exception:
                pass
        return np.zeros((len(X), X.shape[1]))
        
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        probs = self.predict_proba(X)
        
        # Safe ROC-AUC calculation in case of single class in evaluation set
        try:
            auc = float(roc_auc_score(y, probs))
        except ValueError:
            auc = 0.50
            
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "auroc": auc
        }
        
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        
    def load_model(self, path: str):
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.explainer = shap.TreeExplainer(self.model)
        else:
            raise FileNotFoundError(f"XGBoost model file not found at {path}")


class CreditScorerLGBM:
    """LightGBM Classifier wrapper for credit risk scoring and probability of default estimation."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary",
            "metric": "binary_logloss",
            "random_state": 42,
            "verbose": -1
        }
        if params:
            self.default_params.update(params)
        self.model = lgb.LGBMClassifier(**self.default_params)
        self.explainer = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set: Optional[list] = None):
        # Silence callbacks to avoid console spam
        self.model.fit(
            X, y, 
            eval_set=eval_set
        )
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            try:
                self.explainer = shap.Explainer(self.model)
            except Exception:
                self.explainer = None
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
        
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        if self.explainer is None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                try:
                    self.explainer = shap.Explainer(self.model)
                except Exception:
                    self.explainer = None
                    
        if self.explainer is not None:
            try:
                if hasattr(self.explainer, "shap_values"):
                    shap_vals = self.explainer.shap_values(X)
                else:
                    shap_vals = self.explainer(X)
                    if hasattr(shap_vals, "values"):
                        shap_vals = shap_vals.values
                if isinstance(shap_vals, list) and len(shap_vals) > 1:
                    return shap_vals[1]
                return shap_vals
            except Exception:
                pass
        return np.zeros((len(X), X.shape[1]))
        
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        probs = self.predict_proba(X)
        
        try:
            auc = float(roc_auc_score(y, probs))
        except ValueError:
            auc = 0.50
            
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "auroc": auc
        }
        
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        
    def load_model(self, path: str):
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.explainer = shap.TreeExplainer(self.model)
        else:
            raise FileNotFoundError(f"LightGBM model file not found at {path}")

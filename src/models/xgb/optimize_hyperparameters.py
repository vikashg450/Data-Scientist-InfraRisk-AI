import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from src.data.world_bank_loader import WorldBankLoader
from src.features.financial_features import FinancialFeaturesCalculator
from src.features.macro_features import MacroFeaturesCalculator
from src.features.satellite_features import SatelliteFeaturesExtractor
from src.features.fusion_features import FusionFeaturesCalculator

def load_training_data():
    """Generates synthetic dataset using project loaders and features."""
    loader = WorldBankLoader(cache_dir="data")
    df_projects = loader.generate_mock_ppi_data(num_records=1000)
    
    # Extract features
    fin_calc = FinancialFeaturesCalculator()
    df_fin = fin_calc.compute_project_features(df_projects)
    
    sat_ext = SatelliteFeaturesExtractor()
    df_sat = sat_ext.compute_all_satellite_features(df_projects)
    
    macro_calc = MacroFeaturesCalculator()
    combined_df = loader.get_combined_dataset(num_projects=1000)
    df_macro = macro_calc.compute_all_macro_features(combined_df)
    
    fusion_calc = FusionFeaturesCalculator()
    df_features = fusion_calc.fuse_all_features(df_projects, df_sat, df_macro)
    
    # Binary default indicator based on DSCR
    y = (df_projects["dscr"] < 1.15).astype(int).values
    X = df_features.drop(columns=["project_id", "country_code", "default_probability_pd"])
    
    return X, y

def objective_xgb(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 250),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        
        try:
            scores.append(roc_auc_score(y_val, preds))
        except ValueError:
            scores.append(0.5)
            
    return np.mean(scores)

def objective_lgb(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 250),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "objective": "binary",
        "random_state": 42,
        "verbose": -1
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        
        try:
            scores.append(roc_auc_score(y_val, preds))
        except ValueError:
            scores.append(0.5)
            
    return np.mean(scores)

def run_optimization(n_trials=200):
    print("Loading training data...")
    X, y = load_training_data()
    
    print(f"Running XGBoost hyperparameter optimization ({n_trials} trials)...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=n_trials)
    
    print("\nBest XGBoost parameters:")
    print(study_xgb.best_params)
    print(f"Best CV AUROC: {study_xgb.best_value:.4f}")
    
    print(f"\nRunning LightGBM hyperparameter optimization ({n_trials} trials)...")
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(lambda trial: objective_lgb(trial, X, y), n_trials=n_trials)
    
    print("\nBest LightGBM parameters:")
    print(study_lgb.best_params)
    print(f"Best CV AUROC: {study_lgb.best_value:.4f}")

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run_optimization(n_trials=20) # Running 20 trials for demonstration, can be set to 200

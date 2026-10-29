import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import shap
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

class CreditApprovalModel:
    def __init__(self, dataset_path: str, target_col: str, sensitive_col: str):
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        
        self.model = None
        self.explainer = None
        self.feature_names = []
        
        self.performance_report = {}
        self.fairness_report = {}
        self.global_shap = {}

    def train_and_evaluate(self):
        # 1. Load and Preprocess Data
        try:
            df = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset '{self.dataset_path}' not found.")

        # The dataset is already tree-ready, but we ensure no NaN values break the model
        df = df.dropna()

        # Isolate target and features
        y = df[self.target_col].astype(int)
        X = df.drop(columns=[self.target_col])
        
        # Ensure booleans are converted to ints (XGBoost preference)
        for col in X.select_dtypes(include=['bool']).columns:
            X[col] = X[col].astype(int)
            
        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Train XGBoost Model
        self.model = XGBClassifier(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.1, 
            random_state=42, 
            use_label_encoder=False, 
            eval_metric="logloss"
        )
        self.model.fit(X_train, y_train)

        # 3. Calculate Performance Metrics
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        cm = confusion_matrix(y_test, y_pred)
        self.performance_report = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": {
                "True_Negative": int(cm[0][0]),
                "False_Positive": int(cm[0][1]),
                "False_Negative": int(cm[1][0]),
                "True_Positive": int(cm[1][1])
            }
        }

        # 4. Calculate Fairness Metrics (Grouped by Gender)
        if self.sensitive_col in X_test.columns:
            sensitive_features = X_test[self.sensitive_col]
            self.fairness_report = {
                "demographic_parity_difference": float(demographic_parity_difference(
                    y_test, y_pred, sensitive_features=sensitive_features
                )),
                "equalized_odds_difference": float(equalized_odds_difference(
                    y_test, y_pred, sensitive_features=sensitive_features
                ))
            }
        else:
            self.fairness_report = {"error": f"Sensitive column '{self.sensitive_col}' not found."}

        # 5. Initialize SHAP
        self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer.shap_values(X_test)
        
        # Average absolute SHAP values for global feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        self.global_shap = dict(zip(self.feature_names, mean_abs_shap.tolist()))

    def prepare_input(self, input_dict: dict) -> pd.DataFrame:
        input_df = pd.DataFrame([input_dict])
        
        # Ensure exact column match and ordering
        for col in self.feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Cast to numeric where necessary
        input_df = input_df[self.feature_names].astype(float)
        return input_df

    def predict(self, input_dict: dict) -> float:
        input_df = self.prepare_input(input_dict)
        prob = self.model.predict_proba(input_df)[0][1]
        return round(prob * 100, 2)

    def get_local_shap(self, input_dict: dict) -> dict:
        input_df = self.prepare_input(input_dict)
        shap_values = self.explainer.shap_values(input_df)
        return dict(zip(self.feature_names, shap_values[0].tolist()))
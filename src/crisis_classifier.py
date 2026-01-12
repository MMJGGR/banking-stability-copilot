"""
==============================================================================
Crisis Classifier - XGBoost with SHAP Interpretability
==============================================================================
CRISP-DM Phase: Modeling

This module implements a supervised crisis prediction model following
academic best practices:
- XGBoost/LightGBM for prediction (Bundesbank, BIS research)
- SHAP values for interpretability (IMF/policy requirement)
- Class imbalance handling (crises are rare events)

Architecture:
    Hybrid Score = 0.4 × Economic_Pillar
                 + 0.4 × Industry_Pillar  
                 + 0.2 × Supervised_Crisis_Probability

Author: Banking Copilot
Date: 2026-01-02
==============================================================================
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings('ignore')

# Check for XGBoost and SHAP
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: XGBoost not installed. Using RandomForest fallback.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: SHAP not installed. Feature importance will be limited.")

# Fallback to RandomForest if needed
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CACHE_DIR


class CrisisClassifier:
    """
    Supervised crisis prediction model with interpretability.
    
    CRISP-DM: Modeling Phase
    
    Following academic recommendations:
    - XGBoost/LightGBM preferred (BIS, Bundesbank research)
    - SHAP for policy-relevant interpretability
    - 3-year prediction horizon (standard in literature)
    """
    
    # Feature priority based on academic literature
    # Higher priority = more predictive in published research
    FEATURE_PRIORITY = {
        # Tier 1: Critical predictors (BIS, IMF consensus)
        'credit_to_gdp_gap': 1,    
        'debt_service_gdp': 1,
        'npl_ratio': 1,
        'external_debt_gdp': 1,
        
        # Tier 2: Strong predictors
        'liquid_assets_st_liab': 2,
        'current_account_gdp': 2,
        'capital_adequacy': 2,
        'govt_debt_gdp': 2,
        
        # Tier 3: Supporting indicators
        'gdp_growth': 3,
        'inflation': 3,
        'roe': 3,
        'fx_loan_exposure': 3,
        'gdp_per_capita': 3,
    }
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 5,
                 learning_rate: float = 0.1,
                 random_state: int = 42):
        """
        Initialize crisis classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Max tree depth (keep low to avoid overfitting)
            learning_rate: Learning rate for boosting
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names_ = []
        self.feature_importance_ = {}
        self.fitted_ = False
        
        # Output directory for visualizations
        self.output_dir = os.path.join(CACHE_DIR, 'eda')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _create_model(self, n_positive: int, n_negative: int):
        """
        Create classifier model with class imbalance handling.
        
        CRISP-DM: Modeling - Algorithm Selection
        """
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = n_negative / max(n_positive, 1)
        
        if HAS_XGBOOST:
            return xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
        else:
            # Fallback to RandomForest with class weighting
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def fit(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> 'CrisisClassifier':
        """
        Fit the crisis classifier with cross-validation and generate diagnostics.
        """
        print("\n" + "="*70)
        print("TRAINING CRISIS CLASSIFIER")
        print("="*70)
        
        self.feature_names_ = list(X.columns)
        
        X_filled = X.copy()
        numeric_cols = X_filled.select_dtypes(include=['number']).columns
        self.numeric_cols_ = list(numeric_cols)
        X_filled[numeric_cols] = X_filled[numeric_cols].fillna(X_filled[numeric_cols].median())
        
        X_scaled = self.scaler.fit_transform(X_filled[numeric_cols])
        
        n_positive = int(y.sum())
        n_negative = len(y) - n_positive
        self.model = self._create_model(n_positive, n_negative)
        
        # --- CROSS-VALIDATION WITH ROC PLOTTING ---
        print(f"\n--- {cv}-Fold Stratified Cross-Validation & ROC Curves ---")
        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        plt.figure(figsize=(10, 8))
        
        min_class_count = min(n_positive, n_negative)
        actual_cv = min(cv, min_class_count) if min_class_count > 1 else 2
        
        cv_splitter = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=self.random_state)
        
        for i, (train_idx, val_idx) in enumerate(cv_splitter.split(X_scaled, y)):
            self.model.fit(X_scaled[train_idx], y.iloc[train_idx])
            probas_ = self.model.predict_proba(X_scaled[val_idx])[:, 1]
            
            fpr, tpr, thresholds = roc_curve(y.iloc[val_idx], probas_)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            
            # Interp
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')
        
        # Plot Mean ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})', lw=2, alpha=0.8)
        
        # Plot Chance
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (Cross-Validation)')
        plt.legend(loc="lower right")
        
        roc_path = os.path.join(self.output_dir, 'cv_roc_curve.png')
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved CV ROC Plot: {roc_path}")
        print(f"  Mean CV ROC-AUC:   {mean_auc:.3f} (+/- {std_auc*2:.3f})")
        
        self.cv_scores_ = np.array(aucs)
        
        # Train final model
        print("\n  Training final model on all data...")
        self.model.fit(X_scaled, y)
        self._compute_feature_importance(X_scaled, y)
        self.fitted_ = True
        
        return self
    
    def _compute_feature_importance(self, X: np.ndarray, y: pd.Series):
        """
        Compute feature importance using SHAP if available.
        
        CRISP-DM: Evaluation - Model Interpretation
        """
        print("\n--- Computing Feature Importance ---")
        
        if HAS_SHAP and HAS_XGBOOST:
            try:
                # SHAP values for global feature importance
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
                
                # Mean absolute SHAP value per feature
                shap_importance = np.abs(shap_values).mean(axis=0)
                
                # Use numeric_cols_ since X_scaled only has numeric features
                feature_names_for_shap = self.numeric_cols_ if hasattr(self, 'numeric_cols_') else self.feature_names_
                self.feature_importance_ = dict(zip(
                    feature_names_for_shap,
                    shap_importance / shap_importance.max()  # Normalize 0-1
                ))
                
                print("  Using SHAP values for feature importance")
                
                # Save SHAP summary plot
                self._plot_shap_summary(X, shap_values)
                
            except Exception as e:
                print(f"  SHAP failed: {e}")
                self._use_builtin_importance()
        else:
            self._use_builtin_importance()
        
        # Print top features
        print("\n  Top 10 Feature Importance:")
        sorted_importance = sorted(
            self.feature_importance_.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for i, (feat, imp) in enumerate(sorted_importance[:10], 1):
            print(f"    {i}. {feat}: {imp:.3f}")
    
    def _use_builtin_importance(self):
        """Use built-in feature importance (fallback)."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            importance = importance / importance.max()
            feature_names = self.numeric_cols_ if hasattr(self, 'numeric_cols_') else self.feature_names_
            self.feature_importance_ = dict(zip(feature_names, importance))
            print("  Using built-in feature importance")
    
    def _plot_shap_summary(self, X: np.ndarray, shap_values: np.ndarray):
        """Generate and save SHAP summary plot."""
        try:
            plt.figure(figsize=(10, 8))
            feature_names = self.numeric_cols_ if hasattr(self, 'numeric_cols_') else self.feature_names_
            shap.summary_plot(
                shap_values, X, 
                feature_names=feature_names,
                show=False
            )
            plt.title('SHAP Feature Importance (CRISP-DM: Interpretation)', fontsize=12)
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, 'shap_summary.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {filepath}")
        except Exception as e:
            print(f"  Could not save SHAP plot: {e}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict crisis probability.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of crisis probabilities
        """
        if not self.fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Only compute median for numeric columns (avoid _year string columns)
        X_filled = X.copy()
        numeric_cols = X_filled.select_dtypes(include=['number']).columns
        X_filled[numeric_cols] = X_filled[numeric_cols].fillna(X_filled[numeric_cols].median())
        X_scaled = self.scaler.transform(X_filled[numeric_cols])
        
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary crisis outcome."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model and save Confusion Matrix."""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        try:
            auc_roc = roc_auc_score(y, y_proba)
        except:
            auc_roc = 0.5
            
        print(f"  AUC-ROC (Training): {auc_roc:.3f}")
        
        # Confusion Matrix Plot
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Crisis', 'Crisis'])
        
        plt.figure(figsize=(6, 6))
        disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
        plt.title('Confusion Matrix (Training Set)')
        
        cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved Confusion Matrix: {cm_path}")
        
        return {
            'auc_roc': auc_roc,
            'accuracy': (y_pred == y).mean()
        }
    
    def save(self, path: str = None):
        """Save trained model."""
        path = path or os.path.join(CACHE_DIR, 'crisis_classifier.pkl')
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names_,
                'numeric_cols': getattr(self, 'numeric_cols_', self.feature_names_),
                'feature_importance': self.feature_importance_,
                'fitted': self.fitted_,
            }, f)
        
        print(f"\n  Saved model to: {path}")
    
    @classmethod
    def load(cls, path: str = None) -> 'CrisisClassifier':
        """Load trained model."""
        path = path or os.path.join(CACHE_DIR, 'crisis_classifier.pkl')
        
        classifier = cls()
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        classifier.feature_names_ = data['feature_names']
        classifier.numeric_cols_ = data.get('numeric_cols', data['feature_names'])  # Backward compatible
        classifier.feature_importance_ = data['feature_importance']
        classifier.fitted_ = data['fitted']
        
        return classifier


class HybridRiskScorer:
    """
    Hybrid risk scoring combining unsupervised and supervised approaches.
    
    CRISP-DM: Deployment-ready scoring system
    
    Architecture:
        Score = 0.4 × Economic_Pillar (PCA-based)
              + 0.4 × Industry_Pillar (PCA-based, includes liquidity)
              + 0.2 × Supervised_Crisis_Probability
    """
    
    def __init__(self):
        self.crisis_classifier = None
        self.fitted_ = False
    
    def compute_hybrid_score(self,
                            economic_score: float,
                            industry_score: float,
                            crisis_probability: float,
                            w_economic: float = 0.4,
                            w_industry: float = 0.4,
                            w_supervised: float = 0.2) -> float:
        """
        Compute hybrid risk score.
        
        Args:
            economic_score: Economic pillar score (0-100, higher = stronger)
            industry_score: Industry pillar score (0-100, higher = stronger)
            crisis_probability: Supervised crisis probability (0-1)
            w_*: Component weights (must sum to 1)
        
        Returns:
            Hybrid risk score (1-10 scale, 1 = lowest risk)
        """
        # Validate weights
        assert abs(w_economic + w_industry + w_supervised - 1.0) < 0.001
        
        # Combine components (higher = better/safer)
        # Convert crisis probability to "safety" score (1 - prob)
        combined = (
            w_economic * economic_score +
            w_industry * industry_score +
            w_supervised * (1 - crisis_probability) * 100
        )
        
        # Clamp combined to 0-100 range before converting
        combined = np.clip(combined, 0, 100)
        
        # Convert to 1-10 risk scale
        # Higher combined = lower risk number
        risk_score = 1 + 9 * (1 - combined / 100)
        
        return np.clip(risk_score, 1, 10)


# =============================================================================
# Main training pipeline
# =============================================================================

def train_crisis_model():
    """
    Train the crisis classifier on available data.
    
    CRISP-DM: Full modeling pipeline
    """
    from src.crisis_labels import CrisisLabels
    
    print("="*70)
    print("CRISIS CLASSIFIER TRAINING PIPELINE")
    print("CRISP-DM Phases: Data Preparation -> Modeling -> Evaluation")
    print("="*70)
    
    # Load features
    features_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
    if not os.path.exists(features_path):
        print(f"ERROR: Features not found at {features_path}")
        print("Run feature_engineering.py first")
        return None
    
    features = pd.read_parquet(features_path)
    print(f"\nLoaded features: {len(features)} countries, {len(features.columns)} columns")
    
    # Create crisis labels
    labels = CrisisLabels()
    
    # Add crisis target using 2005 as reference year
    # This captures 2008 GFC (3-year horizon: 2006-2008)
    features['crisis_target'] = features['country_code'].apply(
        lambda c: labels.get_crisis_target(c, 2005, horizon=3)
    )
    
    print(f"Crisis distribution:")
    print(f"  Positive (crisis): {features['crisis_target'].sum()}")
    print(f"  Negative (no crisis): {(~features['crisis_target'].astype(bool)).sum()}")
    
    # Prepare features for modeling
    # Exclude metadata columns (_year, _period) - they are strings and break median/scaling
    feature_cols = [c for c in features.columns 
                   if c not in ['country_code', 'crisis_target'] 
                   and not c.endswith('_period')
                   and not c.endswith('_year')]
    
    X = features[feature_cols]
    y = features['crisis_target']
    
    # Train classifier
    classifier = CrisisClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1
    )
    
    classifier.fit(X, y)
    
    # Evaluate (on training data - for demonstration)
    metrics = classifier.evaluate(X, y)
    
    # Save model
    classifier.save()
    
    return classifier, metrics


if __name__ == "__main__":
    classifier, metrics = train_crisis_model()

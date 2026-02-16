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
    Hybrid Score = 0.4 * Economic_Pillar
                 + 0.4 * Industry_Pillar  
                 + 0.2 * Supervised_Crisis_Probability

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
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import seaborn as sns
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
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("WARNING: SMOTE not installed. Class imbalance handling will be limited.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: SHAP not installed. Feature importance will be limited.")

# Fallback to RandomForest if needed
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # Already imported above

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
        # Tier 1: Critical predictors (BIS, IMF, Drehmann & Juselius 2014)
        'credit_to_gdp_gap': 1,    
        'debt_service_gdp': 1,
        'npl_ratio': 1,
        'external_debt_gdp': 1,
        'sovereign_exposure_ratio': 1,          # Sovereign-bank nexus (Acharya et al. 2014)
        'sovereign_liability_to_reserves': 1,   # External vulnerability (Obstfeld et al. 2010)
        
        # Tier 2: Strong predictors
        'liquid_assets_st_liab': 2,
        'current_account_gdp': 2,
        'capital_adequacy': 2,
        'govt_debt_gdp': 2,
        'inflation': 2,
        
        # Tier 3: Supporting indicators
        'gdp_growth': 3,
        'roe': 3,
        'fx_loan_exposure': 3,
        'fiscal_space': 3,          # Derived: fiscal balance - debt pressure
    }
    
    # Features EXCLUDED from the classifier to prevent wealth/size bias
    # These dominate SHAP but reflect economic scale, not vulnerability
    # Ref: Borio & Lowe (2002) - ratio-based indicators outperform levels
    EXCLUDED_FROM_CLASSIFIER = [
        'gdp_per_capita',    # Wealth proxy — raw scale dominates SHAP
        'nominal_gdp',       # Economic size — not a crisis predictor (Iceland, Cyprus disprove)
        'credit_to_gdp',     # Redundant with credit_to_gdp_gap (the BIS-validated measure)
        'development_tier',  # Ordinal income tier — used ONLY for sample weighting, not as feature
    ]
    
    # World Bank income tiers (GNI per capita thresholds, 2024 Atlas)
    # Used for ordinal development_tier feature and sample weighting
    DEVELOPMENT_TIERS = {
        'high': 14_005,        # High-income
        'upper_mid': 4_516,    # Upper-middle
        'lower_mid': 1_146,    # Lower-middle
        # Below lower_mid = Low-income
    }
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 5,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 use_smote: bool = False,
                 ensemble: bool = False):
        """
        Initialize crisis classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Max tree depth (keep low to avoid overfitting)
            learning_rate: Learning rate for boosting
            random_state: Random seed for reproducibility
            use_smote: Upgrade minority class (crisis) using SMOTE
            ensemble: Use VotingClassifier (XGB + LR + RF)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.use_smote = use_smote
        self.ensemble = ensemble
        
        self.model = None
        self.calibrated_model = None  # Isotonic-calibrated wrapper
        self.scaler = RobustScaler()
        self.feature_names_ = []
        self.feature_importance_ = {}
        self.fitted_ = False
        
        # Output directory for visualizations
        self.output_dir = os.path.join(CACHE_DIR, 'eda')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _create_model(self, n_positive: int, n_negative: int,
                      monotone_constraints: tuple = None):
        """Create classifier model (Single or Ensemble)."""
        scale_pos_weight = n_negative / max(n_positive, 1)
        
        # Base XGBoost Model
        if HAS_XGBOOST:
            params = dict(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
                # Regularization
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=5,
                reg_alpha=1.0,
                reg_lambda=5.0,
                gamma=1.0,
            )
            if monotone_constraints is not None:
                params['monotone_constraints'] = monotone_constraints
            base_model = xgb.XGBClassifier(**params)
        else:
            base_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight='balanced',
                random_state=self.random_state
            )
            
        if not self.ensemble:
            return base_model
            
        # Ensemble: XGBoost + Logistic Regression + Random Forest
        # Logic: XGB captures non-linear, LR captures linear baseline, RF adds diversity
        lr = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', C=0.1, random_state=self.random_state)
        rf = RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=self.random_state)
        
        # Soft voting averages probabilities
        voting_clf = VotingClassifier(
            estimators=[('xgb', base_model), ('lr', lr), ('rf', rf)],
            voting='soft',
            weights=[2, 1, 1]  # Weight XGBoost higher but let others correct it
        )
        return voting_clf
    
    def fit(self, X: pd.DataFrame, y: pd.Series, cv: int = 5,
            sample_weights: np.ndarray = None,
            monotone_constraints: tuple = None) -> 'CrisisClassifier':
        """
        Fit the crisis classifier with cross-validation and generate diagnostics.
        
        Args:
            X: Feature DataFrame
            y: Binary crisis target
            cv: Number of CV folds
            sample_weights: Per-sample weights for income-tier rebalancing
            monotone_constraints: Tuple of (-1, 0, +1) per feature for economic direction
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
        
        # Reset y to numpy array for safe integer indexing (multi-epoch concat can break iloc)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)
        
        n_positive = int(y_arr.sum())
        n_negative = len(y_arr) - n_positive
        
        # Prepare sample weights (default to uniform if not provided)
        if sample_weights is None:
            sw = np.ones(len(y_arr))
        else:
            sw = np.array(sample_weights) if not isinstance(sample_weights, np.ndarray) else sample_weights
        
        # --- CROSS-VALIDATION WITH ROC PLOTTING ---
        print(f"\n--- {cv}-Fold Stratified Cross-Validation & ROC Curves ---")
        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        plt.figure(figsize=(10, 8))
        
        min_class_count = min(n_positive, n_negative)
        actual_cv = min(cv, min_class_count) if min_class_count > 1 else 2
        
        skf = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_arr)):
            X_train_fold, y_train_fold = X_scaled[train_idx], y_arr[train_idx]
            X_val_fold, y_val_fold = X_scaled[val_idx], y_arr[val_idx]
            w_train_fold = sw[train_idx]
            
            # --- APPLY SMOTE (Training Fold Only) ---
            if self.use_smote and HAS_SMOTE:
                # Note: SMOTE doesn't support sample_weights generation easily, 
                # so we rely on the synthetic data generation to balance classes
                # and assume uniform weights for synthetic samples.
                smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y_train_fold)-1))
                X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)
                w_train_fold = np.ones(len(y_train_fold)) # Reset weights for balanced data
            
            # Create fresh model for this fold
            fold_model = self._create_model(
                n_positive=sum(y_train_fold), 
                n_negative=len(y_train_fold) - sum(y_train_fold),
                monotone_constraints=monotone_constraints
            )
            
            # Fit
            fold_model.fit(X_train_fold, y_train_fold, sample_weight=w_train_fold)
            
            # Evaluate
            try:
                probas_ = fold_model.predict_proba(X_val_fold)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_val_fold, probas_)
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                
                # Interp
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                
                plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold+1} (AUC = {roc_auc:.2f})')
            except Exception as e:
                print(f"  Fold {fold+1} failed: {e}")
                aucs.append(0.5)
        
        # Plot Mean ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} $\\pm$ {std_auc:.2f})', lw=2, alpha=0.8)
        
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
        
        # Train final model on all provided data (with weights/SMOTE)
        print("\n  Training final model on all data...")
        
        X_final, y_final, w_final = X_scaled, y_arr, sw
        if self.use_smote and HAS_SMOTE:
             print("  Applying SMOTE to full training set...")
             smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y_final)-1))
             X_final, y_final = smote.fit_resample(X_final, y_final)
             w_final = np.ones(len(y_final))

        # Re-create model for final fit
        self.model = self._create_model(
            n_positive=sum(y_final), 
            n_negative=len(y_final) - sum(y_final),
            monotone_constraints=monotone_constraints
        )
        self.model.fit(X_final, y_final, sample_weight=w_final)
        
        # Feature importance only available for base XGBoost (not voting classifier)
        if not self.ensemble and hasattr(self.model, 'feature_importances_'):
            self._compute_feature_importance(X_scaled, y)
        
        # --- PROBABILITY CALIBRATION (Isotonic Regression) ---
        # Raw XGBoost probabilities cluster near 0 due to class imbalance.
        # Isotonic calibration maps raw probs to observed frequencies.
        # Ref: Niculescu-Mizil & Caruana (2005)
        try:
            from sklearn.calibration import CalibratedClassifierCV
            print("\n  Calibrating probabilities (isotonic regression)...")
            self.calibrated_model = CalibratedClassifierCV(
                self.model, method='isotonic', cv=min(3, min_class_count)
            )
            self.calibrated_model.fit(X_scaled, y_arr, sample_weight=sw)
            
            # Report calibration improvement
            raw_probs = self.model.predict_proba(X_scaled)[:, 1]
            cal_probs = self.calibrated_model.predict_proba(X_scaled)[:, 1]
            base_rate = y_arr.mean()
            print(f"    Base rate: {base_rate:.1%}")
            print(f"    Raw prob mean: {raw_probs.mean():.1%} (should be ~{base_rate:.0%})")
            print(f"    Calibrated prob mean: {cal_probs.mean():.1%}")
            print(f"    Crisis group: raw={raw_probs[y_arr==1].mean():.1%} -> cal={cal_probs[y_arr==1].mean():.1%}")
            print(f"    Non-crisis group: raw={raw_probs[y_arr==0].mean():.1%} -> cal={cal_probs[y_arr==0].mean():.1%}")
        except Exception as e:
            print(f"  WARNING: Calibration failed ({e}), using raw probabilities")
            self.calibrated_model = None
        
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
    
    def predict_proba(self, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """
        Predict crisis probability.
        
        Args:
            X: Feature DataFrame
            calibrated: Use calibrated probabilities (default True)
        
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
        
        # Use calibrated model if available and requested
        if calibrated and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_scaled)[:, 1]
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary crisis outcome."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 train_auc: float = None) -> Dict[str, float]:
        """Evaluate model with overfitting check."""
        print("\n" + "="*70)
        print("MODEL EVALUATION (HOLDOUT)")
        print("="*70)
        
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        try:
            auc_roc = roc_auc_score(y, y_proba)
        except:
            auc_roc = 0.5
        
        # Classification metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        print(f"  AUC-ROC (Holdout): {auc_roc:.3f}")
        print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
        
        # --- OVERFITTING CHECK ---
        if train_auc is not None:
            gap = train_auc - auc_roc
            print(f"\n  Overfitting Check:")
            print(f"    Train AUC: {train_auc:.3f}")
            print(f"    Test AUC:  {auc_roc:.3f}")
            if gap > 0.15:
                print(f"    Gap: {gap:.3f}  WARNING: Significant overfitting (gap > 0.15)")
            elif gap > 0.08:
                print(f"    Gap: {gap:.3f}  Moderate overfitting (gap > 0.08)")
            else:
                print(f"    Gap: {gap:.3f}  Acceptable (gap <= 0.08)")
        
        # Calibration stats
        print(f"\n  Calibration:")
        print(f"    Mean predicted prob: {y_proba.mean():.1%}")
        print(f"    Actual crisis rate:  {y.mean():.1%}")
        print(f"    Crisis group mean prob:     {y_proba[y==1].mean():.1%}")
        print(f"    Non-crisis group mean prob: {y_proba[y==0].mean():.1%}")
        
        # Confusion Matrix Plot
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Crisis', 'Crisis'])
        
        plt.figure(figsize=(6, 6))
        disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
        plt.title('Confusion Matrix (Holdout Test Set)')
        
        cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved Confusion Matrix: {cm_path}")
        
        return {
            'auc_roc': auc_roc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
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
        Score = 0.4 * Economic_Pillar (PCA-based)
              + 0.4 * Industry_Pillar (PCA-based, includes liquidity)
              + 0.2 * Supervised_Crisis_Probability
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

def _assign_development_tier(gdp_per_capita):
    """
    Convert GDP per capita to ordinal development tier.
    Replaces continuous wealth signal with categorical income group.
    
    Ref: World Bank Atlas method income classifications (2024).
    """
    if pd.isna(gdp_per_capita) or gdp_per_capita <= 0:
        return 2  # Default to lower-middle if unknown
    elif gdp_per_capita >= CrisisClassifier.DEVELOPMENT_TIERS['high']:
        return 4  # High income
    elif gdp_per_capita >= CrisisClassifier.DEVELOPMENT_TIERS['upper_mid']:
        return 3  # Upper-middle
    elif gdp_per_capita >= CrisisClassifier.DEVELOPMENT_TIERS['lower_mid']:
        return 2  # Lower-middle
    else:
        return 1  # Low income


def _compute_income_tier_weights(features_df, y):
    """
    Compute sample weights to correct for reporting bias by income tier.
    
    Richer countries have better-documented crises in Laeven-Valencia (2018).
    This re-weights so each income tier contributes equally to the loss.
    
    Ref: King & Zeng (2001) - "Logistic Regression in Rare Events Data"
    """
    if 'development_tier' not in features_df.columns:
        return np.ones(len(y))
    
    tiers = features_df['development_tier']
    weights = np.ones(len(y))
    
    tier_counts = tiers.value_counts()
    n_tiers = len(tier_counts)
    target_per_tier = len(y) / n_tiers
    
    for tier_val, count in tier_counts.items():
        tier_weight = target_per_tier / count
        weights[tiers == tier_val] = tier_weight
    
    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()
    
    print(f"  Income-tier sample weights:")
    for tier_val in sorted(tier_counts.index):
        tier_name = {1: 'Low', 2: 'Lower-Mid', 3: 'Upper-Mid', 4: 'High'}.get(tier_val, '?')
        mask = tiers == tier_val
        n_crisis = int(y[mask].sum())
        print(f"    Tier {tier_val} ({tier_name}): {tier_counts[tier_val]} countries, "
              f"{n_crisis} crises, weight={weights[mask].mean():.2f}")
    
    return weights


# ==============================================================================
# TEMPORAL PANEL DATA EXTRACTION
# ==============================================================================
# Builds year-matched features for each crisis epoch so the model trains on
# features that PRECEDED each crisis, not a latest snapshot.
# Ref: Drehmann & Juselius (2014), Borio & Lowe (2002)

def _extract_weo_at_year(weo_df, target_year, countries):
    """
    Extract WEO features at a specific year (or nearest prior year).
    
    Args:
        weo_df: Raw WEO DataFrame with country_code, indicator_code, period, value
        target_year: Year to extract features for (e.g. 2004 for 2005 epoch)
        countries: List of country codes to extract for
    
    Returns:
        DataFrame with one row per country, columns = WEO features
    """
    weo_mappings = {
        'gdp_growth': 'NGDP_RPCH',
        'inflation': 'PCPIPCH',
        'current_account_gdp': 'BCA_NGDPD',
        'govt_debt_gdp': 'GGXWDG_NGDP',
        'fiscal_balance_gdp': 'GGXCNL_NGDP',
        'primary_balance_gdp': 'GGXONLB_NGDP',
        'unemployment': 'LUR',
        'gdp_per_capita': 'NGDPDPC',
        'external_debt_gdp': 'D_NGDPD',
    }
    
    df = weo_df.copy()
    df['year'] = pd.to_datetime(df['period']).dt.year
    # Use data up to target_year (no future leakage)
    df = df[df['year'] <= target_year]
    
    # For each country-indicator, get the latest value <= target_year
    latest = df.groupby(['country_code', 'indicator_code']).agg(
        value=('value', 'last'),
        year=('year', 'last')
    ).reset_index()
    
    features_list = []
    for feature_name, code in weo_mappings.items():
        code_data = latest[latest['indicator_code'] == code][['country_code', 'value']]
        code_data = code_data.rename(columns={'value': feature_name})
        features_list.append(code_data.set_index('country_code'))
    
    if features_list:
        result = features_list[0]
        for df_part in features_list[1:]:
            result = result.join(df_part, how='outer')
        result = result.reset_index()
    else:
        result = pd.DataFrame(columns=['country_code'])
    
    # Filter to requested countries
    result = result[result['country_code'].isin(countries)]
    return result


def _extract_fsic_at_year(fsic_df, target_year, countries):
    """
    Extract FSIC features at a specific year (or nearest prior year).
    
    Args:
        fsic_df: Raw FSIC DataFrame with country_code, indicator_name, period, value
        target_year: Year to extract features for
        countries: List of country codes to extract for
    
    Returns:
        DataFrame with one row per country, columns = FSIC features
    """
    fsic_mappings = {
        'capital_adequacy': 'Regulatory capital to risk-weighted assets.*Core FSI',
        'npl_ratio': 'Nonperforming loans to total gross loans.*Core FSI',
        'roe': 'Return on equity.*Core FSI',
        'liquid_assets_total': 'Liquid assets to total assets.*Percent',
        'npl_provisions': 'Provisions to nonperforming loans.*Percent',
        'real_estate_loans': 'Residential real estate loans to total gross loans.*Core FSI',
    }
    
    df = fsic_df.copy()
    df['year'] = pd.to_datetime(df['period']).dt.year
    df = df[df['year'] <= target_year]
    df = df[df['country_code'].isin(countries)]
    
    features_list = []
    for country in countries:
        country_data = df[df['country_code'] == country]
        row = {'country_code': country}
        
        for feature_name, pattern in fsic_mappings.items():
            mask = country_data['indicator_name'].str.contains(
                pattern, case=False, na=False, regex=True
            )
            if mask.any():
                matched = country_data[mask].sort_values('period')
                if len(matched) > 0:
                    row[feature_name] = matched['value'].iloc[-1]
        
        if len(row) > 1:  # Has at least one feature beyond country_code
            features_list.append(row)
    
    return pd.DataFrame(features_list) if features_list else pd.DataFrame(columns=['country_code'])


def _compute_lag_features(weo_df, target_year, countries):
    """
    Compute lag/change features that capture building imbalances.
    
    Imbalances build with lags — credit growth over 3 years, debt acceleration,
    current account deterioration are strong EWIs.
    
    Ref: Drehmann & Juselius (2014), Schularick & Taylor (2012)
    
    Returns:
        DataFrame with lag features per country
    """
    df = weo_df.copy()
    df['year'] = pd.to_datetime(df['period']).dt.year
    df = df[df['country_code'].isin(countries)]
    
    lag_features = []
    
    for country in countries:
        c_data = df[df['country_code'] == country]
        row = {'country_code': country}
        
        # GDP growth: 3-year average (smoothed cyclical position)
        gdp_data = c_data[c_data['indicator_code'] == 'NGDP_RPCH']
        gdp_data = gdp_data[(gdp_data['year'] >= target_year - 3) & 
                            (gdp_data['year'] <= target_year)]
        if len(gdp_data) >= 2:
            row['gdp_growth_3yr_avg'] = gdp_data['value'].mean()
        
        # Inflation acceleration (change in inflation over 3 years)
        infl_data = c_data[c_data['indicator_code'] == 'PCPIPCH']
        infl_t = infl_data[infl_data['year'] == target_year]['value']
        infl_lag = infl_data[infl_data['year'] == target_year - 3]['value']
        if len(infl_t) > 0 and len(infl_lag) > 0:
            row['inflation_acceleration'] = infl_t.iloc[0] - infl_lag.iloc[0]
        
        # Debt buildup: change in govt debt/GDP over 3 years
        debt_data = c_data[c_data['indicator_code'] == 'GGXWDG_NGDP']
        debt_t = debt_data[debt_data['year'] == target_year]['value']
        debt_lag = debt_data[debt_data['year'] == target_year - 3]['value']
        if len(debt_t) > 0 and len(debt_lag) > 0:
            row['debt_buildup_3yr'] = debt_t.iloc[0] - debt_lag.iloc[0]
        
        # Current account deterioration (negative change = worsening)
        ca_data = c_data[c_data['indicator_code'] == 'BCA_NGDPD']
        ca_t = ca_data[ca_data['year'] == target_year]['value']
        ca_lag = ca_data[ca_data['year'] == target_year - 3]['value']
        if len(ca_t) > 0 and len(ca_lag) > 0:
            row['ca_deterioration_3yr'] = ca_t.iloc[0] - ca_lag.iloc[0]
        
        if len(row) > 1:
            lag_features.append(row)
    
    return pd.DataFrame(lag_features) if lag_features else pd.DataFrame(columns=['country_code'])


# ==============================================================================
# MONOTONE CONSTRAINTS MAP
# ==============================================================================
# Forces XGBoost to learn economically correct feature directions.
# +1 = higher value -> MORE crisis risk
# -1 = higher value -> LESS crisis risk  
#  0 = unconstrained (let model learn)
#
# Ref: BIS (2018), Demirguc-Kunt & Detragiache (1998)

MONOTONE_DIRECTION = {
    # Macro vulnerability (+1 = higher -> more crisis)
    'npl_ratio': 1,              # Bad loans = stress
    'govt_debt_gdp': 1,          # Debt pressure
    'unemployment': 1,           # Economic stress
    'inflation': 1,              # Macro instability
    'real_estate_loans': 1,      # Property concentration risk
    'sovereign_exposure_ratio': 1,  # Sovereign-bank nexus
    'external_debt_gdp': 1,      # External vulnerability
    'debt_buildup_3yr': 1,       # Rapid debt accumulation
    'inflation_acceleration': 1, # Accelerating prices
    'large_exposure_ratio': 1,   # Concentration risk
    
    # Resilience (-1 = higher -> less crisis)
    'capital_adequacy': -1,      # Strong buffers
    'gdp_growth': -1,            # Growing economy
    'current_account_gdp': -1,   # Surplus = strong external position
    'fiscal_balance_gdp': -1,    # Fiscal surplus
    'fiscal_space': -1,          # Fiscal headroom
    'roe': -1,                   # Profitable banks
    'liquid_assets_total': -1,   # Liquidity buffer
    'liquid_assets_st_liab': -1, # Liquidity coverage
    'npl_provisions': -1,        # Loss absorption
    'capital_quality': -1,       # Equity-heavy capital
    'deposit_funding_ratio': -1, # Stable funding (renamed from deposit_to_total_assets)
    'voice_accountability': -1,  # Institutional quality
    'political_stability': -1,   # Political resilience
    'gdp_growth_3yr_avg': -1,    # Sustained growth
    'ca_deterioration_3yr': -1,  # Positive = improvement -> less crisis
    'income_diversification': -1, # Diversified revenue
    'deposit_to_total_assets': -1, # Stable funding
    'customer_deposits_loans': -1, # Deposit-funded loans
    
    # Ambiguous (0 = let model learn)
    'credit_to_gdp_gap': 0,      # BIS: both high and rapid change matter
    'bank_liability_to_nfa': 0,  # Complex FX dynamics
    'sovereign_liability_to_reserves': 0,  # Context-dependent
    'securities_to_assets': 0,   # Can indicate market sophistication or risk
    'fx_loan_exposure': 0,       # Depends on hedging
    'loan_concentration': 0,     # Context-dependent
    'debt_service_gdp': 0,       # Ambiguous direction
    'roa': 0,                    # Can be high due to high risk-taking
    
    # Literature gap features
    'inflation_differential_3yr': 1,  # Higher diff vs G7 -> REER misalignment -> crisis
    'interest_cost_gdp': 1,           # Higher interest burden -> fiscal stress -> crisis
    'interest_cost_trend_3yr': 1,     # Rising cost -> deterioration -> crisis
    'credit_growth_3yr': 1,           # Credit boom -> crisis (Schularick & Taylor 2012)
    'm2_to_reserves': 1,              # Higher ratio -> FX vulnerability -> crisis
    'ca_deficit_severity': 1,         # Deeper deficit -> external stress -> crisis
    'real_estate_credit_growth_3yr': 1, # RE boom -> bubble -> crisis
    'tot_deterioration_3yr': 1,       # Worsening terms of trade -> external stress
    'primary_balance_gdp': 0,         # Ambiguous: surplus could be austerity or strength
}


def _build_monotone_constraints(feature_cols):
    """
    Build monotone constraint tuple for XGBoost based on feature column order.
    
    Returns tuple of (-1, 0, +1) matching the column order of the training data.
    """
    constraints = []
    for col in feature_cols:
        constraints.append(MONOTONE_DIRECTION.get(col, 0))
    
    constrained = sum(1 for c in constraints if c != 0)
    print(f"\n--- Monotone Constraints ---")
    print(f"  {constrained}/{len(constraints)} features constrained")
    print(f"  Positive (higher->crisis): {sum(1 for c in constraints if c > 0)}")
    print(f"  Negative (higher->safe):   {sum(1 for c in constraints if c < 0)}")
    print(f"  Unconstrained:            {sum(1 for c in constraints if c == 0)}")
    
    return tuple(constraints)


def train_crisis_model(weo_df=None, fsic_df=None):
    """
    Train the crisis classifier with temporal panel data.
    
    CRISP-DM: Full modeling pipeline
    
    KEY FIX: Uses year-matched features for each crisis epoch instead of a 
    single latest snapshot. This means Nigeria's 1989 macro data is used to
    predict its 1991 crisis, not Nigeria's 2025 data.
    
    Also includes lag features to capture building imbalances:
    - 3-year GDP growth average (cyclical position)
    - Inflation acceleration (price dynamics)
    - Debt buildup over 3 years (fiscal trajectory)
    - Current account deterioration (external vulnerability trend)
    
    Academic references:
    - Drehmann & Juselius (2014): credit-to-GDP gap as best EWI
    - Laeven & Valencia (2018): systemic banking crisis database
    - Borio & Lowe (2002): ratio-based indicators outperform levels
    - Schularick & Taylor (2012): credit growth predicts financial crises
    """
    from src.crisis_labels import CrisisLabels
    
    print("="*70)
    print("TEMPORAL PANEL CRISIS CLASSIFIER")
    print("CRISP-DM Phases: Data Preparation -> Modeling -> Evaluation")
    print("="*70)
    
    # Load latest features (for deployment predictions)
    features_path = os.path.join(CACHE_DIR, 'crisis_features.parquet')
    if not os.path.exists(features_path):
        print(f"ERROR: Features not found at {features_path}")
        return None
    
    features = pd.read_parquet(features_path)
    print(f"\nLoaded latest features: {len(features)} countries, {len(features.columns)} columns")
    
    # Add development tier for sample weighting
    if 'gdp_per_capita' in features.columns:
        features['development_tier'] = features['gdp_per_capita'].apply(_assign_development_tier)
    
    # Add fiscal space
    if 'fiscal_balance_gdp' in features.columns and 'govt_debt_gdp' in features.columns:
        features['fiscal_space'] = (
            features['fiscal_balance_gdp'] - features['govt_debt_gdp'] / 100
        )
    
    labels = CrisisLabels()
    all_countries = features['country_code'].tolist()
    
    # --- BUILD TEMPORAL PANEL ---
    # Each epoch has year-matched features from WEO/FSIC + lag features
    EPOCHS = {
        1990: ('Africa (KEN,NGA,TZA), Nordic (FIN,SWE,NOR)', 1989),
        1995: ('Asian crisis (THA,IDN,KOR,MYS), LatAm (ARG,MEX)', 1994),
        2000: ('Argentina (2001), Turkey (2000)', 1999),
        2005: ('Global Financial Crisis (USA,GBR,ESP,ISL,GRC...)', 2004),
        2015: ('Turkey (2018), Ghana (2017-2018), Lebanon (2019)', 2014),
    }
    
    use_panel = (weo_df is not None)  # Can we build temporal panel?
    
    if use_panel:
        print("\n--- Building Temporal Panel Dataset ---")
        print("  (Year-matched features for each crisis epoch)")
        
        epoch_datasets = []
        for ref_year, (description, feature_year) in EPOCHS.items():
            # Extract features AT the epoch's feature year
            weo_at_year = _extract_weo_at_year(weo_df, feature_year, all_countries)
            
            # Extract FSIC if available
            fsic_at_year = pd.DataFrame(columns=['country_code'])
            if fsic_df is not None:
                fsic_at_year = _extract_fsic_at_year(fsic_df, feature_year, all_countries)
            
            # Compute lag features
            lag_feats = _compute_lag_features(weo_df, feature_year, all_countries)
            
            # Merge WEO + FSIC + lags
            epoch_features = weo_at_year
            if len(fsic_at_year) > 0:
                epoch_features = epoch_features.merge(
                    fsic_at_year, on='country_code', how='left'
                )
            if len(lag_feats) > 0:
                epoch_features = epoch_features.merge(
                    lag_feats, on='country_code', how='left'
                )
            
            # Add fiscal_space if components available
            if 'fiscal_balance_gdp' in epoch_features.columns and 'govt_debt_gdp' in epoch_features.columns:
                epoch_features['fiscal_space'] = (
                    epoch_features['fiscal_balance_gdp'] - epoch_features['govt_debt_gdp'] / 100
                )
            
            # Add development tier from latest data (structural, changes slowly)
            if 'gdp_per_capita' in epoch_features.columns:
                epoch_features['development_tier'] = epoch_features['gdp_per_capita'].apply(
                    _assign_development_tier
                )
            elif 'development_tier' in features.columns:
                tier_map = features.set_index('country_code')['development_tier']
                epoch_features['development_tier'] = epoch_features['country_code'].map(tier_map)
            
            # Add crisis labels
            epoch_features['crisis_target'] = epoch_features['country_code'].apply(
                lambda c, yr=ref_year: labels.get_crisis_target(c, yr, horizon=3)
            )
            epoch_features['epoch'] = ref_year
            
            n_crisis = int(epoch_features['crisis_target'].sum())
            n_countries = len(epoch_features)
            print(f"  Epoch {ref_year} (features from {feature_year}): "
                  f"{n_crisis} crises / {n_countries} countries — {description}")
            
            epoch_datasets.append(epoch_features)
        
        # Concatenate all epochs into panel
        panel_df = pd.concat(epoch_datasets, ignore_index=True)
        
    else:
        # Fallback: latest-snapshot (no raw data provided)
        print("\n--- Multi-Epoch Labels (latest snapshot fallback) ---")
        print("  WARNING: Raw WEO/FSIC not provided — using latest features for all epochs")
        print("  This reduces model quality. Pass weo_df/fsic_df for temporal panel.")
        
        epoch_datasets = []
        for ref_year, (description, _) in EPOCHS.items():
            epoch_labels = features['country_code'].apply(
                lambda c, yr=ref_year: labels.get_crisis_target(c, yr, horizon=3)
            )
            epoch_df = features.copy()
            epoch_df['crisis_target'] = epoch_labels
            epoch_df['epoch'] = ref_year
            n_crisis = int(epoch_labels.sum())
            print(f"  Epoch {ref_year}: {n_crisis} crises / {len(epoch_labels)} countries")
            epoch_datasets.append(epoch_df)
        
        panel_df = pd.concat(epoch_datasets, ignore_index=True)
    
    # De-duplicate non-crisis observations (keep crisis rows from all epochs)
    crisis_rows = panel_df[panel_df['crisis_target'] == 1]
    non_crisis_rows = panel_df[panel_df['crisis_target'] == 0].drop_duplicates(
        subset='country_code', keep='last'
    )
    training_df = pd.concat([crisis_rows, non_crisis_rows], ignore_index=True)
    
    print(f"\n  Panel: {len(training_df)} observations "
          f"(Crisis: {int(training_df['crisis_target'].sum())} | "
          f"Non-crisis: {int((~training_df['crisis_target'].astype(bool)).sum())})")
    
    # --- FEATURE SELECTION ---
    print("\n--- Feature Selection ---")
    
    feature_cols = [c for c in training_df.columns 
                   if c not in ['country_code', 'crisis_target', 'epoch', 'country_name'] 
                   and not c.endswith('_period')
                   and not c.endswith('_year')
                   and c not in CrisisClassifier.EXCLUDED_FROM_CLASSIFIER]
    
    excluded = [c for c in CrisisClassifier.EXCLUDED_FROM_CLASSIFIER 
                if c in training_df.columns]
    print(f"  Using {len(feature_cols)} features (excluded: {excluded})")
    for col in feature_cols:
        direction = MONOTONE_DIRECTION.get(col, 0)
        dir_str = {1: '+crisis', -1: '+safe', 0: 'free'}[direction]
        print(f"    {col}: {dir_str}")
    
    X = training_df[feature_cols]
    y = training_df['crisis_target']
    
    # --- MONOTONE CONSTRAINTS ---
    mc = _build_monotone_constraints(feature_cols)
    
    # --- INCOME-TIER SAMPLE WEIGHTS ---
    print("\n--- Income-Tier Sample Weights ---")
    sample_weights = _compute_income_tier_weights(training_df, y)
    
    # --- TRAIN-TEST SPLIT (80/20, stratified) ---
    print("\n--- Train-Test Split (80/20 stratified) ---")
    
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights, 
        test_size=0.20, 
        random_state=42, 
        stratify=y
    )
    
    print(f"  Train: {len(X_train)} ({y_train.sum():.0f} crises)")
    print(f"  Test:  {len(X_test)} ({y_test.sum():.0f} crises)")
    
    # --- TRAIN CLASSIFIER ---
    classifier = CrisisClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
        use_smote=True,
        ensemble=True
    )
    
    classifier.fit(X_train, y_train, cv=5, sample_weights=w_train,
                   monotone_constraints=mc)
    
    # Compute train AUC for overfitting check
    train_proba = classifier.predict_proba(X_train)
    train_auc = roc_auc_score(y_train, train_proba)
    
    # --- HOLDOUT EVALUATION (with overfitting check) ---
    print("\n" + "="*70)
    print("HOLDOUT TEST SET EVALUATION")
    print("="*70)
    metrics = classifier.evaluate(X_test, y_test, train_auc=train_auc)
    
    # --- FULL DATASET RE-FIT ---
    print("\n  Re-fitting on full dataset for deployment...")
    classifier_full = CrisisClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05
    )
    classifier_full.fit(X, y, cv=5, sample_weights=sample_weights,
                        monotone_constraints=mc)
    
    # --- FORWARD-LOOKING RISK ASSESSMENT (2026-2028) ---
    # NOTE: This uses CURRENT (2025) features to predict crisis in next 3 years.
    # These are NOT retroactive predictions of past crises.
    print("\n" + "="*70)
    print("FORWARD-LOOKING CRISIS RISK (2026-2028)")
    print("(Based on latest available data, calibrated probabilities)")
    print("="*70)
    
    # Predict on latest features for deployment
    deploy_feature_cols = [c for c in feature_cols if c in features.columns]
    # Add lag features from latest WEO data if available
    if weo_df is not None and use_panel:
        from datetime import datetime
        latest_year = datetime.now().year - 1
        latest_lags = _compute_lag_features(weo_df, latest_year, all_countries)
        if len(latest_lags) > 0:
            features = features.merge(latest_lags, on='country_code', how='left')
            for lag_col in ['gdp_growth_3yr_avg', 'inflation_acceleration', 
                           'debt_buildup_3yr', 'ca_deterioration_3yr']:
                if lag_col in features.columns and lag_col not in deploy_feature_cols:
                    deploy_feature_cols.append(lag_col)
    
    orig_X = features[deploy_feature_cols].copy()
    # Fill missing columns with NaN (classifier will handle via median imputation)
    for col in feature_cols:
        if col not in orig_X.columns:
            orig_X[col] = np.nan
    orig_X = orig_X[feature_cols]  # Ensure correct column order
    
    orig_probs = classifier_full.predict_proba(orig_X)
    features['crisis_prob_debiased'] = orig_probs
    
    test_cases = {
        'USA': ('GFC 2007-09', 'Core epicenter, subprime crisis'),
        'GBR': ('GFC 2007-09', 'Northern Rock, RBS nationalization'),
        'ISL': ('GFC 2008-10', 'Total banking collapse, 3 largest banks failed'),
        'ESP': ('GFC 2008-12', 'Savings bank (cajas) crisis, property bubble'),
        'IRL': ('GFC 2008-11', 'Property bubble, Anglo Irish Bank'),
        'THA': ('Asian 1997-00', 'Trigger country, baht collapse'),
        'IDN': ('Asian 1997-01', 'Deepest impact, bank runs'),
        'KOR': ('Asian 1997-98', 'IMF program, chaebol restructuring'),
        'KEN': ('Africa 1992-95', 'Political banking, ethnic tensions'),
        'NGA': ('Africa 1991-95', 'SAP aftermath, bank distress'),
        'GHA': ('Multi 1997,2017,2022', 'Repeated banking + sovereign stress'),
        'TUR': ('2018-19', 'Lira crisis, policy credibility'),
        'LBN': ('2019-24', 'Worst financial crisis in 150 years (World Bank)'),
        'CHE': ('Control', 'No systemic crisis in database'),
        'CAN': ('Control', 'No systemic crisis, strong regulation'),
        'AUS': ('Control', 'Survived GFC without systemic crisis'),
    }
    
    print(f"\n  {'Country':<6} {'P(Crisis)':<10} {'Tier':<9} {'Past Crisis':<20} {'Notes'}")
    print("  " + "-"*80)
    
    for cc, (period, notes) in test_cases.items():
        row = features[features['country_code'] == cc]
        if len(row) > 0:
            prob = row['crisis_prob_debiased'].iloc[0]
            tier = int(row['development_tier'].iloc[0]) if 'development_tier' in row.columns else '?'
            tier_name = {1: 'Low', 2: 'LMid', 3: 'UMid', 4: 'High'}.get(tier, '?')
            print(f"  {cc:<6} {prob:<10.1%} T{tier}({tier_name:<4}) {period:<20} {notes}")
    
    print("\n  NOTE: These are FORWARD-LOOKING probabilities (2026-2028 risk)")
    print("  based on each country's CURRENT macro/banking fundamentals.")
    print("  They do NOT retroactively predict past crises listed above.")
    
    # --- BACKTESTED EPOCH ACCURACY ---
    # How well does the model identify crisis countries in each epoch?
    print("\n" + "="*70)
    print("BACKTESTED EPOCH ACCURACY")
    print("(How well does the model flag crisis countries per epoch?)")
    print("="*70)
    
    total_crisis_flagged = 0
    total_crisis_countries = 0
    total_non_crisis_flagged = 0
    total_non_crisis_countries = 0
    
    for epoch_df in epoch_datasets:
        epoch_year = epoch_df['epoch'].iloc[0]
        epoch_feat_cols = [c for c in feature_cols if c in epoch_df.columns]
        
        epoch_X = epoch_df[epoch_feat_cols].copy()
        for col in feature_cols:
            if col not in epoch_X.columns:
                epoch_X[col] = np.nan
        epoch_X = epoch_X[feature_cols]
        
        epoch_probs = classifier_full.predict_proba(epoch_X)
        epoch_df_local = epoch_df.copy()
        epoch_df_local['prob'] = epoch_probs
        
        crisis_rows = epoch_df_local[epoch_df_local['crisis_target'] == 1]
        non_crisis_rows = epoch_df_local[epoch_df_local['crisis_target'] == 0]
        
        # Flag = probability > 30%
        threshold = 0.30
        n_crisis = len(crisis_rows)
        n_flagged = (crisis_rows['prob'] >= threshold).sum() if n_crisis > 0 else 0
        recall = n_flagged / max(n_crisis, 1)
        
        n_non_crisis = len(non_crisis_rows)
        n_false_alarm = (non_crisis_rows['prob'] >= threshold).sum()
        fpr = n_false_alarm / max(n_non_crisis, 1)
        
        total_crisis_flagged += n_flagged
        total_crisis_countries += n_crisis
        total_non_crisis_flagged += n_false_alarm
        total_non_crisis_countries += n_non_crisis
        
        # Show flagged crisis countries
        if n_crisis > 0:
            flagged_names = crisis_rows[crisis_rows['prob'] >= threshold]['country_code'].tolist()
            missed_names = crisis_rows[crisis_rows['prob'] < threshold]['country_code'].tolist()
            print(f"\n  Epoch {epoch_year}: Recall={recall:.0%} ({n_flagged}/{n_crisis} crises), FPR={fpr:.0%}")
            if flagged_names:
                print(f"    Flagged:  {', '.join(flagged_names)}")
            if missed_names:
                print(f"    Missed:   {', '.join(missed_names)}")
    
    overall_recall = total_crisis_flagged / max(total_crisis_countries, 1)
    overall_fpr = total_non_crisis_flagged / max(total_non_crisis_countries, 1)
    print(f"\n  Overall: Recall={overall_recall:.0%} ({total_crisis_flagged}/{total_crisis_countries}), "
          f"FPR={overall_fpr:.0%} ({total_non_crisis_flagged}/{total_non_crisis_countries})")
    
    # Save
    classifier_full.save()
    metrics['test_cases'] = test_cases
    return classifier_full, metrics


if __name__ == "__main__":
    classifier, metrics = train_crisis_model()

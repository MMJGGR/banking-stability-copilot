"""Persisted transform pipeline for comparable banking-risk pillar scores."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


ECONOMIC_FEATURES = [
    "gdp_growth", "inflation", "current_account_gdp", "gdp_per_capita",
    "govt_debt_gdp", "fiscal_balance_gdp", "unemployment",
    "credit_to_gdp", "credit_to_gdp_relative", "debt_service_gdp",
    "external_debt_gdp", "sovereign_liability_to_reserves",
    "inflation_differential_3yr", "interest_cost_gdp",
    "interest_cost_trend_3yr", "credit_growth_3yr", "m2_to_reserves",
    "ca_deficit_severity", "tot_deterioration_3yr",
    "voice_accountability", "political_stability", "govt_effectiveness",
    # Staged liquidity challenger features (2026-07-11). Curated to genuinely
    # new, data-backed signals that do not duplicate existing pillar features:
    # sovereign fiscal affordability (interest/revenue, debt/revenue) from WEO
    # general government, and external-liquidity stocks/flows from IMF BOP/IIP.
    # Absent columns are ignored by the pipeline, so production runs that omit
    # these features are unaffected.
    "govt_interest_to_revenue", "govt_debt_to_revenue",
    "net_iip_gdp", "external_liabilities_gdp",
    "reserves_to_goods_services_imports",
    "gross_external_financing_need_proxy_gdp",
    "investment_income_debits_to_cxr",
    "govt_revenue_gdp",
    "govt_primary_deficit_gdp",
    "govt_interest_to_revenue_change_3y",
    "govt_debt_to_revenue_change_3y",
    "govt_primary_deficit_gdp_change_3y",
    "govt_revenue_gdp_change_3y",
    "reserves_to_current_account_payments",
    "portfolio_liabilities_gdp",
    "commodity_export_share_pct",
    "wb_total_external_debt_service_gni_pct",
    "wb_ppg_external_debt_service_gdp",
    "wb_public_financing_need_ext_debt_service_proxy_gdp",
]

# Challenger v2 (2026-07-10): the level-share features `real_estate_loans`
# and `fx_loan_exposure` are excluded from the pillar. As unconditional
# levels they penalize mortgage-deep advanced economies and international
# financial centers rather than measuring boom risk; the growth-based
# `real_estate_credit_growth_3yr` stays, and both levels remain available to
# the supervised classifier and the Data Explorer. `years_since_banking_crisis`
# adds crisis-history memory so freshly recapitalized post-crisis balance
# sheets do not read as pristine.
INDUSTRY_FEATURES = [
    "capital_adequacy", "npl_ratio", "roe", "roa",
    "liquid_assets_st_liab", "liquid_assets_total",
    "customer_deposits_loans", "tier1_capital",
    "npl_provisions", "loan_concentration",
    "sovereign_exposure_ratio", "bank_liability_to_nfa",
    "real_estate_credit_growth_3yr", "regulatory_quality",
    "rule_of_law", "control_corruption",
    "years_since_banking_crisis",
]

# Expected credit-risk direction per feature on the transformed scale used for
# PCA: +1 means a higher value indicates higher banking-system risk, -1 means
# a higher value indicates lower risk. Every pillar feature must be declared
# here so a new feature cannot silently enter the score with an unreviewed
# direction. Note two transform-scale flips: `loan_concentration` is stored as
# -log1p(|raw|), so a higher transformed value means LESS concentration.
FEATURE_RISK_DIRECTIONS = {
    # Economic pillar
    "gdp_growth": -1.0,
    "inflation": 1.0,
    "current_account_gdp": -1.0,
    "gdp_per_capita": -1.0,
    "govt_debt_gdp": 1.0,
    "fiscal_balance_gdp": -1.0,
    "unemployment": 1.0,
    "credit_to_gdp": 1.0,
    "credit_to_gdp_relative": 1.0,
    "debt_service_gdp": 1.0,
    "external_debt_gdp": 1.0,
    "sovereign_liability_to_reserves": 1.0,
    "inflation_differential_3yr": 1.0,
    "interest_cost_gdp": 1.0,
    "interest_cost_trend_3yr": 1.0,
    "credit_growth_3yr": 1.0,
    "m2_to_reserves": 1.0,
    "ca_deficit_severity": 1.0,
    "tot_deterioration_3yr": 1.0,
    "voice_accountability": -1.0,
    "political_stability": -1.0,
    "govt_effectiveness": -1.0,
    # Staged liquidity challenger features (economic pillar). +1 = higher value
    # means higher banking-system/sovereign risk.
    "govt_interest_to_revenue": 1.0,   # interest bill vs revenue capacity
    "govt_debt_to_revenue": 1.0,       # debt stock vs revenue capacity
    "net_iip_gdp": -1.0,               # higher net creditor position = safer
    "external_liabilities_gdp": 1.0,   # larger external liabilities = riskier
    "reserves_to_goods_services_imports": -1.0,  # more import cover = safer
    "gross_external_financing_need_proxy_gdp": 1.0,
    "investment_income_debits_to_cxr": 1.0,  # external income-service burden
    "govt_revenue_gdp": -1.0,  # stronger revenue base = more fiscal capacity
    "govt_primary_deficit_gdp": 1.0,
    "govt_interest_to_revenue_change_3y": 1.0,
    "govt_debt_to_revenue_change_3y": 1.0,
    "govt_primary_deficit_gdp_change_3y": 1.0,
    "govt_revenue_gdp_change_3y": -1.0,
    "reserves_to_current_account_payments": -1.0,
    "portfolio_liabilities_gdp": 1.0,
    "commodity_export_share_pct": 1.0,
    "wb_total_external_debt_service_gni_pct": 1.0,
    "wb_ppg_external_debt_service_gdp": 1.0,
    "wb_public_financing_need_ext_debt_service_proxy_gdp": 1.0,
    # Industry pillar
    "capital_adequacy": -1.0,
    "npl_ratio": 1.0,
    "roe": -1.0,
    "roa": -1.0,
    "liquid_assets_st_liab": -1.0,
    "liquid_assets_total": -1.0,
    "customer_deposits_loans": -1.0,
    "fx_loan_exposure": 1.0,
    "tier1_capital": -1.0,
    "npl_provisions": -1.0,
    "loan_concentration": -1.0,
    "real_estate_loans": 1.0,
    "sovereign_exposure_ratio": 1.0,
    "bank_liability_to_nfa": 1.0,
    "real_estate_credit_growth_3yr": 1.0,
    "regulatory_quality": -1.0,
    "rule_of_law": -1.0,
    "control_corruption": -1.0,
    "years_since_banking_crisis": -1.0,
}

# Core banking soundness fields where a KNN-imputed value must not be allowed
# to make a sparse country look better than an observed peer. When these are
# missing for a country, a bounded risk penalty is added instead of trusting
# the imputation alone.
CRITICAL_FEATURES = [
    "npl_ratio",
    "capital_adequacy",
    "liquid_assets_st_liab",
    "credit_to_gdp_relative",
    "loan_concentration",
    "real_estate_loans",
    "fx_loan_exposure",
    "sovereign_exposure_ratio",
]


class ConstrainedRiskComponent:
    """First principal direction constrained to the declared risk directions.

    Inputs must already be oriented so that higher values mean higher risk.
    The first PCA component is flipped to point toward risk, negative loadings
    are clipped to zero (a negative loading would let a feature move the score
    in an economically counterintuitive direction), and the clipped loadings
    are shrunk halfway toward equal weights. The shrinkage prevents the
    component from collapsing onto one or two high-variance features after
    clipping: every declared feature keeps a strictly positive, monotone
    influence on the pillar score.
    """

    equal_weight_shrinkage = 0.5

    def fit(self, frame: pd.DataFrame):
        pca = PCA(n_components=1).fit(frame)
        weights = pca.components_[0].copy()
        if weights.sum() < 0:
            weights = -weights
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        uniform = np.full(frame.shape[1], 1.0 / frame.shape[1])
        if total == 0:
            weights = uniform
        else:
            shrinkage = self.equal_weight_shrinkage
            weights = (
                (1 - shrinkage) * (weights / total) + shrinkage * uniform
            )
        weights = weights / np.linalg.norm(weights)
        self.mean_ = frame.mean().to_numpy()
        self.components_ = weights.reshape(1, -1)
        self.feature_names_ = frame.columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return (
            (frame.to_numpy() - self.mean_) @ self.components_[0]
        ).reshape(-1, 1)


@dataclass
class PillarInferencePipeline:
    """Fit once and transform future feature matrices without refitting."""

    minimum_data_coverage: float = 0.20
    median_risk: float = 5.5
    confidence_exponent: float = 0.5
    apply_risk_floors: bool = True
    critical_missing_max_penalty: float = 1.5
    schema_version: int = 2
    numeric_columns_: list = field(default_factory=list)
    imputed_columns_: list = field(default_factory=list)
    empty_columns_: list = field(default_factory=list)
    economic_columns_: list = field(default_factory=list)
    industry_columns_: list = field(default_factory=list)
    critical_columns_: list = field(default_factory=list)
    log_shifts_: dict = field(default_factory=dict)
    direction_signs_: dict = field(default_factory=dict)
    risk_directions_: dict = field(default_factory=dict)
    anchor_correlations_: dict = field(default_factory=dict)
    reference_scores_: dict = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, features: pd.DataFrame, anchor: pd.Series = None):
        indexed = self._index_features(features)
        numeric = indexed.select_dtypes(include=[np.number]).copy()
        self.numeric_columns_ = numeric.columns.tolist()
        self.economic_columns_ = [
            column for column in ECONOMIC_FEATURES
            if column in self.numeric_columns_
        ]
        self.industry_columns_ = [
            column for column in INDUSTRY_FEATURES
            if column in self.numeric_columns_
        ]
        if not self.economic_columns_ or not self.industry_columns_:
            raise ValueError(
                "Both economic and industry features are required to fit "
                "the pillar pipeline"
            )
        pillar_columns = self.economic_columns_ + self.industry_columns_
        undeclared = [
            column for column in pillar_columns
            if column not in FEATURE_RISK_DIRECTIONS
        ]
        if undeclared:
            raise ValueError(
                "Pillar features without a declared risk direction: "
                f"{undeclared}. Add them to FEATURE_RISK_DIRECTIONS."
            )
        self.risk_directions_ = {
            column: FEATURE_RISK_DIRECTIONS[column]
            for column in pillar_columns
        }
        self.critical_columns_ = [
            column for column in CRITICAL_FEATURES
            if column in self.numeric_columns_
        ]

        coverage = numeric.notna().mean(axis=1)
        training = numeric.loc[coverage >= self.minimum_data_coverage]
        if len(training) < 3:
            raise ValueError("At least three sufficiently covered countries are required")

        self.imputed_columns_ = training.columns[training.notna().any()].tolist()
        self.empty_columns_ = [
            column for column in self.numeric_columns_
            if column not in self.imputed_columns_
        ]
        self.imputer_ = KNNImputer(
            n_neighbors=min(5, len(training) - 1),
            weights="distance",
        )
        imputed = pd.DataFrame(
            self.imputer_.fit_transform(training[self.imputed_columns_]),
            index=training.index,
            columns=self.imputed_columns_,
        )
        for column in self.empty_columns_:
            imputed[column] = 0.0
        imputed = imputed[self.numeric_columns_]

        transformed = self._fit_log_transforms(imputed)
        self.scaler_ = MinMaxScaler(clip=True)
        scaled = pd.DataFrame(
            self.scaler_.fit_transform(transformed),
            index=transformed.index,
            columns=transformed.columns,
        )

        oriented = self._orient(scaled)
        self.economic_pca_ = ConstrainedRiskComponent().fit(
            oriented[self.economic_columns_]
        )
        self.industry_pca_ = ConstrainedRiskComponent().fit(
            oriented[self.industry_columns_]
        )
        combined_columns = self.economic_columns_ + self.industry_columns_
        self.combined_pca_ = ConstrainedRiskComponent().fit(
            oriented[combined_columns]
        )

        # Constrained components are risk-oriented by construction (higher =
        # riskier). Downstream percentile math expects safety orientation, so
        # the signs are deterministic; the GDP anchor no longer decides score
        # direction and is recorded only as a diagnostic.
        raw_scores = {
            "economic": self.economic_pca_.transform(
                oriented[self.economic_columns_]
            )[:, 0],
            "industry": self.industry_pca_.transform(
                oriented[self.industry_columns_]
            )[:, 0],
            "combined": self.combined_pca_.transform(
                oriented[combined_columns]
            )[:, 0],
        }
        self.direction_signs_ = {
            "economic": -1.0,
            "industry": -1.0,
            "combined": -1.0,
        }
        self.anchor_correlations_ = self._anchor_correlations(
            raw_scores,
            anchor,
            oriented.index,
        )
        raw_scores["risk"] = (
            0.5 * raw_scores["economic"] * self.direction_signs_["economic"]
            + 0.5 * raw_scores["industry"] * self.direction_signs_["industry"]
        )
        self.direction_signs_["risk"] = 1.0
        self.reference_scores_ = {
            name: np.sort(
                np.round(values * self.direction_signs_[name], decimals=12)
            )
            for name, values in raw_scores.items()
        }
        self.fitted_ = True
        return self

    def transform(
        self,
        features: pd.DataFrame,
        country_names: pd.Series = None,
    ) -> pd.DataFrame:
        if not self.fitted_:
            raise ValueError("Pillar pipeline is not fitted")

        indexed = self._index_features(features)
        missing = [
            column for column in self.numeric_columns_
            if column not in indexed.columns
        ]
        for column in missing:
            indexed[column] = np.nan

        numeric = indexed[self.numeric_columns_].apply(
            pd.to_numeric,
            errors="coerce",
        )
        data_coverage = numeric.notna().mean(axis=1)
        eligible = numeric.loc[
            data_coverage >= self.minimum_data_coverage
        ].copy()
        if eligible.empty:
            return pd.DataFrame()

        original_missing = eligible.isna()
        imputed = pd.DataFrame(
            self.imputer_.transform(eligible[self.imputed_columns_]),
            index=eligible.index,
            columns=self.imputed_columns_,
        )
        for column in self.empty_columns_:
            imputed[column] = 0.0
        imputed = imputed[self.numeric_columns_]

        transformed = self._apply_log_transforms(imputed)
        scaled = pd.DataFrame(
            self.scaler_.transform(transformed),
            index=transformed.index,
            columns=transformed.columns,
        )
        oriented = self._orient(scaled)
        combined_columns = self.economic_columns_ + self.industry_columns_
        raw_scores = {
            "economic": self.economic_pca_.transform(
                oriented[self.economic_columns_]
            )[:, 0] * self.direction_signs_["economic"],
            "industry": self.industry_pca_.transform(
                oriented[self.industry_columns_]
            )[:, 0] * self.direction_signs_["industry"],
            "combined": self.combined_pca_.transform(
                oriented[combined_columns]
            )[:, 0] * self.direction_signs_["combined"],
        }
        raw_scores["risk"] = (
            0.5 * raw_scores["economic"]
            + 0.5 * raw_scores["industry"]
        )
        percentiles = {
            name: self._reference_percentile(name, values)
            for name, values in raw_scores.items()
        }

        confidence = 1 - original_missing.mean(axis=1)
        score_raw = 1 + 9 * (1 - percentiles["risk"])
        confidence_weight = confidence.pow(self.confidence_exponent)
        confidence_adjusted_scores = (
            confidence_weight * score_raw
            + (1 - confidence_weight) * self.median_risk
        )
        confidence_adjustment = confidence_adjusted_scores - score_raw

        economic_coverage = (
            1 - original_missing[self.economic_columns_].mean(axis=1)
        )
        industry_coverage = (
            1 - original_missing[self.industry_columns_].mean(axis=1)
        )
        risk_floor = pd.Series(1.0, index=eligible.index)
        if self.apply_risk_floors:
            risk_floor.loc[confidence < 0.50] = 6.0
            risk_floor.loc[
                (confidence >= 0.50) & (confidence < 0.70)
            ] = 4.0
            weak_pillar = (
                (economic_coverage < 0.30)
                | (industry_coverage < 0.30)
            )
            risk_floor.loc[weak_pillar] = np.maximum(
                risk_floor.loc[weak_pillar],
                5.0,
            )
        score_after_risk_floor = np.maximum(
            confidence_adjusted_scores,
            risk_floor,
        )
        risk_floor_delta = (
            score_after_risk_floor - confidence_adjusted_scores
        )
        risk_floor_applied = risk_floor_delta > 1e-12

        # Critical-field missingness penalty: countries missing core banking
        # soundness fields cannot be scored safer than observed peers purely
        # on the strength of imputed values.
        critical_columns = [
            column for column in getattr(self, "critical_columns_", [])
            if column in original_missing.columns
        ]
        if critical_columns:
            critical_missing_share = (
                original_missing[critical_columns].mean(axis=1)
            )
            critical_missing_fields = original_missing[critical_columns].apply(
                lambda row: tuple(
                    column for column in critical_columns if bool(row[column])
                ),
                axis=1,
            )
        else:
            critical_missing_share = pd.Series(0.0, index=eligible.index)
            critical_missing_fields = pd.Series(
                [tuple() for _ in range(len(eligible))],
                index=eligible.index,
                dtype=object,
            )
        critical_penalty = (
            critical_missing_share
            * getattr(self, "critical_missing_max_penalty", 0.0)
        )
        pre_round_structural_scores = np.minimum(
            score_after_risk_floor + critical_penalty,
            10.0,
        )
        critical_penalty_applied = (
            pre_round_structural_scores - score_after_risk_floor
        )
        structural_risk_scores = pre_round_structural_scores.round(1)

        results = pd.DataFrame(
            {
                # ``risk_score`` remains the backward-compatible structural
                # output of this pipeline. ``train_model.py`` may subsequently
                # add the separately persisted crisis uplift. The additive
                # bridge fields below expose each policy adjustment without
                # changing the score arithmetic.
                "risk_score": np.asarray(structural_risk_scores),
                "structural_risk_score": np.asarray(structural_risk_scores),
                "pillar_risk_score": np.asarray(score_raw),
                "confidence_adjustment": np.asarray(confidence_adjustment),
                "confidence_adjusted_risk_score": np.asarray(
                    confidence_adjusted_scores
                ),
                "risk_floor_value": np.asarray(risk_floor),
                "risk_floor_delta": np.asarray(risk_floor_delta),
                "score_after_risk_floor": np.asarray(score_after_risk_floor),
                "economic_pillar": percentiles["economic"] * 10,
                "industry_pillar": percentiles["industry"] * 10,
                "combined_pillar": percentiles["combined"] * 10,
                "development_level": 0.0,
                "data_coverage": data_coverage.loc[
                    eligible.index
                ].to_numpy(),
                "economic_coverage": economic_coverage.to_numpy(),
                "industry_coverage": industry_coverage.to_numpy(),
                "risk_floor_applied": np.asarray(risk_floor_applied),
                "critical_missing_share": critical_missing_share.round(
                    3
                ).to_numpy(),
                "critical_missing_fields": critical_missing_fields,
                # Preserve the exact policy amount for reconciliation; display
                # layers may round it without changing the stored bridge.
                "critical_penalty": critical_penalty.to_numpy(),
                "critical_penalty_applied": np.asarray(
                    critical_penalty_applied
                ),
                "pre_round_structural_risk_score": np.asarray(
                    pre_round_structural_scores
                ),
            },
            index=eligible.index,
        )
        results.insert(0, "country_code", eligible.index)
        results = results.reset_index(drop=True)
        results["risk_category"] = results["risk_score"].map(
            self._risk_category
        )
        if country_names is None:
            results["country_name"] = ""
        else:
            results["country_name"] = (
                results["country_code"].map(country_names).fillna("")
            )
        return results.sort_values("risk_score")

    def impute(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return the KNN-imputed numeric matrix in natural units.

        This is the imputation stage of ``transform`` without the log/scale/
        PCA steps, so callers can persist what the model actually used to
        fill gaps (e.g. the dashboard's imputed-value sidecar).
        """
        if not self.fitted_:
            raise ValueError("Pillar pipeline is not fitted")
        indexed = self._index_features(features)
        for column in self.numeric_columns_:
            if column not in indexed.columns:
                indexed[column] = np.nan
        numeric = indexed[self.numeric_columns_].apply(
            pd.to_numeric,
            errors="coerce",
        )
        eligible = numeric.loc[
            numeric.notna().mean(axis=1) >= self.minimum_data_coverage
        ].copy()
        if eligible.empty:
            return pd.DataFrame(columns=self.numeric_columns_)
        imputed = pd.DataFrame(
            self.imputer_.transform(eligible[self.imputed_columns_]),
            index=eligible.index,
            columns=self.imputed_columns_,
        )
        for column in self.empty_columns_:
            imputed[column] = 0.0
        return imputed[self.numeric_columns_]

    def loadings(self) -> dict:
        """Signed loadings on the original (unoriented) scaled feature axes.

        A positive value means a higher feature value raises the risk score;
        the constrained fit guarantees the sign matches the declared risk
        direction of each feature.
        """
        if not self.fitted_:
            return {}
        directions = getattr(self, "risk_directions_", {})

        def _signed(columns, component) -> dict:
            return {
                column: float(weight) * directions.get(column, 1.0)
                for column, weight in zip(columns, component)
            }

        return {
            "economic_loadings": _signed(
                self.economic_columns_,
                self.economic_pca_.components_[0],
            ),
            "industry_loadings": _signed(
                self.industry_columns_,
                self.industry_pca_.components_[0],
            ),
        }

    @staticmethod
    def _index_features(features: pd.DataFrame) -> pd.DataFrame:
        if "country_code" in features.columns:
            if features["country_code"].duplicated().any():
                raise ValueError("Feature matrix contains duplicate country codes")
            return features.set_index("country_code")
        if features.index.has_duplicates:
            raise ValueError("Feature matrix contains duplicate country codes")
        return features.copy()

    def _orient(self, scaled: pd.DataFrame) -> pd.DataFrame:
        """Flip scaled pillar columns so higher always means higher risk.

        Pipelines pickled before schema version 2 carry no risk-direction map
        and their PCA objects were fitted on unoriented data, so this is a
        no-op for them.
        """
        directions = getattr(self, "risk_directions_", None)
        if not directions:
            return scaled
        oriented = scaled.copy()
        for column, direction in directions.items():
            if direction < 0 and column in oriented.columns:
                oriented[column] = 1.0 - oriented[column]
        return oriented

    def _fit_log_transforms(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        if "loan_concentration" in transformed:
            transformed["loan_concentration"] = -np.log1p(
                transformed["loan_concentration"].abs()
            )
        for column in ("inflation", "sovereign_exposure_ratio"):
            if column not in transformed:
                continue
            minimum = float(transformed[column].min())
            shift = minimum if minimum < 0 else 0.0
            self.log_shifts_[column] = shift
            transformed[column] = np.log1p(
                (transformed[column] - shift).clip(lower=0)
            )
        return transformed

    def _apply_log_transforms(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        if "loan_concentration" in transformed:
            transformed["loan_concentration"] = -np.log1p(
                transformed["loan_concentration"].abs()
            )
        for column, shift in self.log_shifts_.items():
            transformed[column] = np.log1p(
                (transformed[column] - shift).clip(lower=0)
            )
        return transformed

    @staticmethod
    def _anchor_correlations(raw_scores, anchor, index) -> dict:
        """Diagnostic only: correlation of each risk-oriented raw score with
        log GDP per capita. Recorded for the policy audit; it no longer
        decides score orientation."""
        if anchor is None:
            return {}
        aligned = pd.to_numeric(anchor.reindex(index), errors="coerce")
        aligned = aligned.fillna(aligned.median()).clip(lower=100)
        anchor_log = np.log10(aligned)
        if anchor_log.std() == 0:
            return {}
        correlations = {}
        for name, values in raw_scores.items():
            correlation = np.corrcoef(anchor_log, values)[0, 1]
            if np.isfinite(correlation):
                correlations[name] = float(correlation)
        return correlations

    def _reference_percentile(self, name, values) -> np.ndarray:
        reference = self.reference_scores_[name]
        stable_values = np.round(values, decimals=12)
        return (
            np.searchsorted(reference, stable_values, side="right")
            / len(reference)
        )

    @staticmethod
    def _risk_category(score):
        if score <= 2:
            return "1-2: Very Low Risk"
        if score <= 4:
            return "3-4: Low Risk"
        if score <= 6:
            return "5-6: Moderate Risk"
        if score <= 8:
            return "7-8: High Risk"
        return "9-10: Very High Risk"

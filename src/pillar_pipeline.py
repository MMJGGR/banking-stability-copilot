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
]

INDUSTRY_FEATURES = [
    "capital_adequacy", "npl_ratio", "roe", "roa",
    "liquid_assets_st_liab", "liquid_assets_total",
    "customer_deposits_loans", "fx_loan_exposure", "tier1_capital",
    "npl_provisions", "loan_concentration", "real_estate_loans",
    "sovereign_exposure_ratio", "bank_liability_to_nfa",
    "real_estate_credit_growth_3yr", "regulatory_quality",
    "rule_of_law", "control_corruption",
]


@dataclass
class PillarInferencePipeline:
    """Fit once and transform future feature matrices without refitting."""

    minimum_data_coverage: float = 0.20
    median_risk: float = 5.5
    confidence_exponent: float = 0.5
    apply_risk_floors: bool = True
    schema_version: int = 1
    numeric_columns_: list = field(default_factory=list)
    imputed_columns_: list = field(default_factory=list)
    empty_columns_: list = field(default_factory=list)
    economic_columns_: list = field(default_factory=list)
    industry_columns_: list = field(default_factory=list)
    log_shifts_: dict = field(default_factory=dict)
    direction_signs_: dict = field(default_factory=dict)
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

        self.economic_pca_ = self._fit_pca(scaled[self.economic_columns_])
        self.industry_pca_ = self._fit_pca(scaled[self.industry_columns_])
        combined_columns = self.economic_columns_ + self.industry_columns_
        self.combined_pca_ = self._fit_pca(scaled[combined_columns])

        raw_scores = {
            "economic": self.economic_pca_.transform(
                scaled[self.economic_columns_]
            )[:, 0],
            "industry": self.industry_pca_.transform(
                scaled[self.industry_columns_]
            )[:, 0],
            "combined": self.combined_pca_.transform(
                scaled[combined_columns]
            )[:, 0],
        }
        self.direction_signs_ = self._direction_signs(
            raw_scores,
            anchor,
            scaled.index,
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
        combined_columns = self.economic_columns_ + self.industry_columns_
        raw_scores = {
            "economic": self.economic_pca_.transform(
                scaled[self.economic_columns_]
            )[:, 0] * self.direction_signs_["economic"],
            "industry": self.industry_pca_.transform(
                scaled[self.industry_columns_]
            )[:, 0] * self.direction_signs_["industry"],
            "combined": self.combined_pca_.transform(
                scaled[combined_columns]
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
        risk_scores = (
            confidence_weight * score_raw
            + (1 - confidence_weight) * self.median_risk
        )

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
        before_floor = risk_scores.copy()
        risk_scores = np.maximum(risk_scores, risk_floor).round(1)

        results = pd.DataFrame(
            {
                "risk_score": np.asarray(risk_scores),
                "economic_pillar": percentiles["economic"] * 10,
                "industry_pillar": percentiles["industry"] * 10,
                "combined_pillar": percentiles["combined"] * 10,
                "development_level": 0.0,
                "data_coverage": data_coverage.loc[
                    eligible.index
                ].to_numpy(),
                "economic_coverage": economic_coverage.to_numpy(),
                "industry_coverage": industry_coverage.to_numpy(),
                "risk_floor_applied": np.asarray(
                    risk_scores > before_floor.round(1)
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
        if not self.fitted_:
            return {}
        return {
            "economic_loadings": dict(
                zip(
                    self.economic_columns_,
                    self.economic_pca_.components_[0],
                )
            ),
            "industry_loadings": dict(
                zip(
                    self.industry_columns_,
                    self.industry_pca_.components_[0],
                )
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

    @staticmethod
    def _fit_pca(frame: pd.DataFrame) -> PCA:
        components = min(5, frame.shape[1] - 1, frame.shape[0] - 1)
        if components < 1:
            raise ValueError("PCA requires at least two features and countries")
        return PCA(n_components=components).fit(frame)

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
    def _direction_signs(raw_scores, anchor, index) -> dict:
        signs = {name: 1.0 for name in raw_scores}
        if anchor is None:
            return signs
        aligned = pd.to_numeric(anchor.reindex(index), errors="coerce")
        aligned = aligned.fillna(aligned.median()).clip(lower=100)
        anchor_log = np.log10(aligned)
        if anchor_log.std() == 0:
            return signs
        for name, values in raw_scores.items():
            correlation = np.corrcoef(anchor_log, values)[0, 1]
            if np.isfinite(correlation) and correlation < 0:
                signs[name] = -1.0
        return signs

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

"""Global risk triage view and its presentation-safe data helpers."""

import html
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry_convert as pc
import streamlit as st

from src.dashboard.presentation import accessible_plotly_config


# Codes pycountry_convert cannot map (IMF aggregates and small territories).
CONTINENT_OVERRIDES = {
    "TLS": "Asia",
    "WBG": "Asia",
    "KOS": "Europe",
    "UVK": "Europe",
    "SXM": "North America",
    "CUW": "North America",
}

HIGH_RISK_THRESHOLD = 6.0
LOW_RISK_THRESHOLD = 4.0
WATCHLIST_LIMIT = 10
RISK_COLOR_SCALE = [
    [0.0, "#D9EAF5"],
    [0.35, "#75ADD1"],
    [0.55, "#F2C572"],
    [0.75, "#D96B55"],
    [1.0, "#8F1D2C"],
]


def get_continent_name(alpha3_code: str) -> str:
    """Convert an ISO-3 country code to a continent name."""
    if alpha3_code in CONTINENT_OVERRIDES:
        return CONTINENT_OVERRIDES[alpha3_code]
    try:
        alpha2 = pc.country_alpha3_to_country_alpha2(alpha3_code)
        continent_code = pc.country_alpha2_to_continent_code(alpha2)
        continent_names = {
            "NA": "North America",
            "SA": "South America",
            "AS": "Asia",
            "EU": "Europe",
            "AF": "Africa",
            "OC": "Oceania",
            "AN": "Antarctica",
        }
        return continent_names.get(continent_code, "Other")
    except Exception:
        return "Other"


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def weaker_pillar_label(row: pd.Series) -> str:
    """Name the weaker strength pillar without claiming feature attribution."""
    economic = pd.to_numeric(row.get("economic_pillar"), errors="coerce")
    banking = pd.to_numeric(row.get("industry_pillar"), errors="coerce")
    if pd.isna(economic) or pd.isna(banking):
        return "Not comparable"
    if economic < banking:
        return "Operating environment"
    if banking < economic:
        return "Banking system"
    return "Similar strength"


def build_systemic_watchlist(
    df: pd.DataFrame,
    *,
    risk_threshold: float = HIGH_RISK_THRESHOLD,
    limit: int = WATCHLIST_LIMIT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the disclosed large-economy watchlist and its audit metadata."""
    required = {"risk_score", "nominal_gdp"}
    if not required.issubset(df.columns):
        return pd.DataFrame(), {
            "available": False,
            "reason": "Risk score and nominal GDP are required.",
        }

    working = df.copy()
    working["risk_score"] = _numeric_series(working, "risk_score")
    working["nominal_gdp"] = _numeric_series(working, "nominal_gdp")
    valid_mask = (
        working["risk_score"].notna()
        & working["nominal_gdp"].notna()
        & (working["nominal_gdp"] > 0)
    )
    eligible = working.loc[valid_mask].copy()
    excluded_missing = int((~valid_mask).sum())
    if eligible.empty:
        return pd.DataFrame(), {
            "available": False,
            "reason": "No country has both a risk score and positive nominal GDP.",
            "excluded_missing": excluded_missing,
        }

    gdp_median = float(eligible["nominal_gdp"].median())
    matched = eligible.loc[
        (eligible["nominal_gdp"] > gdp_median)
        & (eligible["risk_score"] > risk_threshold)
    ].copy()
    matched = matched.sort_values(
        ["risk_score", "nominal_gdp"], ascending=[False, False]
    )
    matched_total = len(matched)
    matched = matched.head(limit).copy()
    matched["weaker_pillar"] = matched.apply(weaker_pillar_label, axis=1)
    return matched, {
        "available": True,
        "risk_threshold": float(risk_threshold),
        "gdp_median": gdp_median,
        "eligible_countries": len(eligible),
        "excluded_missing": excluded_missing,
        "matched_total": matched_total,
        "omitted_by_limit": max(matched_total - len(matched), 0),
        "limit": int(limit),
    }


def regional_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate GDP-weighted regional risk, falling back to an unweighted mean."""
    rows: list[dict[str, Any]] = []
    for region, region_df in df.groupby("Region", dropna=False):
        risks = _numeric_series(region_df, "risk_score")
        gdps = _numeric_series(region_df, "nominal_gdp")
        valid_weight = risks.notna() & gdps.notna() & (gdps > 0)
        if valid_weight.any() and float(gdps.loc[valid_weight].sum()) > 0:
            score = float(
                (risks.loc[valid_weight] * gdps.loc[valid_weight]).sum()
                / gdps.loc[valid_weight].sum()
            )
            basis = "GDP weighted"
        else:
            score = float(risks.mean()) if risks.notna().any() else float("nan")
            basis = "Unweighted; GDP unavailable"
        rows.append(
            {
                "Region": region,
                "Weighted Risk": score,
                "Countries": int(len(region_df)),
                "Basis": basis,
            }
        )
    return (
        pd.DataFrame(rows)
        .dropna(subset=["Weighted Risk"])
        .sort_values("Weighted Risk", ascending=False)
        .reset_index(drop=True)
    )


def calculate_weighted_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate nominal-GDP-weighted global metrics."""
    if "nominal_gdp" not in df.columns:
        return {}
    valid_df = df.copy()
    valid_df["nominal_gdp"] = _numeric_series(valid_df, "nominal_gdp")
    valid_df = valid_df[
        valid_df["nominal_gdp"].notna() & (valid_df["nominal_gdp"] > 0)
    ]
    if valid_df.empty:
        return {}

    def weighted_average(column: str) -> float:
        values = _numeric_series(valid_df, column)
        mask = values.notna()
        if not mask.any():
            return float("nan")
        weights = valid_df.loc[mask, "nominal_gdp"]
        return float((values.loc[mask] * weights).sum() / weights.sum())

    total_gdp = float(valid_df["nominal_gdp"].sum())
    return {
        "total_gdp_trillions": total_gdp / 1000,
        "global_risk_score": weighted_average("risk_score"),
        "global_economic_pillar": weighted_average("economic_pillar"),
        "global_industry_pillar": weighted_average("industry_pillar"),
        "global_npl": weighted_average("npl_ratio"),
        "global_capital_adequacy": weighted_average("capital_adequacy"),
    }


def _render_kpi_grid(items: list[tuple[str, str, str]]) -> None:
    """Render compact, theme-aware KPIs without misleading delta arrows."""
    cards = []
    for label, value, context in items:
        cards.append(
            "<div class='bankenv-kpi-card' role='listitem'>"
            f"<div class='bankenv-kpi-label'>{html.escape(str(label))}</div>"
            f"<div class='bankenv-kpi-value'>{html.escape(str(value))}</div>"
            f"<div class='bankenv-kpi-context'>{html.escape(str(context))}</div>"
            "</div>"
        )
    markup = (
        '<div class="bankenv-kpi-grid" role="list" '
        'aria-label="Global risk summary">'
        + "".join(cards)
        + "</div>"
    )
    html_renderer = getattr(st, "html", None)
    if callable(html_renderer):
        html_renderer(markup)
    else:  # pragma: no cover - serving requirements include st.html
        st.markdown(markup, unsafe_allow_html=True)


def _prepare_country_profile(country_code: str) -> None:
    """Seed the existing Country selector and a future page-navigation hook."""
    st.session_state["profile_country_code"] = country_code
    st.session_state["primary_view"] = "Country"


def _render_watchlist(df: pd.DataFrame) -> None:
    st.markdown("## Systemic Risk Watchlist")
    watchlist, metadata = build_systemic_watchlist(df)
    if not metadata.get("available"):
        st.info(
            "Watchlist unavailable. "
            + str(metadata.get("reason", "Required data is missing."))
        )
        return

    rule = (
        f"Rule: risk score > {metadata['risk_threshold']:.1f} and nominal GDP "
        f"above the valid-country median of ${metadata['gdp_median']:,.0f}bn. "
        f"{metadata['eligible_countries']:,} countries had both inputs; "
        f"{metadata['excluded_missing']:,} were excluded for missing or "
        "non-positive GDP or a missing risk score."
    )
    if metadata["omitted_by_limit"]:
        rule += (
            f" Showing the {metadata['limit']} highest scores; "
            f"{metadata['omitted_by_limit']} additional matches are omitted."
        )
    st.caption(rule)

    if watchlist.empty:
        st.info("No country meets the disclosed watchlist rule in this snapshot.")
        return

    compact = pd.DataFrame(
        {
            "Country": watchlist["country_name"],
            "Risk Score": watchlist["risk_score"].astype(float),
            "Weaker Pillar": watchlist["weaker_pillar"],
        }
    ).reset_index(drop=True)
    st.dataframe(
        compact,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Country": st.column_config.TextColumn("Country", width="medium"),
            "Risk Score": st.column_config.NumberColumn(
                "Risk Score", format="%.1f", help="1 lower risk; 10 higher risk"
            ),
            "Weaker Pillar": st.column_config.TextColumn(
                "Weaker Pillar",
                help=(
                    "Lower of the two strength percentiles; this is a fallback "
                    "comparison, not feature attribution."
                ),
            ),
        },
    )
    st.caption(
        "Weaker pillar compares Operating Environment and Banking System strength "
        "percentiles, where higher is stronger. It is not a feature-level risk driver."
    )

    country_options = watchlist["country_code"].astype(str).tolist()
    country_names = dict(zip(country_options, watchlist["country_name"].astype(str)))
    action_col, button_col = st.columns([3, 1])
    with action_col:
        country_to_open = st.selectbox(
            "Country to analyze",
            options=country_options,
            format_func=lambda code: country_names.get(code, code),
            key="global_watchlist_country",
        )
    with button_col:
        st.write("")
        st.button(
            "Open country",
            key="global_open_country",
            use_container_width=True,
            on_click=_prepare_country_profile,
            args=(country_to_open,),
        )
    with st.expander("Watchlist details", expanded=False):
        details = pd.DataFrame(
            {
                "Country": watchlist["country_name"],
                "Region": watchlist["Region"],
                "Risk Score": watchlist["risk_score"].astype(float),
                "Nominal GDP (USD bn)": watchlist["nominal_gdp"].astype(float),
                "Operating Environment Strength": _numeric_series(
                    watchlist, "economic_pillar"
                ),
                "Banking System Strength": _numeric_series(
                    watchlist, "industry_pillar"
                ),
            }
        ).reset_index(drop=True)
        st.dataframe(details, use_container_width=True, hide_index=True)


def _render_risk_map(df: pd.DataFrame) -> None:
    st.markdown("## Risk Distribution Map")
    st.caption(
        "The map uses color as an overview only. The ranked table below provides "
        "the same risk values without relying on color or hover."
    )
    map_df = df.dropna(subset=["risk_score", "country_code"]).copy()
    if map_df.empty:
        st.info("Map unavailable because no country has a valid risk score.")
        return
    if "country_name" not in map_df.columns:
        map_df["country_name"] = map_df["country_code"]
    map_df["gdp_billions"] = _numeric_series(map_df, "nominal_gdp").map(
        lambda value: f"${value:,.0f}bn" if pd.notna(value) else "Unavailable"
    )

    hover_data: dict[str, Any] = {
        "risk_score": ":.1f",
        "gdp_billions": True,
        "country_code": False,
    }
    if "economic_pillar" in map_df.columns:
        hover_data["economic_pillar"] = ":.1f"
    if "industry_pillar" in map_df.columns:
        hover_data["industry_pillar"] = ":.1f"
    fig_map = px.choropleth(
        map_df,
        locations="country_code",
        color="risk_score",
        hover_name="country_name",
        hover_data=hover_data,
        labels={
            "risk_score": "Risk score (1 lower; 10 higher)",
            "economic_pillar": "Operating environment strength",
            "industry_pillar": "Banking system strength",
            "gdp_billions": "Nominal GDP",
        },
        color_continuous_scale=RISK_COLOR_SCALE,
        range_color=(1, 10),
        projection="natural earth",
    )
    fig_map.update_layout(
        margin={"r": 0, "t": 5, "l": 0, "b": 55},
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        geo_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar={
            "title": "Risk: 1 lower, 10 higher",
            "tickvals": [1, 3, 5, 7, 9],
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.12,
            "len": 0.75,
            "thickness": 12,
        },
    )
    st.plotly_chart(
        fig_map,
        use_container_width=True,
        theme="streamlit",
        key="global_risk_map",
        config=accessible_plotly_config(),
    )

    ranked = map_df.sort_values(
        ["risk_score", "country_name"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    ranked_table = ranked[["Rank", "country_name", "Region", "risk_score"]].rename(
        columns={"country_name": "Country", "risk_score": "Risk Score"}
    )
    with st.expander("View ranked country risk data", expanded=False):
        st.dataframe(
            ranked_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "Risk Score": st.column_config.NumberColumn(
                    "Risk Score", format="%.1f", help="1 lower risk; 10 higher risk"
                ),
            },
        )


def _render_regional_charts(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("## Regional Risk Profile")
        if summary.empty:
            st.info("Regional risk is unavailable because no valid scores were loaded.")
        else:
            chart_regions = summary.sort_values("Weighted Risk")
            positions = [
                max(0.0, min(1.0, (value - 1.0) / 9.0))
                for value in chart_regions["Weighted Risk"]
            ]
            colors = px.colors.sample_colorscale(RISK_COLOR_SCALE, positions)
            fig_bar = go.Figure(
                go.Bar(
                    x=chart_regions["Weighted Risk"],
                    y=chart_regions["Region"],
                    orientation="h",
                    marker_color=colors,
                    text=chart_regions["Weighted Risk"].map(
                        lambda value: f"{value:.1f}"
                    ),
                    textposition="outside",
                    cliponaxis=False,
                    customdata=chart_regions[["Countries", "Basis"]],
                    hovertemplate=(
                        "%{y}<br>Risk %{x:.1f}/10<br>Countries %{customdata[0]}"
                        "<br>%{customdata[1]}<extra></extra>"
                    ),
                )
            )
            fig_bar.update_layout(
                xaxis_title="Risk score (1 lower; 10 higher)",
                yaxis_title="",
                showlegend=False,
                margin={"r": 35, "t": 5, "l": 10, "b": 45},
                height=max(320, 48 * len(chart_regions)),
                xaxis={"range": [0, 10]},
            )
            st.plotly_chart(
                fig_bar,
                use_container_width=True,
                theme="streamlit",
                key="global_regional_risk",
                config=accessible_plotly_config(),
            )

    with col_right:
        st.markdown("## Risk and GDP Growth")
        required = {"gdp_growth", "risk_score", "nominal_gdp"}
        if not required.issubset(df.columns):
            st.info(
                "Risk and growth comparison unavailable because GDP growth or "
                "nominal GDP is not packaged in this snapshot."
            )
            return

        source = df[df["Region"] != "Other"].copy()
        for column in required:
            source[column] = _numeric_series(source, column)
        missing_inputs = int(source[list(required)].isna().any(axis=1).sum())
        complete = source.dropna(subset=list(required))
        complete = complete[complete["nominal_gdp"] > 0]
        in_range = complete["gdp_growth"].between(-10, 15)
        outlier_count = int((~in_range).sum())
        plot_df = complete.loc[in_range].copy()
        if plot_df.empty:
            st.info(
                "No countries remain after requiring positive GDP and GDP growth "
                "between -10% and 15%."
            )
            return

        fig_scatter = px.scatter(
            plot_df,
            x="gdp_growth",
            y="risk_score",
            size="nominal_gdp",
            color="Region",
            hover_name="country_name",
            size_max=45,
            labels={
                "gdp_growth": "Real GDP growth (%)",
                "risk_score": "Risk score (1 lower; 10 higher)",
            },
        )
        fig_scatter.update_layout(
            height=390,
            margin={"r": 5, "t": 5, "l": 5, "b": 70},
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.18,
                "xanchor": "left",
                "x": 0,
                "title": None,
            },
            yaxis={"range": [1, 10]},
        )
        st.plotly_chart(
            fig_scatter,
            use_container_width=True,
            theme="streamlit",
            key="global_risk_growth",
            config=accessible_plotly_config(),
        )
        st.caption(
            f"Includes {len(plot_df):,} countries with growth from -10% to 15%; "
            f"excludes {outlier_count:,} observations outside that range and "
            f"{missing_inputs:,} with missing chart inputs."
        )


def render_global_summary(
    scores_df: pd.DataFrame,
    model_features: pd.DataFrame,
    loader,
) -> None:
    """Render the Global triage view with explicit risk semantics."""
    del loader  # Reserved for future source-linked drill-through.
    st.markdown("# Global Risk Landscape")
    st.caption(
        "Country risk scores use a 1–10 scale: 1 is lower relative risk and 10 "
        "is higher relative risk. Global and regional aggregates are weighted "
        "by nominal GDP where GDP is available."
    )
    if scores_df is None or len(scores_df) == 0:
        st.warning(
            "Global summary unavailable because no country scores were loaded. "
            "Retry after the active snapshot finishes loading."
        )
        return

    df = scores_df.copy()
    if model_features is not None:
        columns = [
            column
            for column in model_features.columns
            if column not in df.columns or column == "country_code"
        ]
        df = df.merge(model_features[columns], on="country_code", how="left")
    df["Region"] = df["country_code"].apply(get_continent_name)

    metrics = calculate_weighted_metrics(df)
    risk_values = _numeric_series(df, "risk_score")
    countries_scored = int(risk_values.notna().sum())
    high_risk_count = int((risk_values > HIGH_RISK_THRESHOLD).sum())
    low_risk_count = int((risk_values < LOW_RISK_THRESHOLD).sum())
    low_risk_share = low_risk_count / countries_scored if countries_scored else 0.0
    global_score = metrics.get("global_risk_score", float("nan"))
    global_score_text = (
        f"{global_score:.1f}/10" if pd.notna(global_score) else "Unavailable"
    )

    region_summary = regional_risk_summary(df)
    if region_summary.empty:
        highest_region = "Unavailable"
        highest_context = "No regional score available"
    else:
        highest = region_summary.iloc[0]
        highest_region = str(highest["Region"])
        highest_context = (
            f"Risk {highest['Weighted Risk']:.1f}/10; {str(highest['Basis']).lower()}"
        )
    _render_kpi_grid(
        [
            ("Global weighted risk", global_score_text, "Higher means more relative risk"),
            (
                "Countries scored",
                f"{countries_scored:,}",
                f"{high_risk_count:,} above {HIGH_RISK_THRESHOLD:.1f}",
            ),
            ("Highest-risk region", highest_region, highest_context),
            (
                "Lower-risk economies",
                f"{low_risk_count:,}",
                f"{low_risk_share:.0%} below {LOW_RISK_THRESHOLD:.1f}",
            ),
        ]
    )

    _render_watchlist(df)
    _render_risk_map(df)
    _render_regional_charts(df, region_summary)
    with st.expander("Download global analysis data", expanded=False):
        export_columns = [
            column
            for column in (
                "country_code",
                "country_name",
                "Region",
                "risk_score",
                "nominal_gdp",
                "gdp_growth",
                "economic_pillar",
                "industry_pillar",
                "data_coverage",
            )
            if column in df.columns
        ]
        country_export = df[export_columns].sort_values(
            ["risk_score", "country_name"],
            ascending=[False, True],
        )
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "Download country risk data",
                data=country_export.to_csv(index=False).encode("utf-8"),
                file_name="bankenv_global_country_risk.csv",
                mime="text/csv",
                key="download_global_country_risk",
            )
        with export_col2:
            st.download_button(
                "Download regional risk data",
                data=region_summary.to_csv(index=False).encode("utf-8"),
                file_name="bankenv_global_regional_risk.csv",
                mime="text/csv",
                key="download_global_regional_risk",
            )

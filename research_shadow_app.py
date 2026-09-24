"""Read-only Streamlit viewer for the frozen Banking Copilot research bundle.

This is a separate research entry point. It never writes production artifacts,
never alters production scores and never serves the rejected Phase 4 crisis
overlay.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.forecasting.phase5_shadow_view import (
    country_options,
    load_country_record,
    load_shadow_bundle,
)

st.set_page_config(
    page_title="Banking Copilot — Research Shadow",
    layout="wide",
)
st.title("Banking Copilot — Research Shadow")
st.warning(
    "Research only. This view does not replace the production score or crisis "
    "classifier, and no output shown here is deployed."
)

bundle_path = Path(
    os.environ.get("BANKING_COPILOT_SHADOW_BUNDLE", "shadow_bundle")
)
if not bundle_path.is_dir():
    st.error(
        "Shadow bundle not found. Set BANKING_COPILOT_SHADOW_BUNDLE to the "
        "frozen Phase 5 bundle directory."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def cached_bundle(path: str):
    return load_shadow_bundle(path, verify_hashes=True)


@st.cache_data(show_spinner=False)
def cached_record(path: str, entity_code: str, horizon: int):
    bundle = load_shadow_bundle(path, verify_hashes=False)
    return load_country_record(bundle, entity_code, horizon)


bundle = cached_bundle(str(bundle_path))
options = country_options(bundle)
lookup = dict(zip(options.display, options.entity_code))
selection = st.selectbox("Country", options.display.tolist())
horizon = st.radio("Forecast horizon", [1, 2], horizontal=True)
record = cached_record(str(bundle_path), lookup[selection], horizon)
summary = record["summary"]

st.caption(
    f"Batch {summary['forecast_batch_id']} · source cutoff "
    f"{summary['source_cutoff']} · state origin {summary['current_state_year']} · "
    f"forecast year {summary['forecast_year']}"
)

production, research = st.columns(2)
with production:
    st.subheader("Production benchmark")
    if summary["production_reference_available"]:
        st.metric("Risk score", f"{summary['production_risk_score']:.1f}")
        st.write(summary["production_risk_category"])
        if summary["production_crisis_probability"] is not None:
            st.metric(
                "Production crisis probability",
                f"{100 * summary['production_crisis_probability']:.1f}%",
            )
        st.caption(
            "Read-only production reference; not blended into the research forecast."
        )
    else:
        st.info("No production benchmark exists for this research entity.")

with research:
    st.subheader("Research state forecast")
    st.metric("Forecast quality", str(summary["forecast_quality"]).title())
    st.metric(
        "Observed model features",
        f"{int(summary['observed_model_features']):,}",
    )
    st.metric(
        "State uncertainty",
        f"{summary['state_uncertainty_proxy']:.3f}",
    )
    st.caption(
        "The 96-dimensional state is internal and is not a production risk score."
    )

st.subheader("Probabilistic movement")
columns = st.columns(4)
columns[0].metric(
    "50% movement radius",
    f"{summary['movement_radius_q50']:.3f}",
)
columns[1].metric(
    "80% movement radius",
    f"{summary['movement_radius_q80']:.3f}",
)
columns[2].metric(
    "95% movement radius",
    f"{summary['movement_radius_q95']:.3f}",
)
relative_improvement = summary["probability_relative_percentile_improves"]
columns[3].metric(
    "Relative percentile improves",
    (
        f"{100 * relative_improvement:.1f}%"
        if relative_improvement is not None
        else "Not available"
    ),
)

peer_left, peer_right = st.columns(2)
with peer_left:
    st.write("**Future peer-position range**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Current percentile": summary[
                        "current_peer_distance_percentile"
                    ],
                    "Future p10": summary["future_peer_percentile_q10"],
                    "Future median": summary["future_peer_percentile_q50"],
                    "Future p90": summary["future_peer_percentile_q90"],
                }
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
with peer_right:
    st.write("**Center movement probabilities**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Farther from peer center": summary[
                        "probability_farther_from_contemporary_peer_center"
                    ],
                    "Closer to peer center": summary[
                        "probability_closer_to_contemporary_peer_center"
                    ],
                }
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

st.subheader("What explains this country?")
state_tab, movement_tab, observable_tab = st.tabs(
    ["Current state", "Expected movement", "Observable implications"]
)
with state_tab:
    st.dataframe(
        pd.DataFrame(record["current_state_explanations"]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Dimension labels are machine-generated descriptions and require "
        "analyst review."
    )
with movement_tab:
    st.dataframe(
        pd.DataFrame(record["movement_explanations"]),
        hide_index=True,
        use_container_width=True,
    )
with observable_tab:
    st.dataframe(
        pd.DataFrame(record["observable_implications"]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Most observable implications are directional attribution, not "
        "validated raw-level forecasts."
    )

st.subheader("Peers and historical context")
peers_column, analogues_column = st.columns(2)
with peers_column:
    st.write("**Likely future nearest peers**")
    st.dataframe(
        pd.DataFrame(record["future_peers"]),
        hide_index=True,
        use_container_width=True,
    )
with analogues_column:
    st.write("**Historical analogues**")
    st.dataframe(
        pd.DataFrame(record["historical_analogues"]),
        hide_index=True,
        use_container_width=True,
    )

st.subheader("Crisis and provider-scenario boundaries")
st.error(
    "The Phase 4 state-based crisis overlay was rejected and is not served. "
    "Only the independent production crisis probability may appear above."
)
if summary["provider_scenario_available"]:
    st.info(
        f"IMF WEO provider projections are available for "
        f"{summary['forecast_year']} as a separate scenario lane. They are "
        "not included in this baseline forecast."
    )
else:
    st.caption(
        "No WEO provider-scenario rows are available for this country/year."
    )
st.caption(summary["weo_2025_status_caveat"])

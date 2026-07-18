"""
Fraud detection app -- loads the trained artifact, takes real transaction
inputs, and returns a probability with an explanation of what drove it.

Run: streamlit run app.py

Design notes for the interview:
- The app NEVER retrains the model. It loads models/fraud_model.joblib,
  which was produced once by train.py. This mirrors how real ML systems
  separate an offline training job from an online serving path.
- Feature engineering here calls the exact same engineer_features()
  function used in train.py (imported from features.py), so the model
  always sees inputs shaped the way it was trained on.
- Because the winning model is Logistic Regression, its coefficients are
  directly interpretable. The "why" panel multiplies each standardized
  feature value by its coefficient to show which factors pushed this
  specific prediction toward fraud vs. legitimate -- this is the kind
  of thing interviewers like to hear you explain (a poor-man's SHAP).
"""

import json
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features

st.set_page_config(page_title="Fraud Radar", page_icon="\U0001F6F0", layout="wide")

# ---------------------------------------------------------------- styling --
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #0b0f19 0%, #10162b 100%); }
h1, h2, h3, p, label, .stMarkdown { color: #e8ebf5 !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; }
[data-testid="stMetricLabel"] { color: #9aa4c4 !important; }
div[data-testid="stForm"] {
    background: #141a2e; border: 1px solid #262e4a; border-radius: 14px;
    padding: 1.6rem 1.8rem;
}
.result-card {
    background: #141a2e; border: 1px solid #262e4a; border-radius: 14px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.badge {
    display: inline-block; padding: 6px 16px; border-radius: 999px;
    font-weight: 600; font-size: 0.95rem; letter-spacing: 0.02em;
}
.badge-high { background: #3a1220; color: #ff8a9e; border: 1px solid #7a2338; }
.badge-medium { background: #3a2e10; color: #ffcf7a; border: 1px solid #7a5c1e; }
.badge-low { background: #10301f; color: #7affb0; border: 1px solid #1e7a49; }
.factor-row { display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid #262e4a; font-size: 0.92rem; }
.stButton>button,
button[kind="formSubmit"],
div[data-testid="stFormSubmitButton"] button {
    background-color: #64e38a !important;
    color: black !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
}

.stButton>button:hover,
button[kind="formSubmit"]:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #2e8f4b !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

CATEGORY_OPTIONS = [
    "grocery_pos", "grocery_net", "shopping_pos", "shopping_net", "misc_pos",
    "misc_net", "gas_transport", "entertainment", "food_dining",
    "personal_care", "health_fitness", "travel",
]


@st.cache_resource
def load_artifact():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model = joblib.load(os.path.join(BASE_DIR, "models", "fraud_model.joblib"))
    with open(os.path.join(BASE_DIR, "models", "metadata.json")) as f:
        metadata = json.load(f)
    return model, metadata


def explain_logistic(pipeline, row_df):
    """Per-prediction feature contributions for a Logistic Regression
    pipeline: standardized_value * coefficient, summed to the log-odds.
    Only meaningful for the linear model -- tree ensembles need a
    different approach (e.g. SHAP TreeExplainer), noted in the README.
    """
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None

    transformed = prep.transform(row_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = prep.get_feature_names_out()
    contributions = transformed[0] * clf.coef_[0]

    contrib_df = pd.DataFrame({"feature": feature_names, "contribution": contributions})
    contrib_df["abs"] = contrib_df["contribution"].abs()
    return contrib_df.sort_values("abs", ascending=False).head(6)


def risk_gauge(probability):
    color = "#ff5c7a" if probability >= 0.5 else ("#ffcf7a" if probability >= 0.2 else "#38d68a")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"color": "#e8ebf5", "size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9aa4c4"},
            "bar": {"color": color},
            "bgcolor": "#1a2138",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "#10301f"},
                {"range": [20, 50], "color": "#3a2e10"},
                {"range": [50, 100], "color": "#3a1220"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


# ------------------------------------------------------------------- app --
st.markdown("## \U0001F6F0 Fraud Radar")
st.markdown(
    "<p style='color:#9aa4c4;margin-top:-8px'>Real-time credit card transaction risk scoring</p>",
    unsafe_allow_html=True,
)

try:
    model, metadata = load_artifact()
except FileNotFoundError:
    st.error("No trained model found. Run `python3 train.py` first to produce models/fraud_model.joblib.")
    st.stop()

with st.sidebar:
    st.markdown("### Model info")
    st.write(f"**Model:** {metadata['model_name']}")
    st.write(f"**Trained on:** {metadata['n_train']:,} transactions")
    st.write(f"**Tested on:** {metadata['n_test']:,} transactions "
             f"({metadata['test_fraud_rate']:.2%} fraud, real ratio)")
    st.markdown("### Held-out test metrics")
    for k, v in metadata["test_metrics"].items():
        st.write(f"**{k}:** {v:.3f}")
    st.caption(
        "Metrics come from an untouched test set kept at the real "
        "fraud ratio -- not a balanced sample -- so they reflect "
        "real-world performance, not an inflated number."
    )

left, right = st.columns([1.1, 1])

with left:
    with st.form("transaction_form"):
        st.markdown("#### Transaction details")
        c1, c2 = st.columns(2)
        amt = c1.number_input("Amount ($)", min_value=0.01, value=84.50, step=1.0)
        category = c2.selectbox("Merchant category", CATEGORY_OPTIONS, index=2)

        c3, c4 = st.columns(2)
        trans_date = c3.date_input("Transaction date", value=date(2023, 6, 15))
        trans_time = c4.time_input("Transaction time", value=datetime(2023, 6, 15, 22, 30).time())

        dob = st.date_input(
            "Cardholder date of birth", value=date(1985, 4, 12),
            min_value=date(1920, 1, 1), max_value=date(2007, 1, 1),
        )

        st.markdown("#### Location")
        c5, c6 = st.columns(2)
        home_lat = c5.number_input("Cardholder latitude", value=36.08, format="%.4f")
        home_lon = c6.number_input("Cardholder longitude", value=-81.18, format="%.4f")
        c7, c8 = st.columns(2)
        merch_lat = c7.number_input("Merchant latitude", value=36.01, format="%.4f")
        merch_lon = c8.number_input("Merchant longitude", value=-82.05, format="%.4f")

        city_pop = st.number_input("Cardholder city population", min_value=1, value=3495, step=100)

        submitted = st.form_submit_button("Score transaction")

with right:
    if submitted:
        trans_dt = datetime.combine(trans_date, trans_time)
        raw_row = pd.DataFrame([{
            "trans_date_trans_time": trans_dt,
            "dob": dob,
            "amt": amt,
            "city_pop": city_pop,
            "lat": home_lat, "long": home_lon,
            "merch_lat": merch_lat, "merch_long": merch_lon,
            "category": category,
        }])
        feats = engineer_features(raw_row)[ALL_FEATURES]

        probability = model.predict_proba(feats)[0, 1]
        label = "FRAUD" if probability >= 0.5 else "LEGITIMATE"

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.plotly_chart(risk_gauge(probability), use_container_width=True)

        if probability >= 0.5:
            badge = '<span class="badge badge-high">High risk \u2014 flagged as fraud</span>'
        elif probability >= 0.2:
            badge = '<span class="badge badge-medium">Medium risk \u2014 review suggested</span>'
        else:
            badge = '<span class="badge badge-low">Low risk \u2014 likely legitimate</span>'
        st.markdown(badge, unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:#9aa4c4;margin-top:10px'>Distance between cardholder and merchant: "
            f"<b>{feats['distance_km'].iloc[0]:.1f} km</b> &nbsp;|&nbsp; "
            f"Transaction hour: <b>{feats['hour'].iloc[0]:02d}:00</b></p>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        contrib = explain_logistic(model, feats)
        if contrib is not None:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("#### What drove this score")
            st.caption("Each factor's contribution to the fraud log-odds. Positive pushes toward fraud, negative pushes toward legitimate.")
            for _, r in contrib.iterrows():
                direction = "\u2191 toward fraud" if r["contribution"] > 0 else "\u2193 toward legitimate"
                color = "#ff8a9e" if r["contribution"] > 0 else "#7affb0"
                st.markdown(
                    f'<div class="factor-row"><span>{r["feature"]}</span>'
                    f'<span style="color:{color}">{direction} ({r["contribution"]:+.2f})</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="result-card"><p style="color:#9aa4c4">'
            'Fill in the transaction details and click <b>Score transaction</b> '
            'to see the fraud risk assessment.</p></div>',
            unsafe_allow_html=True,
        )

"""
Shared feature engineering.

This module is imported by BOTH train.py and app.py. That's deliberate:
if training builds a feature one way and the serving app builds it a
different way, the model silently sees different inputs at inference time
than it saw during training ("train/serve skew"). Keeping one function
that both sides call is the standard fix.
"""

import numpy as np
import pandas as pd

# Columns the model is actually trained on, in order. Anything not in
# here (name, street, cc_num, trans_num, raw dob/timestamp, etc.) is
# either PII, an identifier with no predictive value, or already
# converted into a derived feature below.
NUMERIC_FEATURES = ["amt_log", "distance_km", "city_pop_log", "age", "hour", "day_of_week"]
CATEGORICAL_FEATURES = ["category"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_fraud"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between cardholder and merchant location.

    Distance matters for fraud: a card physically in Mumbai suddenly being
    used 3000km away is a much stronger signal than the raw lat/long values
    on their own, which is why we engineer this instead of feeding raw
    coordinates to the model.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0 * c


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw transaction columns into model-ready features.

    Expects raw columns: trans_date_trans_time, dob, amt, city_pop,
    lat, long, merch_lat, merch_long, category (is_fraud optional).
    Returns a new dataframe containing only ALL_FEATURES (+ target if present).
    """
    out = pd.DataFrame(index=df.index)

    trans_time = pd.to_datetime(df["trans_date_trans_time"])
    out["hour"] = trans_time.dt.hour
    out["day_of_week"] = trans_time.dt.dayofweek

    dob = pd.to_datetime(df["dob"])
    out["age"] = ((trans_time - dob).dt.days / 365.25).round(1)

    # Log-transform skewed monetary/population values so a handful of
    # extreme outliers don't dominate distance-based or linear models.
    out["amt_log"] = np.log1p(df["amt"].astype(float))
    out["city_pop_log"] = np.log1p(df["city_pop"].astype(float))

    out["distance_km"] = haversine_km(
        df["lat"].astype(float), df["long"].astype(float),
        df["merch_lat"].astype(float), df["merch_long"].astype(float),
    )

    out["category"] = df["category"].astype(str)

    if TARGET in df.columns:
        out[TARGET] = df[TARGET].astype(int)

    return out

# Fraud Radar — Credit Card Fraud Detection

A production-oriented version of the fraud detection
project: honest evaluation methodology, real feature engineering, a
saved model artifact instead of retraining on every run, and a Streamlit
app with per-prediction explainability.

 **[Live Demo](https://fraud-radar-swami.streamlit.app)**

 ![Fraud Radar demo](assets/demo1.png)
 ![Fraud Radar demo](assets/demo2.png)
 ![Fraud Radar demo](assets/demo3.png)

## Project structure

```
fraud_detection/
├── features.py                  # shared feature engineering (train + serve)
├── train.py                     # trains, evaluates, saves the model
├── app.py                       # Streamlit UI, loads the saved model
├── requirements.txt
├── data/fraudTrain.csv          # generated data lives here
└── models/
    ├── fraud_model.joblib       # the saved, trained pipeline
    └── metadata.json            # test metrics + run info
```

## How to run it

```bash
pip install -r requirements.txt
python3 generate_synthetic_data.py   # or drop in your real fraudTrain.csv
python3 train.py                     # trains + saves models/fraud_model.joblib
streamlit run app.py                 # launches the UI
```

## Results on this run

Logistic Regression was selected (highest PR-AUC on the held-out,
realistically imbalanced test set):

| Metric | Value |
|---|---|
| Precision | 0.23 |
| Recall | 0.86 |
| F1 | 0.36 |
| ROC-AUC | 0.97 |
| PR-AUC | 0.82 |

Random Forest was the close alternative: precision 0.76, recall 0.74,
F1 0.75. Both are in `models/metadata.json` — the notes below explain
why you'd pick one over the other.

---

### 1. The test set is no longer artificially balanced

**Before:** legitimate transactions were undersampled to match the
fraud count *before* the train/test split, so the test set was ~50/50.

**Now:** the split happens first, stratified on the real label. Only
the *training* data is rebalanced (via `class_weight='balanced'`, not
undersampling — more on that below). The test set keeps the real
~2% fraud rate.

**Why it matters:** a model evaluated on a
balanced test set gets scored on a different, easier problem than the
one it'll face in production. If someone asks *"why not just balance
the whole dataset"*, the answer is: doing so makes your test metrics
describe a world that doesn't exist at inference time. The only way to
know how a fraud model will actually perform is to test it on data
that looks like production traffic.

### 2. Used precision, recall, F1, ROC-AUC, PR-AUC

**Why:** with ~2% fraud (real-world fraud is often even rarer, ~0.1–0.5%),
a model that predicts "not fraud" every single time scores 98%+ accuracy
while catching zero fraud. Precision and recall are the metrics that
actually describe fraud-catching ability:
- **Recall** — of all real fraud, how much did we catch? Missing fraud
  costs the business/cardholder directly.
- **Precision** — of everything we flagged, how much was real fraud?
  Low precision means legitimate customers get blocked or annoyed.
- **PR-AUC** (average precision) is generally preferred over ROC-AUC
  for imbalanced problems, because ROC-AUC can look deceptively good
  even when precision is poor — it's diluted by the huge number of
  true negatives. That's why `train.py` selects the best model by PR-AUC.



### 3. Real feature engineering instead of dropping almost everything

**Before:** most columns were dropped, and the ones kept (`cc_num`,
raw `lat`/`long`, `zip`) were fed straight into a `OneHotEncoder`.

**Now**, in `features.py`:
- **`distance_km`** — haversine distance between cardholder and merchant
  location. A transaction happening 3,000km from where the card usually
  is used is a much stronger fraud signal than raw coordinates on their
  own.
- **`hour`, `day_of_week`** — extracted from the transaction timestamp.
  Fraud disproportionately happens at odd hours.
- **`age`** — computed from date of birth relative to transaction date.
- **`amt_log`, `city_pop_log`** — log-transformed, because both are
  heavily right-skewed (a few very large values would otherwise dominate
  a linear model's coefficients or distance-based calculations).
- **`category`** — kept as the one true categorical feature, one-hot
  encoded.
- Direct identifiers (`cc_num`, name, street, `trans_num`, `job`, raw
  `dob`/timestamp) are dropped — they're either PII with no predictive
  value, or already converted into a derived feature above.



### 4. Numeric and categorical features are now preprocessed correctly

**Before:** everything, including continuous numeric columns, went
through `OneHotEncoder`.

**Now:** a `ColumnTransformer` applies `StandardScaler` to numeric
features and `OneHotEncoder` only to `category`. One-hot encoding a
continuous variable like `amt` treats every unique dollar value as an
unrelated category — it throws away the fact that $50 and $51 are
close together, and blows up dimensionality for no benefit. Scaling
preserves that continuous relationship and is what linear/distance-based
models expect.

### 5. Models are tuned with cross-validation, not one arbitrary split

**Before:** `RandomForestClassifier(max_depth=2)` — an arbitrary,
untuned choice that likely explains why it originally underperformed
even a plain guess.

**Now:** `RandomizedSearchCV` with 5-fold stratified cross-validation,
optimizing for PR-AUC, searches over regularization strength for
Logistic Regression and depth/estimators/leaf size for Random Forest.
Cross-validation matters because a single train/test split is noisy —
your reported performance can swing a lot just from which rows
happened to land in the test set, especially with so few fraud
examples. Averaging over multiple folds gives a more reliable estimate
of how the model generalizes.

### 6. Class imbalance is handled with `class_weight='balanced'`, not undersampling

**Why the change:** undersampling throws away most of your legitimate
transaction data — that's real information the model never sees. Using
`class_weight='balanced'` instead makes the loss function penalize
mistakes on the minority class more heavily, without discarding any
data. It's not the only valid approach (SMOTE — synthetic oversampling
— is another common one), but it's a good default and simpler to
explain and reproduce.

### 7. The model is trained once and saved, not retrained on every app run

**Before:** the Streamlit script retrained a fresh model from scratch
every time it ran, and never actually applied that model to the form
inputs — it just printed `type(input_df)`.

**Now:** `train.py` is the offline training job. It saves the winning
pipeline — preprocessing *and* classifier together — to
`models/fraud_model.joblib` with `joblib.dump`. `app.py` only ever
calls `joblib.load` (cached with `@st.cache_resource` so it loads once
per session, not per request) and calls `.predict_proba()`. This
mirrors how real ML systems separate training (batch, offline, can
take minutes/hours) from serving (must respond in milliseconds).

### 8. Train and serve share one feature engineering function

`features.py`'s `engineer_features()` is imported by both `train.py`
and `app.py`. This avoids **train/serve skew** — a common real-world
bug where the training pipeline computes a feature one way and the
serving code computes it slightly differently (different rounding,
different timezone handling, a forgotten edge case), so the model
sees different-shaped inputs live than it learned from. Having one
function that both paths call eliminates that entire class of bug.

### 9. Explainability: the app shows why, not just a probability

Because the winning model is Logistic Regression, its coefficients are
directly interpretable. `explain_logistic()` in `app.py` multiplies
each standardized feature value by its coefficient to show which
factors pushed a specific prediction toward or away from fraud — a
simplified, model-specific stand-in for SHAP. This is worth mentioning
in an interview even if not asked: **for a linear model, coefficients
give you interpretability for free; for tree ensembles or anything
non-linear, you'd reach for SHAP or LIME instead**, since there's no
single coefficient per feature to point to.

---

## Honest limitations


- No live monitoring for **model drift** (fraud patterns change over
  time — a static model gets stale). In production you'd track
  prediction distributions and recall over time and retrain on a
  schedule or when performance degrades.
- No API layer — this is a Streamlit demo, not a horizontally-scaled
  service. A next step would be wrapping `models/fraud_model.joblib`
  behind a FastAPI endpoint for low-latency scoring at transaction time.
- The 0.5 decision threshold is a default, not a tuned business
  decision — in a real deployment you'd pick it based on the actual
  cost of false positives vs. false negatives.

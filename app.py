import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import difflib

st.set_page_config(page_title="Cars Playground — Price Predictor", layout="wide")

# -------------------------------------------------------
#                     IMAGE GALLERY
# -------------------------------------------------------
PHOTOS = [
    "https://images.unsplash.com/photo-1502877338535-766e1452684a?auto=format&fit=crop&w=1200&q=60",
    "https://images.unsplash.com/photo-1542362567-b07e54358753?auto=format&fit=crop&w=1200&q=60",
    "https://images.unsplash.com/photo-1503736334956-4c8f8e92946d?auto=format&fit=crop&w=1200&q=60",
    "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1200&q=60",
    "https://images.unsplash.com/photo-1549924231-f129b911e442?auto=format&fit=crop&w=1200&q=60",
    "https://images.unsplash.com/photo-1511919884226-fd3cad34687c?auto=format&fit=crop&w=1200&q=60",
    "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=1200&q=60",
    "https://images.unsplash.com/photo-1493238792000-8113da705763?auto=format&fit=crop&w=1200&q=60",
    "https://images.unsplash.com/photo-1525609004556-c46c7d6cf023?auto=format&fit=crop&w=1200&q=60"
]

st.markdown("<h1 style='text-align:center;'>Cars Playground</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9aa4b2;'>Stylish gallery — Predict selling price below</p>", unsafe_allow_html=True)

cols_per_row = 3
rows = (len(PHOTOS) + cols_per_row - 1) // cols_per_row

for r in range(rows):
    cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        idx = r * cols_per_row + c
        if idx < len(PHOTOS):
            cols[c].image(PHOTOS[idx], use_container_width=True)

st.markdown("---")

# -------------------------------------------------------
#            AUTO-LOAD MODEL + DATASET
# -------------------------------------------------------
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
except:
    SCRIPT_DIR = Path.cwd()

MODEL_PATH = SCRIPT_DIR / "cars.pkl"
CSV_PATH = SCRIPT_DIR / "cardekho_imputated.csv"

df = None
model = None

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
    st.success(f"Loaded Dataset: {CSV_PATH.name}")

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    st.success(f"Loaded Model: {MODEL_PATH.name}")

if df is None:
    st.error("Dataset not found. Keep cardekho_imputated.csv in the same folder.")
    st.stop()

if model is None:
    st.error("Model not found. Keep cars.pkl in the same folder.")
    st.stop()

# -------------------------------------------------------
#            DETECT TARGET COLUMN AUTOMATICALLY
# -------------------------------------------------------
def norm(x): return x.strip().lower().replace("_", " ")

target_col = None
preferred = "selling price"

norm_map = {norm(c): c for c in df.columns}

if norm(preferred) in norm_map:
    target_col = norm_map[norm(preferred)]
else:
    matches = difflib.get_close_matches(norm(preferred), list(norm_map.keys()), n=1, cutoff=0.6)
    if matches:
        target_col = norm_map[matches[0]]

if target_col is None:
    for col in df.columns:
        if "price" in col.lower():
            target_col = col
            break

if target_col is None:
    st.error("Could not detect target column 'selling price'.")
    st.stop()

st.write(f"### Target Column Detected: `{target_col}`")

feature_cols = [c for c in df.columns if c != target_col]

# -------------------------------------------------------
#     MODEL EXPECTED FEATURE NAMES (ONE-HOT HANDLING)
# -------------------------------------------------------
def get_expected_features(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        for _, step in reversed(model.steps):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def build_input(raw, model, df):
    expected = get_expected_features(model)

    if expected is None:
        return pd.DataFrame([raw])

    X = pd.DataFrame(0, index=[0], columns=expected)
    exp_low = [e.lower() for e in expected]

    for key, val in raw.items():

        # direct match
        if key in X.columns:
            X.at[0, key] = val
            continue

        # CI match
        if key.lower() in exp_low:
            idx = exp_low.index(key.lower())
            X.at[0, expected[idx]] = val
            continue

    # One-hot mapping
    for key, val in raw.items():
        if isinstance(val, (int, float)):
            continue

        prefix = key.lower() + "_"
        value_low = str(val).lower()

        for col in expected:
            col_low = col.lower()
            if col_low.startswith(prefix) and value_low in col_low:
                X.at[0, col] = 1

    return X

# -------------------------------------------------------
#                  INPUT FORM
# -------------------------------------------------------
st.header("Predict Selling Price")

left, right = st.columns(2)
user_inputs = {}

for i, col in enumerate(feature_cols):
    if pd.api.types.is_numeric_dtype(df[col]):
        default = float(df[col].median())
        inp = (left if i % 2 == 0 else right).number_input(col, value=default)
        user_inputs[col] = inp
    else:
        opts = df[col].dropna().unique().tolist()[:50]
        inp = (left if i % 2 == 0 else right).selectbox(col, options=opts)
        user_inputs[col] = inp

# -------------------------------------------------------
#                      PREDICT
# -------------------------------------------------------
if st.button("Predict"):
    X_new = build_input(user_inputs, model, df)

    try:
        raw_pred = model.predict(X_new)

        # --- Fix formatting error ---
        arr = np.asarray(raw_pred).ravel()
        pred_value = float(arr[0])

        st.success(f"### Predicted Selling Price: ₹{pred_value:,.2f}")

    except Exception as e:
        st.error("Prediction failed. Debug info:")
        st.exception(e)
        st.write("Model Expected Features:", get_expected_features(model))
        st.write("Input Sent to Model:")
        st.dataframe(X_new)

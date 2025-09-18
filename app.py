import io
import pickle
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# ---------- Utilities to make unpickling more robust ----------
# Some environments pickle models referring to 'numpy._core'.
# Provide a compatibility alias so unpickling doesn't fail.
try:
    import numpy as _np
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = _np.core
except Exception:
    # If this fails in your environment it's harmless; the load below may still work.
    pass


# ---------- Load model ----------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# ---------- Introspection helpers ----------
def _flatten_transformers(ct: ColumnTransformer) -> List[Tuple[str, Any, List[str]]]:
    flat = []
    for name, transformer, cols in ct.transformers:
        if transformer == "drop":
            continue
        if isinstance(cols, (list, tuple, np.ndarray)):
            cols = list(cols)
        elif isinstance(cols, str):
            cols = [cols]
        else:
            cols = list(cols) if cols is not None else []
        flat.append((name, transformer, cols))
    return flat


def _get_column_types_from_transformers(ct: ColumnTransformer) -> Dict[str, str]:
    col_types: Dict[str, str] = {}
    for name, transformer, cols in _flatten_transformers(ct):
        enc = transformer
        if isinstance(transformer, Pipeline) and len(transformer.steps) > 0:
            enc = transformer.steps[0][1]
        if isinstance(enc, OneHotEncoder):
            for c in cols:
                col_types[c] = "categorical"
        else:
            for c in cols:
                col_types.setdefault(c, "numeric")
    return col_types


def _get_ohe_categories(ct: ColumnTransformer) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for name, transformer, cols in _flatten_transformers(ct):
        enc = transformer
        if isinstance(transformer, Pipeline) and len(transformer.steps) > 0:
            enc = transformer.steps[0][1]
        if isinstance(enc, OneHotEncoder):
            if hasattr(enc, "categories_"):
                for c, col in zip(enc.categories_, cols):
                    result[col] = [str(x) for x in list(c)]
    return result


def get_expected_input_schema(model) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    if isinstance(model, Pipeline):
        preproc = None
        for _, step in model.steps:
            if isinstance(step, ColumnTransformer):
                preproc = step
                break
            if isinstance(step, Pipeline):
                for _, inner in step.steps:
                    if isinstance(inner, ColumnTransformer):
                        preproc = inner
                        break
        if preproc is None:
            raise ValueError("No ColumnTransformer found in the pipeline.")

        required_cols: List[str] = []
        for _, _, cols in _flatten_transformers(preproc):
            required_cols.extend(cols)

        type_map = _get_column_types_from_transformers(preproc)
        cat_map = _get_ohe_categories(preproc)

        if not required_cols:
            if hasattr(model, "feature_names_in_"):
                required_cols = list(model.feature_names_in_)
            else:
                raise ValueError(
                    "Could not infer required input columns. Please provide a CSV with the original training column names."
                )

        return required_cols, type_map, cat_map

    raise ValueError("Expected an sklearn Pipeline object.")


def coerce_types(df: pd.DataFrame, type_map: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for col, t in type_map.items():
        if col not in df.columns:
            continue
        if t == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str)
    return df


# ---------- Streamlit app ----------
st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗", layout="centered")
st.title("🚗 Used Car Price Prediction")

MODEL_PATH = "best_model_GradientBoosting.pickle"

with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH)

try:
    required_cols, type_map, cat_map = get_expected_input_schema(model)
except Exception as e:
    st.error(f"Could not introspect model input schema: {e}")
    st.stop()

with st.expander("Expected input columns (from the trained pipeline)", expanded=False):
    st.write(pd.DataFrame({
        "column": required_cols,
        "type (best-effort)": [type_map.get(c, "unknown") for c in required_cols],
        "known categories (if any)": [", ".join(cat_map.get(c, [])) for c in required_cols],
    }))

st.markdown("### Enter values below **or** upload a CSV with exactly these columns.")

# ---- Option A: form entry for a single prediction ----
with st.form("single_inference"):
    st.subheader("Single prediction (form)")

    form_values = {}
    for col in required_cols:
        inferred_type = type_map.get(col, "numeric")
        if inferred_type == "categorical":
            if col in cat_map and len(cat_map[col]) > 0:
                default = cat_map[col][0]
                form_values[col] = st.selectbox(col, cat_map[col], index=0)
            else:
                form_values[col] = st.text_input(col, value="")
        else:
            # Numeric input
            if col in ["km_driven", "car_age"]:
                form_values[col] = st.number_input(col, min_value=0, step=1, format="%d")
            else:
                form_values[col] = st.number_input(col, value=0.0, step=1.0, format="%.6f")

    submitted = st.form_submit_button("Predict")
    if submitted:
        input_df = pd.DataFrame([form_values], columns=required_cols)
        input_df = coerce_types(input_df, type_map)

        missing = [c for c in required_cols if c not in input_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if input_df.isna().any().any():
                st.warning("Some numeric fields were invalid and coerced to NaN; the model/imputer will handle them if configured.")
            try:
                pred = model.predict(input_df)[0]
                st.success(f"**Predicted price:** {float(pred):,.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.divider()

# ---- Option B: batch prediction from CSV ----
st.subheader("Batch prediction (CSV)")
uploaded = st.file_uploader("Upload a CSV with exactly the expected columns (case-sensitive column names).", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    missing_cols = [c for c in required_cols if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in required_cols]

    if missing_cols:
        st.error(f"CSV is missing required columns: {missing_cols}")
        st.info("Tip: Rename columns to match the expected names shown above.")
    else:
        if extra_cols:
            st.warning(f"CSV has extra columns that will be ignored: {extra_cols}")
            df = df[required_cols]

        df = coerce_types(df, type_map)

        try:
            preds = model.predict(df)
            out = df.copy()
            out["prediction"] = preds
            st.success("Predictions complete.")
            st.dataframe(out.head(20))

            buf = io.BytesIO()
            out.to_csv(buf, index=False)
            st.download_button(
                "Download predictions as CSV",
                data=buf.getvalue(),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption(
    "If you still see a 'columns are missing' error, double-check that your column **names** and **types** match the expected inputs above. The pipeline’s ColumnTransformer requires an exact match to the training schema."
)

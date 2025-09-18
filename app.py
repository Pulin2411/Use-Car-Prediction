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
    """
    Returns a flat list of (name, transformer, columns) for a ColumnTransformer.
    If a transformer is itself a Pipeline whose first step is a transformer,
    we still surface its columns.
    """
    flat = []
    for name, transformer, cols in ct.transformers:
        # Skip 'drop' and 'remainder'
        if transformer == "drop":
            continue
        if isinstance(cols, (list, tuple, np.ndarray)):
            cols = list(cols)
        elif isinstance(cols, str):
            cols = [cols]
        else:
            # Unknown spec; best effort
            cols = list(cols) if cols is not None else []

        flat.append((name, transformer, cols))
    return flat


def _get_column_types_from_transformers(ct: ColumnTransformer) -> Dict[str, str]:
    """
    Best-effort guess: mark columns passed into OneHotEncoder as 'categorical',
    and everything else as 'numeric' unless ambiguous.
    """
    col_types: Dict[str, str] = {}
    for name, transformer, cols in _flatten_transformers(ct):
        enc = transformer
        # If wrapped in a Pipeline, unwrap to the first actual transformer
        if isinstance(transformer, Pipeline) and len(transformer.steps) > 0:
            enc = transformer.steps[0][1]

        if isinstance(enc, OneHotEncoder):
            for c in cols:
                col_types[c] = "categorical"
        else:
            # If we haven't already marked it categorical, assume numeric.
            for c in cols:
                col_types.setdefault(c, "numeric")
    return col_types


def _get_ohe_categories(ct: ColumnTransformer) -> Dict[str, List[str]]:
    """
    If OneHotEncoder(s) are fitted, expose per-column category lists.
    """
    result: Dict[str, List[str]] = {}
    for name, transformer, cols in _flatten_transformers(ct):
        enc = transformer
        if isinstance(transformer, Pipeline) and len(transformer.steps) > 0:
            enc = transformer.steps[0][1]

        if isinstance(enc, OneHotEncoder):
            # categories_ exists only after fit
            if hasattr(enc, "categories_"):
                for c, col in zip(enc.categories_, cols):
                    # Cast to str for Streamlit selectbox stability
                    result[col] = [str(x) for x in list(c)]
    return result


def get_expected_input_schema(model) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    """
    Returns:
      - required original input column names (order preserved per transformer blocks)
      - a best-effort type map {col: 'numeric'|'categorical'}
      - known categories for categoricals (if OHE is fitted) {col: [c]()

<img width="811" height="438" alt="image" src="https://github.com/user-attachments/assets/dffbc02a-b7a4-4019-b5e9-363696176185" />

# Used Car Price Prediction

A complete, reproducible pipeline to train, evaluate, and deploy a Used Car Price Prediction model using scikit-learn. It includes data cleaning, exploratory plots, model comparison with hyperparameter tuning, feature importance, model persistence (.pickle), and a Flask app for web and API inference.

## Highlights
- End-to-end pipeline in Python with clear, modular functions.
- Automatic dataset discovery within a configured data directory.
- Robust data cleaning (price parsing with lakh/crore handling, km normalization, year coercion, outlier trimming).
- EDA plots saved to disk (distribution, price vs. age, price vs. km).
- Multiple models compared: LinearRegression, Ridge, RandomForest, GradientBoosting, plus tuned RandomForest (GridSearchCV).
- Best model saved as a single .pickle pipeline for portable inference.
- Flask app (app.py) exposing a simple UI and a JSON API for predictions.

## Project Structure
```
.
├── used_car_price_pipeline_pickle.py    # Train & evaluate; saves best model as .pickle
├── app.py                               # Flask app: HTML form + /api/predict endpoint
├── requirements.txt                     # Minimal dependencies
└── data/                                # (Optional) Place dataset CSVs here if not using Drive
```

Note: By default, the scripts point to a Google Drive path via `DATA_DIR`. Update this to a local `data/` folder (or any desired path) before running locally.

## Requirements
See `requirements.txt` for minimal package requirements:
- numpy
- pandas
- matplotlib
- scikit-learn
- Streamlit

Install:
```
pip install -r requirements.txt
```



## Data
Place a CSV file with used car listings in the configured `DATA_DIR`. The pipeline auto-detects the first CSV containing `car` in its filename (fallback to first CSV). Expected columns (variants are auto-mapped):
- name (car name)
- year
- selling_price (supports values in rupees or with lakh/crore)
- km_driven
- fuel
- seller_type
- transmission
- owner

The script standardizes column names and maps common variations automatically.

## Training Pipeline
Run the training script:
```
python used_car_price_pipeline_pickle.py
```
Key steps performed:
- Column standardization and cleaning (handles lakh/crore to rupees; coerces numeric types; removes duplicates; trims top 0.5% price outliers).
- Feature engineering: `car_age = CURRENT_YEAR - year`.
- Train/test split (80/20, random_state=42).
- Preprocessing: StandardScaler for numeric, OneHotEncoder for categorical features.
- Models evaluated: LinearRegression, Ridge, RandomForest, GradientBoosting.
- Hyperparameter tuning: RandomForest via GridSearchCV (`n_estimators`, `max_depth`, `min_samples_split`).
- Metrics reported: R2, MAE, and MSE (labeled as RMSE in legacy logs for continuity).
- Best model selection by lowest MSE and saved as `.pickle`.

Artifacts saved to `DATA_DIR`:
- `car_price_plots/price_distribution.png`
- `car_price_plots/price_vs_age.png`
- `car_price_plots/price_vs_km.png`
- `feature_importance.csv` (for tree-based best models)
- `best_model_<ModelName>.pickle`
- `car_price_pipeline_summary.txt` (dataset used, rows, features, best model + metrics, saved paths)

Environment variables:
- `DATA_DIR`: Folder containing the CSV(s) and where artifacts are saved. Default is the Google Drive path; set to a local folder as needed.
- `CURRENT_YEAR`: Used to compute `car_age`. Default: 2025.

## Inference App (Streamlit)
<img width="285" height="316" alt="image" src="https://github.com/user-attachments/assets/4f62def3-5666-49cf-823c-7e7c9d5aa029" />


Features:
- Auto-loads the most recent `best_model_*.pickle` from `DATA_DIR`, or use `MODEL_PATH` env var to point to a specific model.
- Web form for single prediction with inputs: Year, Km Driven, Fuel, Seller Type, Transmission, Owner.
- JSON API at `POST /api/predict` accepting a single JSON object or a list for batch predictions.

Example JSON request:
```
POST /api/predict
Content-Type: application/json

{
  "year": 2016,
  "km_driven": 52000,
  "fuel": "Petrol",
  "seller_type": "Individual",
  "transmission": "Manual",
  "owner": "First Owner"
}
```
Response:
```
{
  "predictions": [437215.42]
}
```

App environment variables:
- `DATA_DIR`: Directory to search for `best_model_*.pickle` if `MODEL_PATH` is not set.
- `MODEL_PATH`: Absolute path to a specific `.pickle` model file.
- `CURRENT_YEAR`: Year used to compute `car_age` in the app (default 2025).
- `PORT`: Port to run the Flask server (default 8000).

## Reproducibility Notes
- Random seeds: `random_state=42` is used for splits and certain models for consistent results.
- Versioning: Use the provided `requirements.txt` to keep package versions consistent.
- Serialization: The entire `Pipeline` is pickled to ensure preprocessing and model are stored together.

## Results Snapshot
Example metrics from a typical run (will vary by dataset split):
- GradientBoosting: R2≈0.58, MAE≈1.50e5, MSE≈5.05e10
- Tuned RandomForest: R2≈0.58, MAE≈1.50e5, MSE≈5.09e10

These values indicate moderate explanatory power on the sample dataset and are included in the generated summary file for transparency.

## Next Steps / Improvements
- Use `squared=False` for RMSE reporting and add cross-validation reporting across folds.
- Extend features (e.g., brand/model extraction from `name`, city/region, service history, engine specs).
- Try additional models: XGBoost/LightGBM/CatBoost with categorical handling.
- Calibrate predictions and/or apply quantile regression for interval estimates.
- Add model registry and experiment tracking (e.g., MLflow) and containerize the app.

### About Author
Pulin Shah — Lead IT Support Coordinator | IT Service Delivery | n8n Automations |ML|DL|Data Science|Prompt Enigneering|RAG|Gen-AI|EDA

IT service delivery leader with 20+ years across incident, change, and problem management, PSA/RMM operations (ConnectWise), and endpoint security. Designs and operates Service Desk workflows with SLA rigor, and builds support automations using n8n, LLMs, and vector search for policy-aligned responses.

Leads Service Desk operations: ticket triage, scheduling, vendor coordination, SLA governance, and reporting.

Builds LLM-powered assistants on Telegram with strict topic guardrails and end-of-conversation flows.

Implements RAG pipelines: Google Drive ingestion, OpenAI embeddings (512-dim), Pinecone indexing, and chunking strategies for retrieval quality.

Experienced with ESET/SentinelOne endpoints, Microsoft stack, AWS (Solutions Architect), and ITIL-based practices for change/incident/request management.

Links

GitHub: https://github.com/Pulin2411

LinkedIn: linkedin.com/in/pulin-shah-741212b

Email: pulin2411@gmail.com

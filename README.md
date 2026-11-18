# RAKSHA AI — Road Risk Intelligence & Near-Miss Prediction

GIS-based crash risk mapping combined with causal machine learning for early near-miss prediction.

## Key Highlights
- **Real-Traffic ML Benchmark (Argoverse 2):** Evaluated across `4,988` scenarios (`304,988` road-user pairs, `1,956,954` causal anchors at 10 Hz). XGBoost trained on 27 causal features (TTC, projected PET, DRAC, kinematics, and context over a 1.5 s window) achieves **0.966 ROC-AUC** and **0.067 AP** (`57x` random) versus **0.745 ROC-AUC** and **0.016 AP** for a TTC threshold rule, keeping strong performance up to 3 s before near-miss onset (`0.938` AUC at `2–3 s`).
- **Multi-Country GIS Mapping:** Processes `217,160` crash records (`210,300` real records from New York, Chicago, and UK STATS19 + `6,860` simulated demo records across 8 Indian cities) into `~150 m` grid cells with severity weighting (`1 + 2*injured + 10*killed`), identifying `296` blackspots and `100` emerging risk zones.
- **OAuth 2.0 + OpenID Connect Authentication:** Built-in Streamlit native OIDC integration (`st.login`, `st.logout`, `st.user`) with PKCE (`S256`) and `RS256` JWT `id_token` verification, backed by `src/oidc_provider.py` (or drop-in Google Cloud OAuth 2.0).

## Quick Start

1. **Create environment & install dependencies**
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure OAuth 2.0 + OpenID Connect**
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

3. **Run the platform**
   ```bash
   streamlit run app/raksha.py
   ```
   *(The app automatically ensures the local OIDC Identity Provider `src/oidc_provider.py` is running on port `8502` when configured in `.streamlit/secrets.toml`.)*

## Re-running the Offline Pipeline
```bash
set PYTHONPATH=src
python src/fetch_cities.py     # Download/harmonise crash records (NYC, Chicago, UK STATS19) + India demo
python src/simulate.py         # Generate 6,000 simulated mixed-traffic interactions
python src/experiments.py      # Run Experiments 1-4, SHAP explanations, and save model
python src/build_gis.py        # Aggregate 150 m grid cells, blackspots, emerging zones, and reports
```

## Project Structure
- `app/raksha.py` — Streamlit platform (Research Landing Page, OAuth 2.0/OIDC auth, GIS Map, Model Workbench + Live Conflict Simulator, Research Hub, Community Reporting, Authority Triage, Crash Analytics)
- `app/report_content.py` — Structured project reference tables, feature dictionary, and presets
- `src/oidc_provider.py` — OpenID Connect Identity Provider (`/.well-known/openid-configuration`, RSA JWKS, PKCE `S256`, `RS256` ID tokens)
- `src/features.py` — Causal sliding-window TTC, projected PET, DRAC, and kinematic feature extraction
- `src/experiments.py` — Model training, ablations, lead-time analysis, and TreeSHAP attributions
- `src/build_gis.py` — 150 m spatial grid aggregation, blackspots, and emerging zone clustering
- `src/detect_track.py` — YOLO + ByteTrack roadside video trajectory extraction extension

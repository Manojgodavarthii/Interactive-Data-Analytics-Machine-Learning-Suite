import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.utils import detect_column_types, render_chart
from modules.ai_engine import forecast_insight


def _style(fig, height=400):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=20, t=55, b=40),
        height=height,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
    return fig


def _insight_box(text, color="#0f3460"):
    return (
        f'<div style="background:{color}0d;border-left:4px solid {color};'
        f'border-radius:0 12px 12px 0;padding:0.8rem 1rem;margin-bottom:0.6rem;'
        f'font-size:0.88rem;color:#374151;">{text}</div>'
    )


def _train_predictive_model(df, target_col, problem_type="regression"):
    X = df.drop(columns=[target_col]).dropna(how="all")
    y = df.loc[X.index, target_col].dropna()
    X = X.loc[y.index]

    if len(X) < 10:
        return None, "Dataset has fewer than 10 complete rows for training."

    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    if not num_cols and not cat_cols:
        return None, "No valid predictor columns available."

    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)

    if problem_type != "regression":
        y = y.astype(str)

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    best_pipe = None
    best_score = -999999

    if problem_type == "regression":
        models = {
            "Ridge Baseline": Ridge(alpha=1.0),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42)
        }
        for name, m in models.items():
            try:
                pipe = Pipeline(steps=[('prep', preprocessor), ('model', m)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                r2 = float(r2_score(y_test, preds))
                results[name] = {"RMSE": round(rmse, 4), "R2 Score": round(r2, 4), "pipeline": pipe}
                if r2 > best_score:
                    best_score = r2
                    best_pipe = pipe
            except Exception as e:
                results[name] = {"RMSE": "N/A", "R2 Score": "N/A", "error": str(e)}
    else:
        models = {
            "Logistic Regression Baseline": LogisticRegression(max_iter=500),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42)
        }
        for name, m in models.items():
            try:
                pipe = Pipeline(steps=[('prep', preprocessor), ('model', m)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                acc = float(accuracy_score(y_test, preds))
                f1 = float(f1_score(y_test, preds, average='weighted'))
                results[name] = {"Accuracy": round(acc, 4), "F1 Score": round(f1, 4), "pipeline": pipe}
                if acc > best_score:
                    best_score = acc
                    best_pipe = pipe
            except Exception as e:
                results[name] = {"Accuracy": "N/A", "F1 Score": "N/A", "error": str(e)}

    return best_pipe, results


import uuid as _uuid

_BG_RESULTS = {}
_BG_STATUS = {}

# Training must stay fast regardless of dataset size. Cap rows, columns and
# category cardinality so OneHotEncoder never explodes in memory / time.
_MAX_TRAIN_ROWS = 1500
_MAX_TRAIN_FEATURES = 15
_MAX_CATEGORIES = 20
_NUM_ESTIMATORS = 20
_TRAIN_TIMEOUT_SECONDS = 90


def _prepare_training_frame(df):
    """Subsample, cap cardinality and cap feature count so training is fast and bounded."""
    if df is None or df.empty or len(df) < 5:
        return df
    work = df
    if len(work) > _MAX_TRAIN_ROWS:
        work = work.sample(n=_MAX_TRAIN_ROWS, random_state=42)
    work = work.copy()

    if len(work.columns) > _MAX_TRAIN_FEATURES + 1:
        # Keep the last column (likely the target) plus the first N predictor
        # columns; drop the rest so the preprocessor stays tiny.
        keep = list(work.columns[-1:]) + list(work.columns[: _MAX_TRAIN_FEATURES])
        work = work[keep]

    for c in work.columns:
        if not pd.api.types.is_numeric_dtype(work[c]):
            s = work[c].astype(str)
            vc = s.value_counts()
            if len(vc) > _MAX_CATEGORIES:
                top = vc.head(_MAX_CATEGORIES).index
                work[c] = s.where(s.isin(top), "Other")
    return work


def _fit_default_model(df):
    """Fit a small default model. Returns a result dict or None."""
    if df is None or df.empty or len(df) < 5:
        return None
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

        df = _prepare_training_frame(df)
        if df is None or len(df) < 5:
            return None

        all_cols = list(df.columns)
        num_cols_all = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        target_col = num_cols_all[-1] if num_cols_all else all_cols[-1]
        feature_cols = [c for c in all_cols if c != target_col]
        if not feature_cols:
            return None

        is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
        unique_vals = df[target_col].nunique()
        problem_type = "Regression" if (is_numeric and unique_vals > 10) else "Classification"

        clean_df = df[[target_col] + feature_cols].dropna()
        if len(clean_df) < 5:
            clean_df = df[[target_col] + feature_cols].copy()

        X = clean_df[feature_cols].copy()
        y = clean_df[target_col].copy()

        num_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        cat_cols = [c for c in feature_cols if c not in num_cols]

        for c in cat_cols:
            X[c] = X[c].astype(str)

        if problem_type != "Regression":
            y = y.astype(str)

        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ],
            remainder='drop'
        )

        if problem_type == "Regression":
            model = RandomForestRegressor(n_estimators=_NUM_ESTIMATORS, max_depth=5, random_state=42, n_jobs=-1)
        else:
            model = HistGradientBoostingClassifier(max_iter=_NUM_ESTIMATORS, random_state=42)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        metrics = {}
        if problem_type == "Regression":
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            metrics = {"RMSE": f"{rmse:.4f}", "R2": f"{r2:.4f}"}
        else:
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            metrics = {"Accuracy": f"{acc:.2%}", "F1": f"{f1:.4f}"}

        return {
            "pipeline": pipeline,
            "features": feature_cols,
            "target": target_col,
            "problem_type": problem_type,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "metrics": metrics,
        }
    except Exception:
        return None


def start_background_training(df):
    """Train the default model immediately (bounded to be fast) and return a
    token that is always in a terminal state — never stuck in 'training'."""
    token = str(_uuid.uuid4())
    result = _fit_default_model(df)
    _BG_RESULTS[token] = result
    _BG_STATUS[token] = "ready" if result is not None else "error"
    return token


def get_bg_status(token):
    return _BG_STATUS.get(token) if token else None


def get_bg_result(token):
    return _BG_RESULTS.get(token) if token else None


def auto_train_default_model(df):
    """Back-compat wrapper: fit the default model synchronously and store into session state."""
    result = _fit_default_model(df)
    if result is None:
        return
    st.session_state['ml_pipeline'] = result["pipeline"]
    st.session_state['ml_features'] = result["features"]
    st.session_state['ml_target'] = result["target"]
    st.session_state['ml_problem_type'] = result["problem_type"]
    st.session_state['num_cols'] = result["num_cols"]
    st.session_state['cat_cols'] = result["cat_cols"]
    st.session_state['ml_metrics'] = result["metrics"]
    st.session_state['ml_auto_trained'] = True


def render_prediction_module(df=None):
    if df is None:
        df = st.session_state.get("df")

    st.markdown('<div class="section-title">🔮 Machine Learning & Interactive Predictions</div>', unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("Please upload a dataset first.")
        return

    # 1. Ensure a background model is training for this dataset
    token = st.session_state.get("ml_train_token")
    if not token:
        token = start_background_training(df)
        st.session_state["ml_train_token"] = token
        st.rerun()
        return

    status = get_bg_status(token)
    result = get_bg_result(token)

    if status == "error":
        st.error("The background model could not be trained on this dataset. Try uploading a different dataset.")
        if st.button("🔄 Retry Background Training", type="primary", key="pred_retry_bg"):
            st.session_state.pop("ml_train_token", None)
            st.rerun()
        return

    if status != "ready" or result is None:
        # 2. Training still in progress — auto-refresh so prediction unlocks on its own
        @st.fragment(run_every=2.0)
        def _training_status():
            _status = get_bg_status(st.session_state.get("ml_train_token"))
            if _status == "ready":
                st.rerun(scope="app")
                return
            if _status == "error":
                st.rerun(scope="app")
                return
            st.markdown(
                '<div style="background:linear-gradient(135deg,#eef2ff,#e0e7ff);border:1px solid #c7d2fe;border-radius:16px;'
                'padding:1.4rem 1.6rem;margin:1rem 0 1.4rem 0;text-align:center;">'
                '<div style="font-size:1.9rem;margin-bottom:0.4rem;">⏳</div>'
                '<div style="font-weight:800;font-size:1.15rem;color:#1e1b4b;">Training model in the background…</div>'
                '<div style="font-size:0.92rem;color:#6366f1;margin-top:0.3rem;font-weight:600;">You can keep using the rest of the app. Prediction unlocks automatically once training finishes.</div>'
                '</div>',
                unsafe_allow_html=True
            )
            st.progress(0.35, text="Fitting default ML pipeline (auto-refreshing)…")
            st.info("💡 Tip: You can browse the other analysis sections while the model trains. This page refreshes automatically every 2 seconds.")

        _training_status()
        return

    pipeline = result["pipeline"]
    feature_cols = result["features"]
    target_col = result["target"]
    problem_type = result["problem_type"]
    num_cols = result["num_cols"]
    cat_cols = result["cat_cols"]
    metrics = result["metrics"]

    # 3. Model status card
    m_color = "#22c55e" if problem_type == "Regression" else "#6366f1"
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#ffffff,#f0f4ff);border:1px solid #c7d2fe;border-radius:16px;'
        f'padding:1rem 1.3rem;margin:0.2rem 0 1.2rem 0;box-shadow:0 4px 16px rgba(99,102,241,0.10);">'
        f'<div style="display:flex;align-items:center;gap:0.6rem;flex-wrap:wrap;">'
        f'<span style="background:linear-gradient(135deg,#4f46e5,#6366f1);color:#fff;font-weight:800;font-size:0.78rem;padding:0.35rem 0.8rem;border-radius:20px;">✅ MODEL READY</span>'
        f'<span style="font-weight:800;color:#0f172a;font-size:1.05rem;">Target: {target_col}</span>'
        f'<span style="margin-left:auto;font-weight:700;color:{m_color};font-size:1.05rem;">{problem_type}</span>'
        f'</div>'
        f'<div style="display:flex;gap:1.4rem;flex-wrap:wrap;margin-top:0.6rem;font-size:0.9rem;color:#475569;">'
        f'<span>📊 {len(feature_cols)} features</span>'
        f'<span>{" · ".join(f"{k}: <strong>{v}</strong>" for k, v in metrics.items())}</span>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='color:#64748b;font-size:0.98rem;margin-bottom:1rem;'>Provide your custom data below — either enter values manually or upload a file — to get instant predictions from the pre-trained model.</div>", unsafe_allow_html=True)

    # 4. Prediction input modes: manual entry OR batch file upload
    tab_manual, tab_batch = st.tabs(["✍️ Enter Data Manually", "📁 Upload File for Batch Predictions"])

    # ─── Manual entry ─────────────────────────────────────────────────────
    with tab_manual:
        st.markdown('<div style="font-weight:800;font-size:1.15rem;color:#0f172a;margin:0.4rem 0 0.6rem 0;">Enter values for each feature</div>', unsafe_allow_html=True)
        user_inputs = {}
        cols = st.columns(2)

        for idx, feat in enumerate(feature_cols):
            col = cols[idx % 2]
            with col:
                if feat in num_cols:
                    s_feat = df[feat].dropna()
                    default_val = float(s_feat.median()) if len(s_feat) > 0 else 0.0
                    min_val = float(s_feat.min()) if len(s_feat) > 0 else 0.0
                    max_val = float(s_feat.max()) if len(s_feat) > 0 else 100.0
                    if min_val >= max_val:
                        max_val = min_val + 1.0
                    user_inputs[feat] = st.number_input(f"🔢 {feat}", min_value=min_val, max_value=max_val, value=default_val, key=f"pm_num_{feat}")
                else:
                    opts = list(df[feat].dropna().unique()) if feat in df.columns else ["Unknown"]
                    if not opts:
                        opts = ["Unknown"]
                    user_inputs[feat] = st.selectbox(f"🏷️ {feat}", opts, key=f"pm_cat_{feat}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Run Prediction on My Values", type="primary", key="pm_btn_predict", use_container_width=True):
            try:
                input_df = pd.DataFrame([user_inputs])
                for c in cat_cols:
                    if c in input_df.columns:
                        input_df[c] = input_df[c].astype(str)
                pred = pipeline.predict(input_df)[0]

                st.markdown("---")
                if problem_type == "Regression":
                    st.markdown(
                        f'<div style="background:linear-gradient(135deg,#0f172a,#312e81);border-radius:18px;padding:1.6rem 2rem;'
                        f'text-align:center;color:#ffffff;margin:0.8rem 0;box-shadow:0 12px 35px rgba(49,46,129,0.35);">'
                        f'<div style="font-size:0.85rem;font-weight:700;letter-spacing:1px;color:rgba(255,255,255,0.6);text-transform:uppercase;">Predicted {target_col}</div>'
                        f'<div style="font-size:2.6rem;font-weight:900;color:#a5b4fc;margin-top:0.2rem;">{pred:,.2f}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="background:linear-gradient(135deg,#0f172a,#312e81);border-radius:18px;padding:1.6rem 2rem;'
                        f'text-align:center;color:#ffffff;margin:0.8rem 0;box-shadow:0 12px 35px rgba(49,46,129,0.35);">'
                        f'<div style="font-size:0.85rem;font-weight:700;letter-spacing:1px;color:rgba(255,255,255,0.6);text-transform:uppercase;">Predicted Class for {target_col}</div>'
                        f'<div style="font-size:2.4rem;font-weight:900;color:#a5b4fc;margin-top:0.2rem;">{pred}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ─── Batch file upload ────────────────────────────────────────────────
    with tab_batch:
        st.markdown('<div style="font-weight:800;font-size:1.15rem;color:#0f172a;margin:0.4rem 0 0.6rem 0;">Upload a test dataset (same feature columns)</div>', unsafe_allow_html=True)
        batch_file = st.file_uploader("📂 Drop CSV / Excel test file here for batch predictions", type=["csv", "xlsx", "xls"], key="uplo_batch_file_pred")
        if batch_file:
            try:
                from modules.utils import read_dataset
                test_df = read_dataset(batch_file)
                if test_df is not None:
                    needed_feats = feature_cols
                    missing_in_test = [f for f in needed_feats if f not in test_df.columns]
                    if missing_in_test:
                        st.warning(f"Note: Uploaded file is missing {len(missing_in_test)} feature column(s): {missing_in_test}. Filling with dataset defaults...")
                        for mf in missing_in_test:
                            if mf in num_cols:
                                test_df[mf] = float(df[mf].median()) if mf in df.columns else 0.0
                            else:
                                test_df[mf] = str(list(df[mf].dropna().unique())[0]) if mf in df.columns else "Unknown"

                    X_batch = test_df[needed_feats].copy()
                    for c in cat_cols:
                        if c in X_batch.columns:
                            X_batch[c] = X_batch[c].astype(str)

                    with st.spinner("Running batch predictions…"):
                        batch_preds = pipeline.predict(X_batch)
                    res_df = test_df.copy()
                    res_df[f"Predicted_{target_col}"] = batch_preds

                    st.success(f"🎉 Successfully generated predictions for {len(test_df):,} test rows!")
                    st.dataframe(res_df.head(25), use_container_width=True)

                    csv_batch = res_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Batch Predictions CSV",
                        csv_batch,
                        f"batch_predictions_{batch_file.name}.csv",
                        "text/csv",
                        type="primary",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")


def render():
    from scipy import stats as scipy_stats
    st.markdown('<div class="section-title">🚀 Advanced Analytics</div>', unsafe_allow_html=True)
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.df
    col_types = detect_column_types(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    date_cols = [c for c, t in col_types.items() if t == "date"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    tab_trend, tab_ts, tab_forecast, tab_cluster, tab_compare, tab_anomaly, tab_rolling, tab_growth, tab_whatif = st.tabs([
        "📈 Trend", "📅 Time Series", "🔮 Forecast", "🔵 Clusters",
        "⚖️ Compare", "🚨 Anomaly", "🔄 Rolling Stats", "📊 Growth Rate", "🎯 What-If"
    ])

    # ─── TAB 1: Trend Analysis ─────────────────────────────────────────────
    with tab_trend:
        if not num_cols:
            st.warning("No numeric columns.")
        else:
            col = st.selectbox("Select column", num_cols, key="trend_col")
            s = df[col].dropna()
            # Compute trend via linear regression
            x_vals = np.arange(len(s))
            slope, intercept, r_val, p_val, _ = scipy_stats.linregress(x_vals, s.values)
            trend_line = slope * x_vals + intercept

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=s.values, mode="lines", name=col,
                                     line=dict(color="#6366f1", width=1.5)))
            fig.add_trace(go.Scatter(y=trend_line, mode="lines", name="Trend",
                                     line=dict(color="#f59e0b", width=2, dash="dash")))
            fig.update_layout(title=f"Trend Analysis — {col}", showlegend=True)
            _style(fig)
            render_chart(fig, "trend")

            direction = "📈 Increasing" if slope > 0 else "📉 Decreasing"
            strength = "strong" if abs(r_val) > 0.7 else "moderate" if abs(r_val) > 0.4 else "weak"
            st.markdown(_insight_box(
                f"<strong>Trend:</strong> {direction} at {abs(slope):.4f} units/step. "
                f"R²={r_val**2:.3f} ({strength} trend). "
                f"{'Statistically significant (p<0.05).' if p_val < 0.05 else 'Not statistically significant (p≥0.05).'}"
            , "#6366f1"), unsafe_allow_html=True)

    # ─── TAB 2: Time-Series ────────────────────────────────────────────────
    with tab_ts:
        if not date_cols or not num_cols:
            st.warning("Need both date and numeric columns for time-series analysis.")
        else:
            dcol = st.selectbox("Date column", date_cols, key="ts_dcol")
            ncol = st.selectbox("Value column", num_cols, key="ts_ncol")
            freq_map = {"Daily (D)": "D", "Weekly (W)": "W", "Monthly (ME)": "ME",
                        "Quarterly (QE)": "QE", "Yearly (YE)": "YE"}
            freq = st.selectbox("Resample frequency", list(freq_map.keys()), key="ts_freq")

            ts_df = df[[dcol, ncol]].dropna().copy()
            ts_df[dcol] = pd.to_datetime(ts_df[dcol])
            ts_df = ts_df.sort_values(dcol).set_index(dcol)
            resampled = ts_df.resample(freq_map[freq]).agg(["mean", "min", "max"]).reset_index()
            resampled.columns = [dcol, "Mean", "Min", "Max"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=resampled[dcol], y=resampled["Max"], mode="lines",
                                     name="Max", line=dict(color="#ef4444", width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=resampled[dcol], y=resampled["Mean"], mode="lines",
                                     name="Mean", line=dict(color="#6366f1", width=2),
                                     fill="tonexty", fillcolor="rgba(99,102,241,0.07)"))
            fig.add_trace(go.Scatter(x=resampled[dcol], y=resampled["Min"], mode="lines",
                                     name="Min", line=dict(color="#22c55e", width=1, dash="dot"),
                                     fill="tonexty", fillcolor="rgba(34,197,94,0.05)"))
            fig.update_layout(title=f"Time Series — {ncol} ({freq})", showlegend=True)
            _style(fig, height=430)
            render_chart(fig, "time_series")
            st.dataframe(resampled, use_container_width=True, hide_index=True)

    # ─── TAB 3: Forecast ──────────────────────────────────────────────────
    with tab_forecast:
        if not num_cols:
            st.warning("No numeric columns available for forecasting.")
        else:
            st.markdown(
                '<div style="font-size:0.88rem;color:#6b7280;margin-bottom:1rem;">'
                'Forecast uses <strong>Ordinary Least Squares (OLS) linear regression</strong> '
                'to project future values based on historical trends. Best for data with a clear linear trend.</div>',
                unsafe_allow_html=True,
            )
            col_f = st.selectbox("Select column to forecast", num_cols, key="fc_col")
            future_n = st.slider("Periods to forecast", 5, 100, 20, key="fc_periods")

            s_f = df[col_f].dropna().reset_index(drop=True)
            x_f = np.arange(len(s_f))
            slope_f, intercept_f, r_val_f, _, _ = scipy_stats.linregress(x_f, s_f.values)

            # Forecast
            future_x = np.arange(len(s_f), len(s_f) + future_n)
            future_y = slope_f * future_x + intercept_f
            trend_y = slope_f * x_f + intercept_f

            # Confidence interval (95%)
            residuals = s_f.values - trend_y
            se = residuals.std()
            ci_upper = future_y + 1.96 * se
            ci_lower = future_y - 1.96 * se

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=s_f.values, mode="lines", name="Historical",
                                     line=dict(color="#6366f1", width=2)))
            fig.add_trace(go.Scatter(y=trend_y, mode="lines", name="Trend (fit)",
                                     line=dict(color="#0f3460", width=1.5, dash="dash")))
            fig.add_trace(go.Scatter(
                x=list(range(len(s_f), len(s_f) + future_n)),
                y=ci_upper, mode="lines", name="95% CI Upper",
                line=dict(color="rgba(239,68,68,0.3)", width=0),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(s_f), len(s_f) + future_n)),
                y=ci_lower, mode="lines", name="95% CI Lower",
                fill="tonexty", fillcolor="rgba(239,68,68,0.1)",
                line=dict(color="rgba(239,68,68,0.3)", width=0),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(s_f), len(s_f) + future_n)),
                y=future_y, mode="lines+markers", name=f"Forecast (+{future_n})",
                line=dict(color="#ef4444", width=2.5, dash="dot"),
                marker=dict(size=4),
            ))
            fig.update_layout(title=f"🔮 Forecast — {col_f} (+{future_n} periods)")
            _style(fig, height=450)
            render_chart(fig, "forecast")

            # Forecast table
            fc_df = pd.DataFrame({
                "Period": [f"+{i+1}" for i in range(future_n)],
                "Forecast": np.round(future_y, 4),
                "CI Lower (95%)": np.round(ci_lower, 4),
                "CI Upper (95%)": np.round(ci_upper, 4),
            })
            with st.expander("📋 Forecast Values Table"):
                st.dataframe(fc_df, use_container_width=True, hide_index=True)

            st.markdown(
                _insight_box(forecast_insight(col_f, slope_f, intercept_f, r_val_f**2, future_n), "#6366f1"),
                unsafe_allow_html=True,
            )

    # ─── TAB 4: Cluster Analysis ───────────────────────────────────────────
    with tab_cluster:
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for cluster analysis.")
        else:
            st.markdown(
                '<div style="font-size:0.88rem;color:#6b7280;margin-bottom:1rem;">'
                'K-Means clustering groups records based on similarity across numeric columns.</div>',
                unsafe_allow_html=True,
            )
            cluster_cols = st.multiselect(
                "Select columns for clustering", num_cols,
                default=num_cols[:min(4, len(num_cols))],
                key="cluster_cols"
            )
            k = st.slider("Number of clusters (K)", 2, 10, 3, key="cluster_k")

            if len(cluster_cols) >= 2 and st.button("🔵 Run Clustering", key="run_cluster"):
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans

                cluster_df = df[cluster_cols].dropna()
                scaler = StandardScaler()
                scaled = scaler.fit_transform(cluster_df)
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(scaled)
                cluster_df = cluster_df.copy()
                cluster_df["Cluster"] = [f"Cluster {l+1}" for l in labels]

                c1, c2 = st.columns(2)
                with c1:
                    x_ax = st.selectbox("X-axis", cluster_cols, index=0, key="cl_x")
                with c2:
                    y_ax = st.selectbox("Y-axis", cluster_cols, index=min(1, len(cluster_cols)-1), key="cl_y")

                fig = px.scatter(cluster_df, x=x_ax, y=y_ax, color="Cluster",
                                 title=f"K-Means Clusters (K={k})",
                                 color_discrete_sequence=px.colors.qualitative.Bold,
                                 opacity=0.8)
                _style(fig)
                render_chart(fig, "cluster")

                # Cluster summary
                summary = cluster_df.groupby("Cluster")[cluster_cols].mean().round(3)
                st.subheader("Cluster Centroids (Mean Values)")
                st.dataframe(summary, use_container_width=True)

                sizes = cluster_df["Cluster"].value_counts().reset_index()
                sizes.columns = ["Cluster", "Count"]
                fig2 = px.bar(sizes, x="Cluster", y="Count", color="Cluster",
                              title="Records per Cluster",
                              color_discrete_sequence=px.colors.qualitative.Bold)
                _style(fig2, height=300)
                render_chart(fig2, "cluster_bar")

                st.markdown(
                    _insight_box(
                        f"K-Means partitioned the data into <strong>{k} clusters</strong> based on "
                        f"<strong>{', '.join(cluster_cols)}</strong>. "
                        "Inspect the centroids table to understand what characterises each group."
                    , "#6366f1"),
                    unsafe_allow_html=True,
                )

    # ─── TAB 5: Distribution Comparison ───────────────────────────────────
    with tab_compare:
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            comp_cols = st.multiselect(
                "Select columns to compare", num_cols,
                default=num_cols[:min(4, len(num_cols))],
                key="comp_cols"
            )
            if len(comp_cols) >= 2:
                melted = df[comp_cols].melt(var_name="Column", value_name="Value")
                fig1 = px.histogram(melted, x="Value", color="Column", barmode="overlay",
                                    title="Overlapping Distribution Comparison",
                                    opacity=0.6, nbins=40,
                                    color_discrete_sequence=px.colors.qualitative.Bold)
                _style(fig1)
                render_chart(fig1, "dist_comp_hist")

                fig2 = px.box(melted, x="Column", y="Value", color="Column",
                              title="Side-by-Side Box Plots",
                              color_discrete_sequence=px.colors.qualitative.Bold)
                _style(fig2)
                render_chart(fig2, "dist_comp_box")

                if len(comp_cols) >= 2:
                    c_pair = comp_cols[:2]
                    fig3 = px.violin(melted[melted["Column"].isin(c_pair)], x="Column", y="Value",
                                     color="Column", box=True,
                                     title=f"Violin — {c_pair[0]} vs {c_pair[1]}",
                                     color_discrete_sequence=["#6366f1", "#f59e0b"])
                    _style(fig3)
                    render_chart(fig3, "dist_comp_violin")

    # ─── TAB 6: Anomaly Detection ──────────────────────────────────────────
    with tab_anomaly:
        if not num_cols:
            st.warning("No numeric columns.")
        else:
            col_a = st.selectbox("Select column", num_cols, key="anom_col")
            method = st.radio("Method", ["Z-Score (σ>3)", "IQR (1.5×IQR)"], horizontal=True)
            s = df[col_a].dropna()

            if method == "Z-Score (σ>3)":
                z = np.abs(scipy_stats.zscore(s))
                anomaly_idx = s.index[z > 3]
            else:
                Q1 = s.quantile(0.25)
                Q3 = s.quantile(0.75)
                IQR = Q3 - Q1
                anomaly_idx = df[(df[col_a] < Q1 - 1.5 * IQR) | (df[col_a] > Q3 + 1.5 * IQR)].index

            anomalies = df.loc[anomaly_idx]
            pct = len(anomalies) / len(df) * 100

            c1, c2 = st.columns(2)
            c1.markdown(f'<div style="background:rgba(255,255,255,0.6);border-radius:14px;padding:1rem;box-shadow:0 2px 10px rgba(0,0,0,0.05);border-left:4px solid #ef4444;text-align:center;"><div style="font-size:1.8rem;font-weight:800;color:#ef4444;">{len(anomalies)}</div><div style="font-size:0.72rem;color:#6b7280;font-weight:600;">Anomalies Found</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div style="background:rgba(255,255,255,0.6);border-radius:14px;padding:1rem;box-shadow:0 2px 10px rgba(0,0,0,0.05);border-left:4px solid #6366f1;text-align:center;"><div style="font-size:1.8rem;font-weight:800;color:#6366f1;">{pct:.2f}%</div><div style="font-size:0.72rem;color:#6b7280;font-weight:600;">of Total Records</div></div>', unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[col_a], mode="markers",
                                     name="Normal", marker=dict(color="#6366f1", size=4, opacity=0.6)))
            if len(anomalies) > 0:
                fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies[col_a], mode="markers",
                                         name="Anomaly", marker=dict(color="#ef4444", size=10, symbol="x")))
            fig.update_layout(title=f"Anomaly Detection — {col_a}")
            _style(fig, height=420)
            render_chart(fig, "anomaly")

            if len(anomalies) > 0:
                with st.expander(f"📋 View {len(anomalies)} Anomalous Records"):
                    st.dataframe(anomalies, use_container_width=True)

            st.markdown(_insight_box(
                f"<strong>{col_a}</strong>: {len(anomalies)} anomalies detected ({pct:.2f}% of data) using {method}. "
                f"{'Consider investigating or removing these records before modelling.' if pct > 1 else 'Low anomaly rate — data looks clean.'}"
            , "#ef4444" if pct > 1 else "#22c55e"), unsafe_allow_html=True)

    # ─── TAB 7: Rolling Statistics ─────────────────────────────────────────
    with tab_rolling:
        if not num_cols:
            st.warning("No numeric columns.")
        else:
            rcol = st.selectbox("Select column", num_cols, key="roll_col")
            window = st.slider("Window size", 2, min(100, max(2, len(df) // 2)), 10, key="roll_win")
            roll_mean = df[rcol].rolling(window=window).mean()
            roll_std = df[rcol].rolling(window=window).std()
            roll_df = pd.DataFrame({
                rcol: df[rcol],
                f"{window}-period Mean": roll_mean,
                f"+1σ": roll_mean + roll_std,
                f"-1σ": roll_mean - roll_std,
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=roll_df[rcol], mode="lines", name=rcol,
                                     line=dict(color="rgba(99,102,241,0.4)", width=1)))
            fig.add_trace(go.Scatter(y=roll_df[f"+1σ"], mode="lines", name="+1σ",
                                     line=dict(color="#f59e0b", width=1, dash="dot"),
                                     showlegend=True))
            fig.add_trace(go.Scatter(y=roll_df[f"-1σ"], mode="lines", name="-1σ",
                                     fill="tonexty", fillcolor="rgba(245,158,11,0.07)",
                                     line=dict(color="#f59e0b", width=1, dash="dot")))
            fig.add_trace(go.Scatter(y=roll_df[f"{window}-period Mean"], mode="lines",
                                     name=f"Rolling Mean ({window})",
                                     line=dict(color="#6366f1", width=2.5)))
            fig.update_layout(title=f"Rolling Statistics — {rcol} (window={window})")
            _style(fig, height=430)
            render_chart(fig, "rolling")

    # ─── TAB 8: Growth Rate ────────────────────────────────────────────────
    with tab_growth:
        if not num_cols:
            st.warning("No numeric columns.")
        else:
            col_g = st.selectbox("Select column", num_cols, key="gr_col")
            s_g = df[col_g].dropna()
            growth = s_g.pct_change() * 100
            growth_df = pd.DataFrame({"Value": s_g.values, "Growth Rate (%)": growth.values})

            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.line(growth_df, y="Growth Rate (%)",
                               title=f"Period-over-Period Growth — {col_g}",
                               color_discrete_sequence=["#6366f1"])
                fig1.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.3)")
                _style(fig1)
                render_chart(fig1, "growth_line")

            with c2:
                fig2 = px.histogram(growth_df.dropna(), x="Growth Rate (%)", nbins=40,
                                    title="Distribution of Growth Rates",
                                    color_discrete_sequence=["#f59e0b"],
                                    marginal="box")
                _style(fig2)
                render_chart(fig2, "growth_hist")

            avg_growth = growth.mean()
            pos_pct = (growth > 0).sum() / len(growth.dropna()) * 100
            st.markdown(_insight_box(
                f"<strong>{col_g}:</strong> Average growth = <strong>{avg_growth:.2f}%</strong> per period. "
                f"Positive growth in <strong>{pos_pct:.1f}%</strong> of periods. "
                f"{'Consistently growing.' if pos_pct > 70 else 'Consistently declining.' if pos_pct < 30 else 'Mixed growth pattern.'}"
            , "#22c55e" if avg_growth > 0 else "#ef4444"), unsafe_allow_html=True)

    # ─── TAB 9: What-If Scenario Simulator ────────────────────────────────
    with tab_whatif:
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns to run What-If simulations.")
        else:
            st.markdown(
                '<div style="font-size:0.88rem;color:#6b7280;margin-bottom:1rem;">'
                'Adjust the sliders to simulate "what would happen" to your target KPI. '
                'A <strong>ridge regression model</strong> predicts the outcome and estimates '
                'a 95% confidence interval around the projection.</div>',
                unsafe_allow_html=True,
            )
            target = st.selectbox("Select target KPI (column to predict)", num_cols, key="wi_target")

            # Features correlated with target
            other = [c for c in num_cols if c != target]
            corrs = {}
            for c in other:
                cc = df[[c, target]].dropna()
                if len(cc) > 5:
                    corrs[c] = abs(cc[c].corr(cc[target]))
            top_feats = sorted(corrs, key=corrs.get, reverse=True)[:4]
            if not top_feats:
                st.warning("No other numeric columns correlate with the target.")
            else:
                st.markdown(f'<div style="font-size:0.8rem;color:#6b7280;margin-bottom:0.5rem;">Top drivers of <strong>{target}</strong>: {", ".join(f"{c} (r={corrs[c]:.2f})" for c in top_feats)}</div>', unsafe_allow_html=True)

                # Feature sliders centered on median
                slider_vals = {}
                for c in top_feats:
                    med = float(df[c].median())
                    lo, hi = float(df[c].quantile(0.05)), float(df[c].quantile(0.95))
                    if hi <= lo:
                        lo, hi = float(df[c].min()), float(df[c].max())
                    span = max(hi - lo, 1e-9)
                    slider_vals[c] = st.slider(f"{c} (baseline: {med:,.1f})", lo, hi, med, key=f"wi_slider_{c}", format="%.1f")

                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import StandardScaler
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import r2_score

                model_df = df[top_feats + [target]].dropna()
                if len(model_df) < 10:
                    st.warning("Not enough complete rows to fit the model.")
                else:
                    X = model_df[top_feats].values
                    y = model_df[target].values
                    scaler = StandardScaler()
                    Xs = scaler.fit_transform(X)
                    model = Ridge(alpha=1.0)
                    model.fit(Xs, y)
                    r2 = r2_score(y, model.predict(Xs))

                    # Projection for current slider values
                    X_proj = scaler.transform(np.array([[slider_vals[c] for c in top_feats]]))
                    proj = float(model.predict(X_proj)[0])
                    # Residual-based 95% CI
                    resid = y - model.predict(Xs)
                    se = np.std(resid) * (1 + 1 / len(y)) ** 0.5
                    ci = 1.96 * se

                    st.markdown(f'<div style="background:linear-gradient(135deg,#eef2ff,#e0e7ff);border-radius:16px;padding:1.2rem;text-align:center;margin:0.8rem 0;border:1px solid #c7d2fe;"><div style="font-size:0.75rem;color:#4f46e5;font-weight:700;letter-spacing:0.05em;">PROJECTED {target.upper()}</div><div style="font-size:2.4rem;font-weight:900;color:#1e1b4b;">{proj:,.2f}</div><div style="font-size:0.8rem;color:#6b7280;">95% Confidence Interval: {proj-ci:,.2f} — {proj+ci:,.2f}</div><div style="font-size:0.75rem;color:#9ca3af;margin-top:0.3rem;">Model fit R² = {r2:.3f} on {len(model_df):,} rows</div></div>', unsafe_allow_html=True)

                    # Comparison chart: baseline prediction vs scenario
                    X_base = scaler.transform(np.array([[df[c].median() for c in top_feats]]))
                    base_proj = float(model.predict(X_base)[0])
                    fig_wi = go.Figure()
                    fig_wi.add_trace(go.Bar(x=["Baseline (all medians)", "Your Scenario"], y=[base_proj, proj],
                                            text=[f"{base_proj:,.0f}", f"{proj:,.0f}"], textposition="outside",
                                            marker_color=["#9ca3af", "#6366f1"]))
                    fig_wi.add_trace(go.Scatter(x=["Baseline (all medians)", "Your Scenario"], y=[proj - ci, proj + ci],
                                                mode="lines", line=dict(color="rgba(99,102,241,0.4)", width=2),
                                                name="95% CI", showlegend=False))
                    fig_wi.update_layout(title=f"What-If Simulation — {target}", yaxis_title=target, showlegend=False, height=380)
                    _style(fig_wi)
                    render_chart(fig_wi, "whatif")

                    delta = proj - base_proj
                    direction = "increase" if delta > 0 else "decrease"
                    st.markdown(_insight_box(
                        f"<strong>Scenario impact:</strong> Changing the top drivers to your selected values projects a "
                        f"<strong>{direction}</strong> of <strong>{abs(delta):,.2f}</strong> in {target} "
                        f"({delta/max(abs(base_proj), 0.01)*100:.1f}%) vs the median baseline. "
                        f"Model confidence (R²) is <strong>{r2:.2f}</strong> — {'high, projections are reliable.' if r2 > 0.6 else 'moderate, treat projections as directional.'}"
                    , "#22c55e" if delta > 0 else "#ef4444"), unsafe_allow_html=True)
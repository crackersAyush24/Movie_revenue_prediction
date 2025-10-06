import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import time

# -------- Page config & minimal theme-aware CSS --------
st.set_page_config(page_title="Movie Revenue Prediction", page_icon="🎬", layout="wide")  # tabs/columns breathe better [web:22][web:33]
st.markdown(
    """
    <style>
        .app-title { text-align:center; color: cyan; margin-top: -6px; }
        .section-subtitle { text-align:center; color: #e6e6e6; margin-bottom: 8px; }
        .result-card {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 1rem 1.25rem;
            border-radius: 0.5rem;
        }
        .kpi {
            background: rgba(0,0,0,0.15);
            padding: 0.75rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(255,255,255,0.08);
            text-align:center;
        }
        .dim { color: #bfbfbf; }
    </style>
    """,
    unsafe_allow_html=True,
)  # theme-friendly styling [web:22]

# -------- Data prep & model (unchanged logic) --------
def prepare_features(df):
    processed_df = preprocess_data(df)
    if "log_gross" in processed_df.columns:
        y = processed_df["log_gross"]
        X = processed_df.drop("log_gross", axis=1)
    else:
        y = None
        X = processed_df
    return X, y

def preprocess_data(df):
    df = df.copy()

    # Log transforms
    if "gross" in df.columns:
        df["log_gross"] = np.log1p(df["gross"])
    df["log_budget"] = np.log1p(df["budget"])

    # Feature engineering
    df["budget_vote_ratio"] = df["budget"] / (df["votes"] + 1)
    df["budget_runtime_ratio"] = df["budget"] / (df["runtime"] + 1)
    df["budget_score_ratio"] = df["log_budget"] / (df["score"] + 1)
    df["vote_score_ratio"] = df["votes"] / (df["score"] + 1)
    df["budget_year_ratio"] = df["log_budget"] / (df["year"] - df["year"].min() + 1)
    df["vote_year_ratio"] = df["votes"] / (df["year"] - df["year"].min() + 1)
    df["score_runtime_ratio"] = df["score"] / (df["runtime"] + 1)
    df["budget_per_minute"] = df["budget"] / (df["runtime"] + 1)
    df["votes_per_year"] = df["votes"] / (df["year"] - df["year"].min() + 1)
    df["is_recent"] = (df["year"] >= df["year"].quantile(0.75)).astype(int)
    df["is_high_budget"] = (df["log_budget"] >= df["log_budget"].quantile(0.75)).astype(int)
    df["is_high_votes"] = (df["votes"] >= df["votes"].quantile(0.75)).astype(int)
    df["is_high_score"] = (df["score"] >= df["score"].quantile(0.75)).astype(int)

    categorical_features = [
        "released","writer","rating","name","genre",
        "director","star","country","company",
    ]
    for feature in categorical_features:
        df[feature] = df[feature].astype(str)
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    numerical_features = [
        "runtime","score","year","votes","log_budget",
        "budget_vote_ratio","budget_runtime_ratio","budget_score_ratio",
        "vote_score_ratio","budget_year_ratio","vote_year_ratio",
        "score_runtime_ratio","budget_per_minute","votes_per_year",
        "is_recent","is_high_budget","is_high_votes","is_high_score",
    ]

    imputer = SimpleImputer(strategy="median")
    df[numerical_features] = imputer.fit_transform(df[numerical_features])

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    if "gross" in df.columns:
        df = df.drop(["gross", "budget"], axis=1)
    else:
        df = df.drop(["budget"], axis=1)
    return df

def run_model():
    df = pd.read_csv("revised datasets/output.csv")
    X, y = prepare_features(df)
    param_grid = {
        "n_estimators": [100, 500],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    }
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, **best_params)
    best_model.fit(X, y)
    return best_model

def predict_gross(input_data, best_model):
    processed_data = preprocess_data(pd.DataFrame([input_data]))
    expected_features = best_model.feature_names_in_
    for feature in expected_features:
        if feature not in processed_data.columns:
            processed_data[feature] = 0
    processed_data = processed_data[expected_features]
    log_prediction = best_model.predict(processed_data)
    prediction = np.exp(log_prediction) - 1
    return prediction[0]

def predict_gross_range(gross):
    if gross <= 10000000:
        return f"Low Revenue (<= 10M)"
    elif gross <= 40000000:
        return f"Medium-Low Revenue (10M - 40M)"
    elif gross <= 70000000:
        return f"Medium Revenue (40M - 70M)"
    elif gross <= 120000000:
        return f"Medium-High Revenue (70M - 120M)"
    elif gross <= 200000000:
        return f"High Revenue (120M - 200M)"
    else:
        return f"Ultra High Revenue (>= 200M)"

# -------- Top headings --------
st.markdown("<h1 class='app-title'>Movie Revenue Prediction</h1>", unsafe_allow_html=True)  # centered [web:22]
st.markdown("<h2 class='section-subtitle'>Movie Details</h2>", unsafe_allow_html=True)  # centered [web:22]

# -------- Sidebar inputs (kept) --------
with st.sidebar:
    st.header("Enter Movie Details")  # sidebar pattern [web:22]

    with st.expander("Basics", expanded=True):  # navigable groups [web:22]
        name = st.text_input("Movie Name", help="Official title.")
        genre = st.text_input("Genre", help="Primary genre, e.g., Action, Drama.")
        rating = st.selectbox("MPAA Rating", ["G", "PG", "PG-13", "R", "NC-17"], help="Content rating.")
        released = st.text_input("Release Date", help="e.g., 2012-07-20.")

    with st.expander("People & Company", expanded=False):
        director = st.text_input("Director", help="Director's name.")
        writer = st.text_input("Writer", help="Primary screenwriter.")
        star = st.text_input("Leading Star", help="Lead actor/actress.")
        company = st.text_input("Production Company", help="Studio/company.")
        country = st.text_input("Country of Production", help="Primary production country.")

    with st.expander("Numbers", expanded=True):
        # Form for atomic submission of numerics
        with st.form(key="movie_form"):
            c1, c2 = st.columns(2)  # aligned numeric inputs [web:22]
            with c1:
                runtime = st.number_input("Runtime (minutes)", min_value=0.0, help="Total runtime in minutes.")
                score = st.number_input("IMDb Score", min_value=0.0, max_value=10.0, help="0–10.")
                year = st.number_input("Year of Release", min_value=1900, max_value=2100, help="YYYY.")
            with c2:
                budget = st.number_input("Budget", min_value=0.0, help="Budget in USD.")
                votes = st.number_input("Initial Votes", min_value=0, help="Initial vote count.")
            submit_button = st.form_submit_button(label="Predict Revenue")

# -------- Tabs for output and information --------
tab_pred, tab_feats, tab_about = st.tabs(["Prediction", "Feature Insight", "About"])  # organized output [web:33][web:22]

with tab_pred:
    if 'submit_button' in locals() and submit_button:
        # Quick validation for key fields
        missing_keys = []
        for key, val in {
            "name": name, "genre": genre, "rating": rating, "released": released,
            "director": director, "writer": writer, "star": star,
            "company": company, "country": country,
        }.items():
            if isinstance(val, str) and len(val.strip()) == 0:
                missing_keys.append(key)

        if missing_keys:
            st.toast(f"Missing fields: {', '.join(missing_keys)}", icon="⚠️", duration="long")  # non-blocking warning [web:37]
        else:
            st.toast("Submitting for prediction…", icon="✅", duration="short")  # success notice [web:37]

        # Status + progress during training/fitting
        with st.status("Training model with GridSearchCV…", expanded=False):  # animated status [web:44]
            # Simulated phases for UX; real work in run_model()
            st.write("Preparing data and parameter grid…")  # status steps [web:44]
            time.sleep(0.2)
            progress = st.progress(5, text="Cross-validating models…")  # progress bar [web:38]
            for pct in range(5, 70, 5):
                time.sleep(0.02)
                progress.progress(pct, text="Cross-validating models…")
            best_model = run_model()
            for pct in range(70, 101, 5):
                time.sleep(0.02)
                progress.progress(pct, text="Finalizing best model…")
            progress.empty()
            st.write("Model ready.")

        input_data = {
            "released": released, "writer": writer, "rating": rating, "name": name,
            "genre": genre, "director": director, "star": star, "country": country,
            "company": company, "runtime": runtime, "score": score, "budget": budget,
            "year": year, "votes": votes,
        }

        predicted_gross = predict_gross(input_data, best_model)
        predicted_gross_range = predict_gross_range(predicted_gross)

        st.markdown("### Prediction Result")  # clear section [web:22]
        # KPI row
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Predicted Revenue", f"${predicted_gross:,.0f}")  # KPI style [web:22]
        with k2:
            st.metric("Revenue Range", predicted_gross_range)
        with k3:
            st.metric("IMDb Score", f"{score:.1f}")

        # Detail card
        st.markdown(
            f"""
            <div class="result-card">
              <p><strong>Title:</strong> {name}</p>
              <p><strong>Genre:</strong> {genre} <span class="dim">({rating})</span></p>
              <p><strong>Team:</strong> Dir. {director} · Wri. {writer} · Star {star}</p>
              <p><strong>Release:</strong> {released} · <strong>Year:</strong> {year} · <strong>Runtime:</strong> {runtime} min</p>
              <p><strong>Production:</strong> {company} · {country}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Use the sidebar to enter details and submit to see predictions.")  # guide [web:22]

with tab_feats:
    st.markdown("### Feature Insight")
    st.write(
        "Engineered features include log transforms and interaction ratios such as budget-to-runtime, vote-to-score, and time-normalized metrics; these assist tree models in capturing non-linear patterns without manual binning. [Docs: Layout & containers reference]",  # description
    )  # [web:22]
    # Simple textual summary to avoid changing model
    bullets = [
        "Log transforms: log_budget and optionally log_gross improve scale handling.",
        "Ratios normalize budget/votes by runtime, score, and year.",
        "Binary flags like is_recent and is_high_budget create coarse buckets.",
    ]
    for b in bullets:
        st.write(f"- {b}")  # simple list in info tab [web:22]

with tab_about:
    st.markdown("### About")
    st.write(
        "This app predicts movie revenue using an XGBoost regressor with a GridSearchCV over depth, estimators, and learning rate, trained on the provided dataset. Use the sidebar to provide movie details and submit to view predictions and KPIs.",  # concise about
    )  # [web:22]

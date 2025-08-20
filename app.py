# ------------------------------
# Cognitive Pharma Plant Digital Twin Dashboard (Lazy-Load & Cloud-Safe)
# ------------------------------

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
from pyvis.network import Network
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit.components.v1 as components

# ========== Page Setup ==========
st.set_page_config(page_title="Cognitive Pharma Digital Twin", layout="wide")
st.title("🧬 Cognitive Pharma Plant Digital Twin")

st.markdown("""
Use the sidebar to **load data** (Kaggle / CSV upload / demo), then simulate batches.
This app is cloud-safe: it **won’t crash** if Kaggle fails — it falls back to a demo dataset.
""")

# ========== Session State ==========
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "simulated_batches" not in st.session_state:
    st.session_state.simulated_batches = []
if "cumulative_effect" not in st.session_state:
    st.session_state.cumulative_effect = None
if "G" not in st.session_state:
    st.session_state.G = None

# ========== Sidebar: Data Loading ==========
st.sidebar.header("📊 Data")
data_source = st.sidebar.radio(
    "Choose data source",
    ["Kaggle dataset", "Upload CSV", "Use demo dataset"],
    index=0
)

# Optional: read Kaggle creds from secrets (does nothing if not set)
if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

dataset_id = st.sidebar.text_input(
    "Kaggle Dataset ID",
    value="stephengoldie/big-databiopharmaceutical-manufacturing",
    help="Example: username/dataset-slug"
)

uploaded_csv = None
if data_source == "Upload CSV":
    uploaded_csv = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

load_btn = st.sidebar.button("📥 Load data")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@st.cache_data(show_spinner=False)
def _demo_df():
    return pd.DataFrame({
        "temperature": [37, 38, 39, 40, 41, 42],
        "pressure":    [1.00, 1.10, 1.20, 1.05, 1.15, 1.08],
        "agitation":   [200,  210,  190,  220,  205,  215],
        "yield":       [0.95, 0.96, 0.97, 0.93, 0.94, 0.965]
    })

@st.cache_data(show_spinner=True)
def _load_from_kaggle(slug: str) -> pd.DataFrame:
    """
    Download first CSV/Excel from the Kaggle dataset.
    Returns a DataFrame or raises Exception (caught by caller).
    """
    from kaggle.api.kaggle_api_extended import KaggleApi  # lazy import
    api = KaggleApi()
    api.authenticate()

    download_path = DATA_DIR / "kaggle_download"
    download_path.mkdir(exist_ok=True)
    api.dataset_download_files(slug, path=str(download_path), unzip=True)

    # Load first CSV/Excel found
    for root, _, files in os.walk(download_path):
        for f in files:
            p = Path(root) / f
            if f.lower().endswith(".csv"):
                return pd.read_csv(p)
            if f.lower().endswith((".xls", ".xlsx")):
                return pd.read_excel(p)
    raise FileNotFoundError("No CSV/Excel files found in Kaggle dataset.")

def _set_graph(assets):
    G = nx.DiGraph()
    for a in assets:
        G.add_node(a, type="asset", color="lightblue", size=20)
    return G

# Actually load data only when user clicks
if load_btn:
    try:
        if data_source == "Kaggle dataset":
            with st.status("Authenticating & downloading from Kaggle…", expanded=False):
                df = _load_from_kaggle(dataset_id)
            st.success(f"✅ Loaded Kaggle dataset with shape {df.shape}")
        elif data_source == "Upload CSV":
            if uploaded_csv is None:
                st.warning("Please upload a CSV first.")
                df = _demo_df()
                st.info("Using demo dataset instead.")
            else:
                df = pd.read_csv(uploaded_csv)
                st.success(f"✅ Loaded uploaded CSV with shape {df.shape}")
        else:  # demo
            df = _demo_df()
            st.success(f"✅ Loaded demo dataset with shape {df.shape}")

        st.session_state.df = df
        st.session_state.simulated_batches = []
        st.session_state.cumulative_effect = None
        st.session_state.G = _set_graph(["Reactor_1", "Reactor_2", "Cold_Storage_1", "Cold_Storage_2"])
    except Exception as e:
        st.error(f"Data load failed: {e}")
        st.info("Falling back to demo dataset.")
        st.session_state.df = _demo_df()
        st.session_state.simulated_batches = []
        st.session_state.cumulative_effect = None
        st.session_state.G = _set_graph(["Reactor_1", "Reactor_2", "Cold_Storage_1", "Cold_Storage_2"])

df = st.session_state.df

if df.empty:
    st.info("👈 Load a dataset from the sidebar to get started, or choose **Use demo dataset**.")
    st.stop()

# ========== Show data preview ==========
with st.expander("🔎 Data preview", expanded=False):
    st.write(df.head())
    st.caption(f"Rows: {len(df)} | Columns: {list(df.columns)}")

# ========== Sidebar: Fault Injection & Simulation Controls ==========
st.sidebar.header("⚙️ Simulation")
fault_chance = st.sidebar.slider("Chance of Faulty Batch (%)", 0, 100, 10, 5)
inject_faults = st.sidebar.checkbox("Enable Faulty Batches", value=True)
num_batches = st.sidebar.slider("Number of Batches to Simulate", 1, 10, 1)
selected_event = st.sidebar.selectbox(
    "Event for Simulation",
    ["Normal", "Reactor_Failure", "Cold_Storage_Issue", "Supply_Delay"]
)
simulate_btn = st.sidebar.button("▶️ Simulate")

# ========== Modeling ==========
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found. Please load a dataset with numeric features.")
    st.stop()

target_col = "yield" if "yield" in numeric_cols else numeric_cols[-1]
features = [c for c in numeric_cols if c != target_col]

# Train once per load
@st.cache_data(show_spinner=False)
def _train_model(df_in: pd.DataFrame, features_in, target_in):
    X_train, X_test, y_train, y_test = train_test_split(
        df_in[features_in], df_in[target_in], test_size=0.2, random_state=42
    )
    model_ = RandomForestRegressor(n_estimators=120, random_state=42)
    model_.fit(X_train, y_train)
    y_pred = model_.predict(X_test)
    rmse_ = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    baseline_ = df_in[features_in].tail(1).iloc[0]
    return model_, rmse_, baseline_

with st.spinner("Training model…"):
    model, rmse, baseline = _train_model(df, features, target_col)
st.sidebar.success(f"Model ready. RMSE: {rmse:.3f}")

# Graph (assets) setup if not already done
if st.session_state.G is None:
    st.session_state.G = _set_graph(["Reactor_1", "Reactor_2", "Cold_Storage_1", "Cold_Storage_2"])
G = st.session_state.G
ASSETS = list(G.nodes) if G is not None else ["Reactor_1", "Reactor_2", "Cold_Storage_1", "Cold_Storage_2"]

# ========== Simulation ==========
impact_map = {
    "Normal": 0.0,
    "Reactor_Failure": -0.10,
    "Cold_Storage_Issue": -0.08,
    "Supply_Delay": -0.05,
}

def risk_color(risk):
    if risk < 0.2:
        return "lightgreen"
    elif risk < 0.5:
        return "orange"
    return "red"

def recommend_action(batch):
    if batch["risk_score"] > 0.2:
        return {
            "Reactor_Failure": "Shift production to backup reactor and increase monitoring.",
            "Cold_Storage_Issue": "Prioritize rapid transfer to alternative storage.",
            "Supply_Delay": "Reallocate resources to meet schedule.",
            "Faulty_Batch": "Quarantine output, trigger CAPA, and inspect upstream equipment."
        }.get(batch["event"], "Apply general corrective measures.")
    return "Normal operation."

def simulate_next_batch(event_type, cumulative_effect_vec: pd.Series):
    noise = np.random.normal(0, 0.02, size=len(features))
    batch = baseline + cumulative_effect_vec + noise + impact_map.get(event_type, 0.0)

    # Fault injection
    if inject_faults and (np.random.rand() < (fault_chance / 100.0)):
        batch = batch * np.random.uniform(0.7, 0.9, size=len(batch))
        actual_event = "Faulty_Batch"
    else:
        actual_event = event_type

    batch_df = pd.DataFrame([batch], columns=features)
    batch_pred = float(model.predict(batch_df)[0])

    # Risk: 1 - predicted_yield (+0.1 penalty for non-normal)
    penalty = 0.1 if actual_event != "Normal" else 0.0
    risk_score = round(min(max(1.0 - batch_pred + penalty, 0.0), 1.0), 3)

    # Drift accumulation heuristic
    incr = np.where(batch_pred < 0.95, 0.01, 0.005)
    cumulative_effect_vec = cumulative_effect_vec + incr

    return {"predicted_yield": batch_pred, "risk_score": risk_score, "event": actual_event}, cumulative_effect_vec

# Initialize cumulative effect
if st.session_state.cumulative_effect is None:
    st.session_state.cumulative_effect = pd.Series(0.0, index=features)

if simulate_btn:
    for _ in range(int(num_batches)):
        batch_result, st.session_state.cumulative_effect = simulate_next_batch(
            selected_event, st.session_state.cumulative_effect
        )
        batch_result["batch"] = len(st.session_state.simulated_batches) + 1
        st.session_state.simulated_batches.append(batch_result)

# ========== Display ==========
if st.session_state.simulated_batches:
    results_df = pd.DataFrame(st.session_state.simulated_batches)
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("📈 Simulation Results")
        st.dataframe(results_df, use_container_width=True)

        st.subheader("💡 AI Recommendations")
        recent = st.session_state.simulated_batches[-min(len(st.session_state.simulated_batches), num_batches):]
        for batch in recent:
            st.write(f"• **Batch {batch['batch']}** ({batch['event']}): {recommend_action(batch)}")

    with col2:
        st.subheader("🧪 Stats")
        st.metric("Batches simulated", len(st.session_state.simulated_batches))
        st.metric("Avg predicted yield", f"{results_df['predicted_yield'].mean():.3f}")
        st.metric("Avg risk", f"{results_df['risk_score'].mean():.3f}")

    st.subheader("🌐 Knowledge Graph")
    net = Network(height="600px", width="100%", directed=True)
    # Add assets
    for node, attrs in G.nodes(data=True):
        net.add_node(node, label=node, color=attrs.get("color", "lightblue"))
    # Add batch nodes & edges
    for batch in st.session_state.simulated_batches:
        bnode = f"Batch_{batch['batch']}"
        net.add_node(bnode, label=bnode, color=risk_color(batch["risk_score"]))
        impacted_assets = np.random.choice(ASSETS, size=np.random.randint(1, len(ASSETS) + 1), replace=False)
        for a in impacted_assets:
            net.add_edge(bnode, a)
    html = net.generate_html()
    components.html(html, height=650, scrolling=True)
else:
    st.info("No simulation yet. Choose parameters in the sidebar and click **Simulate**.")

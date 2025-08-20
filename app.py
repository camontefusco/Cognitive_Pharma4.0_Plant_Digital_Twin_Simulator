# ------------------------------
# Cognitive Pharma Plant Digital Twin Dashboard (Robust Update)
# ------------------------------

import os
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit.components.v1 as components
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

# ------------------------------
# Handle Kaggle API credentials
# ------------------------------
if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

# ------------------------------
st.title("🧬 Cognitive Pharma Plant Digital Twin")
st.markdown("""
### Quick Start
1. Add Kaggle credentials in Streamlit Secrets.  
2. Load Dataset – includes normal and faulty batches.  
3. Enable Fault Simulation (optional).  
4. Train Model – Random Forest predicts batch yield.  
5. Simulate Batches – select event type, number of batches, and fault chance.  
6. Review Results – table shows yield, risk, event; knowledge graph links batches to assets.
""")

# ------------------------------
# Dataset Setup
# ------------------------------
dataset_path = Path("data")
dataset_path.mkdir(exist_ok=True)

st.sidebar.title("📊 Dataset Settings")
dataset_id = st.sidebar.text_input(
    "Kaggle Dataset ID",
    "stephengoldie/big-databiopharmaceutical-manufacturing"
)

# ------------------------------
# Safe Kaggle dataset fetch
# ------------------------------
def fetch_dataset_safe(dataset_slug: str) -> pd.DataFrame:
    try:
        api = KaggleApi()
        api.authenticate()
        download_path = dataset_path / "kaggle_download"
        download_path.mkdir(exist_ok=True)
        api.dataset_download_files(dataset_slug, path=str(download_path), unzip=True)
        for root, _, files in os.walk(download_path):
            for f in files:
                if f.endswith(".csv"):
                    return pd.read_csv(os.path.join(root, f))
                elif f.endswith((".xls", ".xlsx")):
                    return pd.read_excel(os.path.join(root, f))
        st.warning("No CSV/Excel found; using sample dataset.")
    except Exception as e:
        st.warning(f"Kaggle download failed: {e}")
    # fallback sample dataset
    return pd.DataFrame({
        "feature1": [0.1,0.2,0.3],
        "feature2": [1.0,1.1,1.2],
        "yield": [0.95,0.97,0.96]
    })

if "df" not in st.session_state:
    st.session_state.df = fetch_dataset_safe(dataset_id)
df = st.session_state.df
st.success("Dataset ready!")

# ------------------------------
# Fault Injection
# ------------------------------
st.sidebar.subheader("Fault Injection")
fault_chance = st.sidebar.slider("Chance of Faulty Batch (%)", 0, 100, 10, 5)
inject_faults = st.sidebar.checkbox("Enable Faulty Batches", value=True)

# ------------------------------
# Train Predictive Model
# ------------------------------
numeric_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in dataset.")
    st.stop()

target_col = 'yield' if 'yield' in numeric_cols else numeric_cols[-1]
features = [c for c in numeric_cols if c != target_col]

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target_col], test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
st.sidebar.write(f"✅ RMSE on test set: {rmse:.3f}")

# ------------------------------
# Knowledge Graph Setup
# ------------------------------
ASSETS = ["Reactor_1", "Reactor_2", "Cold_Storage_1", "Cold_Storage_2"]

if "G" not in st.session_state:
    G = nx.DiGraph()
    for asset in ASSETS:
        G.add_node(asset, type="asset", color='lightblue', size=20)
    st.session_state.G = G
else:
    G = st.session_state.G

baseline = df[features].tail(1).iloc[0]

# ------------------------------
# Event Simulation
# ------------------------------
event_options = ["Normal", "Reactor_Failure", "Cold_Storage_Issue", "Supply_Delay"]

if "cumulative_effect" not in st.session_state:
    st.session_state.cumulative_effect = pd.Series(0, index=features)

impact_map = {
    "Normal": 0.0,
    "Reactor_Failure": -0.1,
    "Cold_Storage_Issue": -0.08,
    "Supply_Delay": -0.05
}

def simulate_next_batch(event_type, cumulative_effect):
    batch = baseline + cumulative_effect + np.random.normal(0, 0.02, size=len(features))
    batch += impact_map.get(event_type, 0)
    if inject_faults and np.random.rand() < (fault_chance / 100):
        batch *= np.random.uniform(0.7, 0.9, size=len(batch))
        actual_event = "Faulty_Batch"
    else:
        actual_event = event_type
    batch_df = pd.DataFrame([batch], columns=features)
    batch_pred = model.predict(batch_df)[0]
    risk_score = round(min(1 - batch_pred + (0.1 if actual_event != "Normal" else 0), 1.0), 2)
    cumulative_effect += np.where(batch_pred < 0.95, 0.01, 0.005)
    return {'predicted_yield': batch_pred, 'risk_score': risk_score, 'event': actual_event}, cumulative_effect

def recommend_action(batch):
    if batch['risk_score'] > 0.2:
        return {
            "Reactor_Failure": "Shift production to backup reactor and increase monitoring.",
            "Cold_Storage_Issue": "Prioritize rapid transfer to alternative storage.",
            "Supply_Delay": "Reallocate resources to meet schedule."
        }.get(batch['event'], "Apply general corrective measures.")
    return "Normal operation."

def risk_color(risk):
    if risk < 0.2: return 'lightgreen'
    elif risk < 0.5: return 'orange'
    return 'red'

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.title("Batch Simulation")
selected_event = st.sidebar.selectbox("Select Event for Simulation", event_options)
num_batches = st.sidebar.slider("Number of Batches to Simulate", 1, 10, 1)
apply_event = st.sidebar.button("Simulate Batches")

if "simulated_batches" not in st.session_state:
    st.session_state.simulated_batches = []

if apply_event:
    for _ in range(num_batches):
        batch_result, st.session_state.cumulative_effect = simulate_next_batch(
            selected_event, st.session_state.cumulative_effect
        )
        batch_result['batch'] = len(st.session_state.simulated_batches) + 1
        st.session_state.simulated_batches.append(batch_result)

# ------------------------------
# Display Results
# ------------------------------
if st.session_state.simulated_batches:
    results_df = pd.DataFrame(st.session_state.simulated_batches)
    st.subheader("📈 Simulation Results")
    st.dataframe(results_df)

    st.subheader("💡 AI Recommendations")
    for batch in st.session_state.simulated_batches[-num_batches:]:
        st.write(f"Batch {batch['batch']} ({batch['event']}): {recommend_action(batch)}")

    st.subheader("🌐 Knowledge Graph")
    if G.nodes:
        net = Network(height="600px", width="100%", directed=True)
        for node in G.nodes(data=True):
            net.add_node(node[0], label=node[0], color=node[1].get("color", "blue"))
        for batch in st.session_state.simulated_batches:
            batch_node = f"Batch_{batch['batch']}"
            net.add_node(batch_node, label=batch_node, color=risk_color(batch['risk_score']))
            impacted_assets = np.random.choice(ASSETS, size=np.random.randint(1, len(ASSETS)+1), replace=False)
            for asset in impacted_assets:
                net.add_edge(batch_node, asset)
        html = net.generate_html()
        components.html(html, height=650, scrolling=True)

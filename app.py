# ------------------------------
# Cognitive Pharma Plant Digital Twin Dashboard
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
# Handle Kaggle API credentials via Streamlit secrets (for Streamlit Cloud)
# ------------------------------
if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

# ------------------------------
# User Instructions
# ------------------------------
st.title("🧬 Cognitive Pharma Plant Digital Twin")

st.markdown("""
### Quick Start

1. **Kaggle API Key** – Add your Kaggle credentials in Streamlit Secrets.  
2. **Load Dataset** – Contains normal and faulty batches. Faulty batches reduce yield and increase risk.  
3. **Enable Fault Simulation (Optional)** – Set “Chance of Faulty Batch (%)” to test AI recommendations.  
4. **Train Model** – Random Forest predicts batch yield; RMSE indicates prediction accuracy.  
5. **Simulate Batches** – Select event type, number of batches, and fault chance; click “Simulate Batches.”  
6. **Review Results** – Table shows predicted yield, risk, and event; Knowledge Graph visualizes batch-asset links; AI recommendations highlight high-risk batches.
""")

# ------------------------------
# 1️⃣ Kaggle API authentication
# ------------------------------
dataset_path = Path("data")
dataset_path.mkdir(exist_ok=True)

api = KaggleApi()
try:
    api.authenticate()
    st.success("✅ Kaggle API authenticated successfully!")
except Exception as e:
    st.error(f"Failed to authenticate with Kaggle API: {e}")
    st.stop()

# ------------------------------
# 2️⃣ Download or load Kaggle Dataset
# ------------------------------
st.sidebar.title("📊 Dataset Settings")
dataset_id = st.sidebar.text_input(
    "Kaggle Dataset ID",
    "stephengoldie/big-databiopharmaceutical-manufacturing"
)

@st.cache_data(show_spinner="Downloading dataset from Kaggle…")
def fetch_dataset(dataset_slug: str) -> pd.DataFrame:
    download_path = dataset_path / "kaggle_download"
    download_path.mkdir(exist_ok=True)
    api.dataset_download_files(dataset_slug, path=str(download_path), unzip=True)

    # Load first CSV/Excel found
    for root, _, files in os.walk(download_path):
        for f in files:
            fp = os.path.join(root, f)
            if f.endswith(".csv"):
                return pd.read_csv(fp)
            elif f.endswith((".xls", ".xlsx")):
                return pd.read_excel(fp)
    raise FileNotFoundError("No CSV or Excel files found in Kaggle dataset.")

if "df" not in st.session_state:
    try:
        st.session_state.df = fetch_dataset(dataset_id)
        st.success("Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Failed to download/load dataset: {e}")
        st.stop()

df = st.session_state.df

# ------------------------------
# 3️⃣ Dataset Summary & Fault Injection
# ------------------------------
st.sidebar.subheader("Fault Injection")
fault_chance = st.sidebar.slider(
    "Chance of Faulty Batch (%)",
    min_value=0, max_value=100, value=10, step=5
)
inject_faults = st.sidebar.checkbox("Enable Faulty Batches", value=True)

# ------------------------------
# 4️⃣ Train Predictive Model
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
# 5️⃣ Knowledge Graph Baseline
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
# 6️⃣ Event Simulation & Risk Scoring
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
        batch *= np.random.uniform(0.7, 0.9, size=len(batch))  # reduce yield
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
# 7️⃣ Sidebar Controls
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
# 8️⃣ Display Results
# ------------------------------
if st.session_state.simulated_batches:
    results_df = pd.DataFrame(st.session_state.simulated_batches)
    st.subheader("📈 Simulation Results")
    st.dataframe(results_df)

    # Recommendations
    st.subheader("💡 AI Recommendations")
    for batch in st.session_state.simulated_batches[-num_batches:]:
        st.write(f"Batch {batch['batch']} ({batch['event']}): {recommend_action(batch)}")

    # Knowledge Graph
    st.subheader("🌐 Knowledge Graph")
    net = Network(height="600px", width="100%", directed=True)
    for node in G.nodes(data=True):
        net.add_node(node[0], label=node[0], color=node[1].get("color", "blue"))
    for batch in st.session_state.simulated_batches:
        batch_node = f"Batch_{batch['batch']}"
        net.add_node(batch_node, label=batch_node, color=risk_color(batch['risk_score']))
        net.add_edge(batch_node, np.random.choice(ASSETS))
    html = net.generate_html()
    components.html(html, height=650, scrolling=True)

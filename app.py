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

# ------------------------------
# User Instructions
# ------------------------------
st.title("🧬 Cognitive Pharma Plant Digital Twin")

st.markdown("""
### Quick Start

1. **Upload Kaggle API Key** – Upload `kaggle.json` to load the dataset.  
2. **Load Dataset** – Contains normal and faulty batches. Faulty batches reduce yield and increase risk.  
3. **Enable Fault Simulation (Optional)** – Set “Chance of Faulty Batch (%)” to test AI recommendations.  
4. **Train Model** – Random Forest predicts batch yield; RMSE indicates prediction accuracy.  
5. **Simulate Batches** – Select event type, number of batches, and fault chance; click “Simulate Batches.”  
6. **Review Results** – Table shows predicted yield, risk, and event; Knowledge Graph visualizes batch-asset links; AI recommendations highlight high-risk batches.
""")

# ------------------------------
# 1️⃣ Set Kaggle API path and authenticate
# ------------------------------
dataset_path = os.path.join(os.getcwd(), "data")
os.makedirs(dataset_path, exist_ok=True)
os.environ['KAGGLE_CONFIG_DIR'] = dataset_path
kaggle_json_path = os.path.join(dataset_path, "kaggle.json")

# Upload kaggle.json if missing
if not os.path.exists(kaggle_json_path):
    st.warning("⚠️ kaggle.json not found. Please upload your Kaggle API key.")
    uploaded_file = st.file_uploader("Upload kaggle.json", type="json")
    if uploaded_file is not None:
        with open(kaggle_json_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        os.chmod(kaggle_json_path, 0o600)
        st.success("kaggle.json uploaded successfully! Please rerun the app.")
        st.stop()
    else:
        st.stop()

os.chmod(kaggle_json_path, 0o600)

# Authenticate Kaggle API
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

if "df" not in st.session_state:
    try:
        download_path = os.path.join(dataset_path, "kaggle_download")
        os.makedirs(download_path, exist_ok=True)
        api.dataset_download_files(dataset_id, path=download_path, unzip=True)
        st.write(f"Dataset downloaded to: {download_path}")

        # Load first CSV/Excel found
        dataframes = {}
        for root, dirs, files in os.walk(download_path):
            for f in files:
                file_path = os.path.join(root, f)
                if f.endswith('.csv'):
                    dataframes[f] = pd.read_csv(file_path)
                elif f.endswith(('.xls', '.xlsx')):
                    dataframes[f] = pd.read_excel(file_path)

        if not dataframes:
            st.error("No CSV or Excel files found in the Kaggle dataset.")
            st.stop()

        df_name = list(dataframes.keys())[0]
        st.session_state.df = dataframes[df_name]
        st.success(f"Loaded dataset: {df_name}")

    except Exception as e:
        st.error(f"Failed to download/load dataset: {e}")
        st.stop()

df = st.session_state.df

# ------------------------------
# 6️⃣ Dataset Summary
# ------------------------------
st.sidebar.subheader("Fault Injection")
fault_chance = st.sidebar.slider(
    "Chance of Faulty Batch (%)",
    min_value=0, max_value=100, value=10, step=5
)
inject_faults = st.sidebar.checkbox("Enable Faulty Batches", value=True)

# ------------------------------
# 3️⃣ Prepare Features & Train Predictive Model
# ------------------------------
numeric_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in dataset.")
    st.stop()

target_col = 'yield' if 'yield' in numeric_cols else numeric_cols[-1]
features = [c for c in numeric_cols if c != target_col]

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_col], test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
st.sidebar.write(f"✅ RMSE on test set: {rmse:.3f}")

# ------------------------------
# 4️⃣ Knowledge Graph & Baseline
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
# 5️⃣ Event Simulation & Risk Scoring
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
    
    # Inject random faults
    if inject_faults and np.random.rand() < (fault_chance / 100):
        batch *= np.random.uniform(0.7, 0.9, size=len(batch))  # reduce yield by 10-30%
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
# Sidebar: Inject Event & Batch Simulation
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

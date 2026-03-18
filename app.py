import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import requests

warnings.filterwarnings("ignore")

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="V21 Master Portfolio", layout="wide")
st.title("Institutional V21 Master Portfolio (75/25 Split)")

# ==========================================
# 2. FRONTEND: SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("Portfolio Parameters")
PORTFOLIO_CAPITAL = st.sidebar.number_input("Total Capital (INR)", value=600000, step=50000)
START_YEAR = st.sidebar.text_input("Start Date (YYYY-MM-DD)", value='2010-01-01')

st.sidebar.subheader("V21 Blue Box Sniper (25%)")
bb_risk = st.sidebar.slider("Risk Per Trade (%)", 0.01, 0.05, 0.02, 0.005)
bb_max_pos = st.sidebar.slider("Max Position Size (%)", 0.10, 0.50, 0.25, 0.05)
bb_slip = st.sidebar.number_input("BB Slippage/Leg", value=0.0013, format="%.4f")

st.sidebar.subheader("Mean Reversion Alpha (75%)")
mr_max_alloc = st.sidebar.slider("Max Alloc Per Trade (%)", 0.10, 1.00, 0.50, 0.05)
mr_cost = st.sidebar.number_input("MR Cost/Trade", value=0.0015, format="%.4f")
mr_max_hold = st.sidebar.slider("Max Hold (Days)", 1, 15, 5)

# Build Dynamic Configs
BB_CONFIG = {
    'initial_capital': PORTFOLIO_CAPITAL * 0.25,
    'risk_per_trade': bb_risk,    
    'max_position_size': bb_max_pos,
    'slippage_brokerage_leg': bb_slip
}

MR_CONFIG = {
    'initial_capital': PORTFOLIO_CAPITAL * 0.75,
    'max_allocation_per_trade': mr_max_alloc,
    'cost_per_trade': mr_cost,
    'max_hold': mr_max_hold
}

# ==========================================
# 3. DATA INGESTION (BULLETPROOF DOWNLOAD)
# ==========================================
@st.cache_data
def load_and_prep_data(start_year):
    local_file_name = "local_deployment_data.parquet"
    # Ensure this URL matches your GitHub v1.1 release exactly
    data_url = "https://github.com/sisodiyaatharva91-del/streamlit-test/releases/download/v1.1/NSE_15Y_Deployment_Ready.parquet"
    
    # 1. Download locally if it doesn't exist on the server yet
    if not os.path.exists(local_file_name):
        response = requests.get(data_url, stream=True)
        response.raise_for_status() 
        
        with open(local_file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    # 2. Read the stable local file
    df = pd.read_parquet(local_file_name)
    df['DATE'] = pd.to_datetime(df['DATE']).dt.tz_localize(None)
    df = df[df['DATE'] >= start_year]
    
    return df

# ==========================================
# 4. PORTFOLIO SIMULATOR (RUNS ON UI CHANGE)
# ==========================================
def run_sleeve_simulator(df, bb_config, mr_config):
    period_df = df.sort_values(by=['DATE', 'Turnover_SMA_50'], ascending=[True, False])
    grouped_by_date = period_df.groupby('DATE')
    
    start_date = df['DATE'].min()
    end_date = df['DATE'].max()
    calendar_dates = pd.date_range(start_date, end_date)
    
    bb_equity = bb_config['initial_capital']
    bb_cash = bb_equity
    active_bb_positions = {}
    
    mr_equity = mr_config['initial_capital']
    mr_cash = mr_equity
    active_mr_positions = {}
    
    trade_log = []
    equity_curve = []
    
    for current_date in calendar_dates:
        # Yield on Idle Cash (

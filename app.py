import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# ==========================================
# 1. PAGE CONFIG & CACHED DATA LOADING
# ==========================================
st.set_page_config(page_title="V21 Master Portfolio Quant Simulator", layout="wide")

@st.cache_resource
def load_data():
    """Loads the pre-computed parquet file ONCE into memory."""
    try:
        df = pd.read_parquet('Deployment_Ready.parquet')
        df['DATE'] = pd.to_datetime(df['DATE']).dt.date # Strip time for easier dict grouping
        return df
    except FileNotFoundError:
        st.error("🚨 Deployment_Ready.parquet not found! Please run the Colab Phase 1 script first.")
        st.stop()

df = load_data()

# ==========================================
# 2. SIDEBAR UI (THE HYPERPARAMETER SOLVER)
# ==========================================
st.sidebar.header("🎛️ Hyperparameter Solver")

with st.sidebar.expander("📅 Regime & Timeframe", expanded=True):
    min_date = df['DATE'].min()
    max_date = df['DATE'].max()
    start_date, end_date = st.slider("Testing Period", min_value=min_date, max_value=max_date, value=(min_date, max_date))

with st.sidebar.expander("💰 Capital & Friction", expanded=False):
    start_capital = st.number_input("Starting Capital (₹)", value=600000, step=100000)
    pledge_yield = st.number_input("Idle Cash Yield (Annual %)", value=7.5, step=0.5) / 100
    slippage_tax_pct = st.slider("Slippage & Taxes (%)", min_value=0.0, max_value=0.5, value=0.15, step=0.05)
    
    broker_model = st.selectbox("Brokerage Model", ["Discount (Flat Fee)", "Full-Service (%)"])
    if broker_model == "Discount (Flat Fee)":
        flat_fee = st.number_input("Flat Fee per leg (₹)", value=20.0, step=5.0)
    else:
        brokerage_pct = st.slider("Brokerage (%)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        min_brokerage = st.number_input("Min Brokerage per leg (₹)", value=25.0, step=5.0)

with st.sidebar.expander("⚖️ Capital Allocation", expanded=False):
    mr_weight_pct = st.slider("Mean Reversion Sleeve Weight (%)", 0, 100, 75, 5)
    bb_weight_pct = 100 - mr_weight_pct
    st.caption(f"Blue Box Sniper Weight: **{bb_weight_pct}%**")

with st.sidebar.expander("🎯 V21 Sniper Parameters", expanded=False):
    bb_risk = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
    bb_max_pos = st.slider("Max Position Size (%)", 10, 100, 25, 5) / 100
    bb_target_atr = st.slider("Take Profit (ATR Multiplier)", 1.0, 6.0, 4.0, 0.5)
    bb_stop_atr = st.slider("Stop Loss (ATR Multiplier)", 1.0, 4.0, 2.5, 0.5)
    min_breadth = st.slider("Min Market Breadth (%)", 0, 100, 50, 5) / 100
    min_liquidity = st.slider("Min Daily Liquidity Percentile", 50, 95, 85, 5) / 100

with st.sidebar.expander("📉 Mean Reversion Parameters", expanded=False):
    mr_max_alloc = st.slider("Max Allocation per Panic (%)", 10, 100, 50, 5) / 100
    mr_time_stop = st.slider("MR Time Stop (Days)", 3, 20, 5, 1)

# ==========================================
# 3. HELPER FUNCTIONS (Friction & Metrics)
# ==========================================
def calculate_friction(gross_value):
    slippage_tax = gross_value * (slippage_tax_pct / 100)
    if broker_model == "Discount (Flat Fee)":
        brokerage = flat_fee
    else:
        brokerage = max(min_brokerage, gross_value * (brokerage_pct / 100))
    return slippage_tax + brokerage

def calculate_metrics(trade_log, equity_curve, initial_cap):
    if not trade_log or len(equity_curve) == 0:
        return {"Trades": 0, "Win Rate %": 0, "Profit Factor": 0, "Max DD %": 0, "CAGR %": 0, "Net Return %": 0}
    
    df_trades = pd.DataFrame(trade_log)
    wins = df_trades[df_trades['PnL INR'] > 0]
    losses = df_trades[df_trades['PnL INR'] <= 0]
    
    win_rate = len(wins) / len(df_trades) * 100
    pf = wins['PnL INR'].sum() / abs(losses['PnL INR'].sum()) if not losses.empty and losses['PnL INR'].sum() != 0 else 0
    
    df_eq = pd.DataFrame(equity_curve)
    ending = df_eq['Combined_Equity'].iloc[-1]
    net_ret = ((ending - initial_cap) / initial_cap) * 100
    
    dd = ((df_eq['Combined_Equity'] - df_eq['Combined_Equity'].cummax()) / df_eq['Combined_Equity'].cummax()).min() * 100
    
    days = (df_eq['DATE'].iloc[-1] - df_eq['DATE'].iloc[0]).days
    years = days / 365.25
    cagr = ((ending / initial_cap) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    return {"Trades": len(df_trades), "Win Rate %": win_rate, "Profit Factor": pf, "Max DD %": dd, "CAGR %": cagr, "Net Return %": net_ret}

# ==========================================
# 4. THE ZERO-COPY SIMULATOR ENGINE
# ==========================================
def run_simulation():
    # 1. Filter Data Once
    sim_df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    if sim_df.empty: return [], []
    
    # 2. Convert to ultra-fast dictionary grouped by date (Avoids pandas overhead in loop)
    # Using python dicts for O(1) lookups
    records = sim_df.to_dict('records')
    daily_data = {}
    for r in records:
        d = r['DATE']
        if d not in daily_data: daily_data[d] = []
        daily_data[d].append(r)
        
    calendar_dates = sorted(list(daily_data.keys()))
    if not calendar_dates: return [], []

    # 3. Initialize Wallets & Trackers
    bb_equity = start_capital * (bb_weight_pct / 100)
    bb_cash = bb_equity
    active_bb = {}
    
    mr_equity = start_capital * (mr_weight_pct / 100)
    mr_cash = mr_equity
    active_mr = {}
    
    trade_log = []
    equity_curve = []
    
    daily_yield_rate = pledge_yield / 365

    # 4. The Daily Ledger Loop
    for current_date in calendar_dates:
        todays_rows = daily_data[current_date]
        
        # Accrue Yield
        bb_yld = max(0, bb_cash) * daily_yield_rate
        bb_cash += bb_yld
        bb_equity += bb_yld
        
        mr_yld = max(0, mr_cash) * daily_yield_rate
        mr_cash += mr_yld
        mr_equity += mr_yld
        
        # Convert today's rows into a quick lookup dict for exits
        today_lookup = {r['SYMBOL']: r for r in todays_rows}
        
        # ==================================
        # EXITS: BLUE BOX
        # ==================================
        bb_removals = []
        for sym, pos in active_bb.items():
            if sym not in today_lookup: continue
            row = today_lookup[sym]
            
            days_held = (current_date - pos['entry_date']).days
            exit_triggered, exit_price, exit_reason = False, 0, ""
            
            if row['LOW'] <= pos['stop_price']:
                exit_triggered, exit_price, exit_reason = True, pos['stop_price'], f"Stop Swept (-{bb_stop_atr} ATR)"
            elif row['HIGH'] >= pos['target_price']:
                exit_triggered, exit_price, exit_reason = True, pos['target_price'], f"Target Hit (+{bb_target_atr} ATR)"
            elif row['BB_Exhaustion_Today'] and row['CLOSE'] > pos['entry_price']:
                exit_triggered, exit_price, exit_reason = True, row['CLOSE'], "Momentum Exhaustion"
                
            if exit_triggered:
                revenue = pos['shares'] * exit_price
                friction = calculate_friction(revenue)
                net_revenue = revenue - friction
                
                profit = net_revenue - pos['net_cost']
                bb_cash += net_revenue
                bb_equity += profit
                
                trade_log.append({
                    'Strategy': 'Blue Box', 'Symbol': sym, 'Entry Date': pos['entry_date'], 'Exit Date': current_date,
                    'Days Held': days_held, 'Exit Reason': exit_reason, 'PnL INR': profit, 
                    'Return %': (profit / pos['net_cost']) * 100
                })
                bb_removals.append(sym)
        for sym in bb_removals: del active_bb[sym]
        
        # ==================================
        # EXITS: MEAN REVERSION
        # ==================================
        mr_removals = []
        for sym, pos in active_mr.items():
            if sym not in today_lookup: continue
            row = today_lookup[sym]
            
            pos['trading_days'] += 1
            exit_triggered, exit_price, exit_reason = False, 0, ""
            
            if row['CLOSE'] > row['SMA_5']:
                exit_triggered, exit_price, exit_reason = True, row['CLOSE'], "Target Hit (SMA 5)"
            elif pos['trading_days'] >= mr_time_stop:
                exit_triggered, exit_price, exit_reason = True, row['CLOSE'], f"Time Stop ({mr_time_stop} Days)"
                
            if exit_triggered:
                revenue = pos['shares'] * exit_price
                friction = calculate_friction(revenue)
                net_revenue = revenue - friction
                
                profit = net_revenue - pos['net_cost']
                mr_cash += net_revenue
                mr_equity += profit
                
                trade_log.append({
                    'Strategy': 'Mean Reversion', 'Symbol': sym, 'Entry Date': pos['entry_date'], 'Exit Date': current_date,
                    'Days Held': pos['trading_days'], 'Exit Reason': exit_reason, 'PnL INR': profit,
                    'Return %': (profit / pos['net_cost']) * 100
                })
                mr_removals.append(sym)
        for sym in mr_removals: del active_mr[sym]
        
        # ==================================
        # ENTRIES: MEAN REVERSION
        # ==================================
        mr_candidates = [r for r in todays_rows if r['MR_Base_Signal'] and sym not in active_mr]
        if mr_candidates and mr_cash > 0:
            alloc_per_signal = min(mr_equity * mr_max_alloc, mr_cash / len(mr_candidates))
            for row in mr_candidates:
                entry_price = row['CLOSE'] # MOC Order
                shares = int(alloc_per_signal / entry_price)
                if shares > 0:
                    gross_cost = shares * entry_price
                    net_cost = gross_cost + calculate_friction(gross_cost)
                    if mr_cash >= net_cost:
                        mr_cash -= net_cost
                        active_mr[row['SYMBOL']] = {'entry_date': current_date, 'entry_price': entry_price, 'shares': shares, 'net_cost': net_cost, 'trading_days': 0}

        # ==================================
        # ENTRIES: BLUE BOX SNIPER
        # ==================================
        current_breadth = todays_rows[0]['Market_Breadth'] if todays_rows else 0
        if current_breadth >= min_breadth and bb_cash > 0:
            
            # Dynamic Filter applied from UI
            bb_candidates = [r for r in todays_rows if r['BB_Enter_Today'] 
                             and r['Daily_Turnover_Rank'] >= min_liquidity 
                             and r['Market_Breadth'] >= min_breadth
                             and r['SYMBOL'] not in active_bb]
                             
            for row in bb_candidates:
                if bb_cash <= 0: break
                entry_price = row['OPEN']
                atr = row['Target_ATR']
                if atr <= 0: continue
                
                stop_loss = entry_price - (bb_stop_atr * atr)
                target_price = entry_price + (bb_target_atr * atr)
                
                risk_per_share = entry_price - stop_loss
                if risk_per_share <= 0: continue
                
                shares = int((bb_equity * bb_risk) / risk_per_share)
                max_shares = int((bb_equity * bb_max_pos) / entry_price)
                shares = min(shares, max_shares)
                
                if shares > 0:
                    gross_cost = shares * entry_price
                    net_cost = gross_cost + calculate_friction(gross_cost)
                    if bb_cash >= net_cost:
                        bb_cash -= net_cost
                        active_bb[row['SYMBOL']] = {
                            'entry_date': current_date, 'entry_price': entry_price, 
                            'stop_price': stop_loss, 'target_price': target_price,
                            'shares': shares, 'net_cost': net_cost
                        }
                        
        equity_curve.append({
            'DATE': current_date, 'BB_Equity': bb_equity, 'MR_Equity': mr_equity, 'Combined_Equity': bb_equity + mr_equity
        })
        
    return trade_log, equity_curve

# ==========================================
# 5. UI EXECUTION & VISUALIZATION
# ==========================================
st.title("📈 V21 Master Portfolio Simulator")

# Init Session State for History
if 'run_history' not in st.session_state:
    st.session_state.run_history = []

col1, col2 = st.columns([4, 1])
with col2:
    run_btn = st.button("🚀 Run Simulation", use_container_width=True, type="primary")

if run_btn:
    with st.spinner("Running 15-Year Zero-Copy Ledger..."):
        trade_log, equity_curve = run_simulation()
        metrics = calculate_metrics(trade_log, equity_curve, start_capital)
        
        # Save to History
        history_entry = {
            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
            "CAGR %": f"{metrics['CAGR %']:.2f}%",
            "Max DD %": f"{metrics['Max DD %']:.2f}%",
            "Win Rate %": f"{metrics['Win Rate %']:.2f}%",
            "Split": f"MR {mr_weight_pct}% / BB {bb_weight_pct}%",
            "BB Risk/Stop/Tgt": f"{bb_risk*100}% / {bb_stop_atr} / {bb_target_atr}",
            "MR Hold": f"{mr_time_stop}d"
        }
        st.session_state.run_history.insert(0, history_entry)
        
        st.session_state.current_metrics = metrics
        st.session_state.current_eq = pd.DataFrame(equity_curve)
        st.session_state.current_trades = pd.DataFrame(trade_log)

# Ensure UI doesn't disappear on re-renders
if 'current_metrics' in st.session_state:
    m = st.session_state.current_metrics
    eq_df = st.session_state.current_eq
    trades_df = st.session_state.current_trades
    
    # ---------------- METRICS ROW ----------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"{m['CAGR %']:.2f}%")
    c2.metric("Max Drawdown", f"{m['Max DD %']:.2f}%")
    c3.metric("Win Rate", f"{m['Win Rate %']:.1f}%")
    c4.metric("Profit Factor", f"{m['Profit Factor']:.2f}")

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📝 Trade Log", "📜 Experiment History"])

    with tab1:
        if not eq_df.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Equity Curves
            fig.add_trace(go.Scatter(x=eq_df['DATE'], y=eq_df['BB_Equity'], name='V21 Sniper', line=dict(color='#00d2ff')), row=1, col=1)
            fig.add_trace(go.Scatter(x=eq_df['DATE'], y=eq_df['MR_Equity'], name='Mean Reversion', line=dict(color='#ff007c')), row=1, col=1)
            fig.add_trace(go.Scatter(x=eq_df['DATE'], y=eq_df['Combined_Equity'], name='Master Portfolio', line=dict(color='#00ff88', width=2.5)), row=1, col=1)
            
            # Drawdown
            eq_df['Peak'] = eq_df['Combined_Equity'].cummax()
            eq_df['Drawdown'] = ((eq_df['Combined_Equity'] - eq_df['Peak']) / eq_df['Peak']) * 100
            fig.add_trace(go.Scatter(x=eq_df['DATE'], y=eq_df['Drawdown'], name='Drawdown', fill='tozeroy', line=dict(color='red')), row=2, col=1)
            
            fig.update_layout(height=600, template="plotly_dark", title_text="Portfolio Performance", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.dataframe(trades_df, use_container_width=True)
        csv = trades_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Trade Log (CSV)", data=csv, file_name="v21_trades.csv", mime="text/csv")

    with tab3:
        hist_df = pd.DataFrame(st.session_state.run_history)
        st.dataframe(hist_df, use_container_width=True)
        if not hist_df.empty:
            hist_csv = hist_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Export History (CSV)", data=hist_csv, file_name="experiment_history.csv", mime="text/csv")
else:
    st.info("👈 Adjust parameters in the sidebar and click **Run Simulation** to begin.")

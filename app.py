import os
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    data_url = "https://github.com/sisodiyaatharva91-del/streamlit-test/releases/download/v1.1/NSE_15Y_Deployment_Ready.parquet"
    
    # 1. Download using the OS native wget tool
    if not os.path.exists(local_file_name):
        try:
            subprocess.run(["wget", "-q", "-O", local_file_name, data_url], check=True)
        except subprocess.CalledProcessError as e:
            st.error("Failed to download the dataset via wget. Check if the GitHub Release URL is exactly correct.")
            raise e
                
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
        # Yield on Idle Cash (7.5%)
        bb_yield = max(0, bb_cash) * (0.075 / 365)
        bb_cash += bb_yield
        bb_equity += bb_yield
        
        mr_yield = max(0, mr_cash) * (0.075 / 365)
        mr_cash += mr_yield
        mr_equity += mr_yield
        
        if current_date in grouped_by_date.groups:
            day_df = grouped_by_date.get_group(current_date)
            active_symbols = set(active_bb_positions.keys()).union(set(active_mr_positions.keys()))
            exit_df = day_df[day_df['SYMBOL'].isin(active_symbols)]
            
            # --- BB EXITS ---
            bb_removals = []
            for symbol, pos in active_bb_positions.items():
                row_match = exit_df[exit_df['SYMBOL'] == symbol]
                if row_match.empty: continue
                row = row_match.iloc[0]
                
                days_in_trade = (current_date - pos['entry_date']).days
                exit_triggered, exit_price, exit_type = False, 0, ""
                
                if row['LOW'] <= pos['stop_price']:
                    exit_triggered, exit_price, exit_type = True, pos['stop_price'], "Fixed Stop Swept (-2.5 ATR)"
                elif row['HIGH'] >= pos['target_price']:
                    exit_triggered, exit_price, exit_type = True, pos['target_price'], "Velocity Target Hit (+4.0 ATR)"
                elif row['BB_Exhaustion_Today'] and row['CLOSE'] > pos['entry_price']:
                    exit_triggered, exit_price, exit_type = True, row['CLOSE'], "Momentum Exhaustion (Profit Locked)"
                    
                if exit_triggered:
                    revenue = pos['shares'] * exit_price
                    cost = revenue * bb_config['slippage_brokerage_leg']
                    capital_returned = revenue - cost
                    profit_inr = capital_returned - (pos['initial_investment'] + pos['entry_cost'])
                    raw_pct = (profit_inr / pos['initial_investment']) * 100
                    
                    bb_cash += capital_returned
                    bb_equity += profit_inr
                    
                    trade_log.append({
                        'Strategy': 'Blue Box', 'Symbol': symbol,
                        'Entry Date': pos['entry_date'], 'Exit Date': current_date,
                        'Days in Trade': days_in_trade, 'Exit Reason': exit_type,
                        'Capital Allocated': round(pos['initial_investment'], 2),
                        'Net PnL %': round(raw_pct, 2), 'Net Profit INR': round(profit_inr, 2)
                    })
                    bb_removals.append(symbol)
            for sym in bb_removals: del active_bb_positions[sym]
            
            # --- MR EXITS ---
            mr_removals = []
            for symbol, pos in active_mr_positions.items():
                row_match = exit_df[exit_df['SYMBOL'] == symbol]
                if row_match.empty: continue
                row = row_match.iloc[0]
                
                pos['trading_days'] += 1
                exit_triggered, exit_price, exit_type = False, 0, ""
                
                if row['CLOSE'] > row['SMA_5']:
                    exit_price = row['CLOSE']
                    exit_triggered, exit_type = True, "Target Hit (SMA 5)"
                elif pos['trading_days'] >= mr_config['max_hold']:
                    exit_price = row['CLOSE']
                    exit_triggered, exit_type = True, f"Time Stop (T+{mr_config['max_hold']})"
                    
                if exit_triggered:
                    revenue = pos['shares'] * exit_price
                    cost = pos['initial_investment'] * mr_config['cost_per_trade']
                    capital_returned = revenue - cost
                    profit_inr = capital_returned - pos['initial_investment']
                    raw_pct = (profit_inr / pos['initial_investment']) * 100
                    
                    mr_cash += capital_returned
                    mr_equity += profit_inr
                    
                    trade_log.append({
                        'Strategy': 'Mean Reversion', 'Symbol': symbol,
                        'Entry Date': pos['entry_date'], 'Exit Date': current_date,
                        'Days in Trade': (current_date - pos['entry_date']).days,
                        'Exit Reason': exit_type,
                        'Capital Allocated': round(pos['initial_investment'], 2),
                        'Net PnL %': round(raw_pct, 2), 'Net Profit INR': round(profit_inr, 2)
                    })
                    mr_removals.append(symbol)
            for sym in mr_removals: del active_mr_positions[sym]
            
            # --- MR ENTRIES ---
            mr_buy_signals = day_df[day_df['MR_Signal'] == True]
            if not mr_buy_signals.empty and mr_cash > 0:
                num_signals = len(mr_buy_signals)
                target_alloc = mr_equity * mr_config['max_allocation_per_trade']
                alloc_per_signal = min(target_alloc, mr_cash / num_signals)
                
                for _, row in mr_buy_signals.iterrows():
                    symbol = row['SYMBOL']
                    if symbol in active_mr_positions: continue
                    
                    entry_price = row['CLOSE']
                    shares = int(alloc_per_signal / entry_price)
                    
                    if shares > 0:
                        actual_invested = shares * entry_price
                        mr_cash -= actual_invested
                        active_mr_positions[symbol] = {
                            'entry_date': current_date, 'entry_price': entry_price,
                            'shares': shares, 'initial_investment': actual_invested,
                            'trading_days': 0
                        }
                        
            # --- BB ENTRIES ---
            current_breadth = day_df['Market_Breadth'].iloc[0] if not day_df.empty else 0
            if current_breadth > 0.65: max_bb_positions = 4  
            elif current_breadth > 0.50: max_bb_positions = 2  
            else: max_bb_positions = 0  
                
            bb_buy_signals = day_df[day_df['BB_Enter_Today'] == True]
            if not bb_buy_signals.empty and bb_cash > 0 and len(active_bb_positions) < max_bb_positions:
                # We sort by Target_ATR as a proxy since RSI_14 was dropped in the lean dataset
                bb_buy_signals = bb_buy_signals.sort_values(by='Target_ATR', ascending=False)
                
                for _, row in bb_buy_signals.iterrows():
                    if len(active_bb_positions) >= max_bb_positions or bb_cash <= 0: break
                    
                    symbol = row['SYMBOL']
                    if symbol in active_bb_positions: continue
                    
                    entry_price = row['OPEN'] 
                    atr_14 = row['Target_ATR']
                    
                    if pd.isna(atr_14) or atr_14 == 0: continue
                    
                    target_price = entry_price + (4.0 * atr_14)
                    stop_loss = entry_price - (2.5 * atr_14)
                    risk_per_share = entry_price - stop_loss
                    
                    if risk_per_share <= 0: continue
                    
                    risk_amount = bb_equity * bb_config['risk_per_trade']
                    shares = int(risk_amount / risk_per_share)
                    max_shares_allowed = int((bb_equity * bb_config['max_position_size']) / entry_price)
                    shares = min(shares, max_shares_allowed)
                    
                    if shares > 0:
                        actual_invested = shares * entry_price
                        entry_cost = actual_invested * bb_config['slippage_brokerage_leg']
                        total_deduction = actual_invested + entry_cost
                        
                        if bb_cash >= total_deduction:
                            bb_cash -= total_deduction
                            bb_equity -= entry_cost  
                            
                            active_bb_positions[symbol] = {
                                'entry_date': current_date, 'entry_price': entry_price,
                                'stop_price': stop_loss, 'target_price': target_price,
                                'shares': shares, 'initial_investment': actual_invested,
                                'entry_cost': entry_cost
                            }
                        
        equity_curve.append({
            'DATE': current_date, 
            'BB_Equity': bb_equity, 'MR_Equity': mr_equity,
            'Combined_Equity': bb_equity + mr_equity
        })
        
    return pd.DataFrame(trade_log), pd.DataFrame(equity_curve)

# ==========================================
# 5. EXECUTION & DISPLAY
# ==========================================
def calculate_metrics(trades_df, equity_df, start_cap, col_name='Combined_Equity'):
    if trades_df.empty or equity_df.empty: return {}
    wins = trades_df[trades_df['Net Profit INR'] > 0]
    losses = trades_df[trades_df['Net Profit INR'] <= 0]
    win_rate = len(wins) / len(trades_df) * 100
    pf = wins['Net Profit INR'].sum() / abs(losses['Net Profit INR'].sum()) if not losses.empty and losses['Net Profit INR'].sum() != 0 else float('inf')
    ending = equity_df[col_name].iloc[-1]
    dd = ((equity_df[col_name] - equity_df[col_name].cummax()) / equity_df[col_name].cummax()).min() * 100
    net_ret = ((ending - start_cap) / start_cap) * 100
    days = (pd.to_datetime(equity_df['DATE'].iloc[-1]) - pd.to_datetime(equity_df['DATE'].iloc[0])).days
    years = max(1, days / 365.25)
    cagr = ((ending / start_cap) ** (1 / years) - 1) * 100
    return {"Trades": len(trades_df), "Win Rate %": round(win_rate, 2), "Profit Factor": round(pf, 2), 
            "Max DD %": round(dd, 2), "Net Return %": round(net_ret, 2), "CAGR %": round(cagr, 2), "Ending Value": round(ending, 2)}

try:
    with st.spinner("Downloading & Loading 15Y Dataset (This happens only once)..."):
        raw_df = load_and_prep_data(START_YEAR)

    with st.spinner("Running Unleveraged Master Ledger..."):
        combined_logs, merged_eq = run_sleeve_simulator(raw_df, BB_CONFIG, MR_CONFIG)

    # --- METRICS UI ---
    st.markdown("### 📊 Performance Metrics")
    bb_metrics = calculate_metrics(combined_logs[combined_logs['Strategy'] == 'Blue Box'], merged_eq, BB_CONFIG['initial_capital'], 'BB_Equity')
    mr_metrics = calculate_metrics(combined_logs[combined_logs['Strategy'] == 'Mean Reversion'], merged_eq, MR_CONFIG['initial_capital'], 'MR_Equity')
    comb_metrics = calculate_metrics(combined_logs, merged_eq, PORTFOLIO_CAPITAL, 'Combined_Equity')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Combined CAGR", f"{comb_metrics.get('CAGR %', 0):.2f}%")
    col2.metric("Combined Max DD", f"{comb_metrics.get('Max DD %', 0):.2f}%")
    col3.metric("Combined Win Rate", f"{comb_metrics.get('Win Rate %', 0):.2f}%")
    col4.metric("Profit Factor", f"{comb_metrics.get('Profit Factor', 0):.2f}")

    st.markdown("---")
    
    # --- PLOTLY INTERACTIVE CHARTS ---
    st.markdown("### 📈 Interactive Equity & Drawdown Curve")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Top: Equity
    fig.add_trace(go.Scatter(x=merged_eq['DATE'], y=merged_eq['Combined_Equity'], name='Master Portfolio', line=dict(color='#00ff88', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=merged_eq['DATE'], y=merged_eq['BB_Equity'], name='V21 Sniper (25%)', line=dict(color='#00d2ff', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=merged_eq['DATE'], y=merged_eq['MR_Equity'], name='Mean Reversion (75%)', line=dict(color='#ff007c', width=1)), row=1, col=1)

    # Bottom: Drawdown
    combined_dd = (merged_eq['Combined_Equity'] - merged_eq['Combined_Equity'].cummax()) / merged_eq['Combined_Equity'].cummax() * 100
    fig.add_trace(go.Scatter(x=merged_eq['DATE'], y=combined_dd, fill='tozeroy', name='Master DD', line=dict(color='red', width=1)), row=2, col=1)

    fig.update_layout(height=600, template='plotly_dark', hovermode='x unified', margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- EXPORT ---
    st.markdown("### 🗄️ Trade Logs")
    st.dataframe(combined_logs.tail(10))
    st.download_button("Download Full Trade Log (CSV)", data=combined_logs.to_csv(index=False).encode('utf-8'), file_name="master_trade_log.csv", mime="text/csv")

except Exception as e:
    st.error(f"Error: {e}. Please check the trace for details.")

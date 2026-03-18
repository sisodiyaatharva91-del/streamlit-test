import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(page_title="V21 Master Portfolio", layout="wide")
st.title("Institutional V21 Master Portfolio (75/25 Split)")

# ==========================================
# 1. FRONTEND: SIDEBAR CONFIGURATION
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
# 2. SIGNAL GENERATORS (CACHED)
# ==========================================
# (Your original apply_blue_box_logic and apply_mean_reversion_logic go here exactly as they were)
def apply_blue_box_logic(df):
    df['EMA_10'] = df.groupby('SYMBOL')['CLOSE'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    df['EMA_20'] = df.groupby('SYMBOL')['CLOSE'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['EMA_50'] = df.groupby('SYMBOL')['CLOSE'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df['EMA_150'] = df.groupby('SYMBOL')['CLOSE'].transform(lambda x: x.ewm(span=150, adjust=False).mean())
    df['EMA_200'] = df.groupby('SYMBOL')['CLOSE'].transform(lambda x: x.ewm(span=200, adjust=False).mean())

    df['Turnover'] = df['CLOSE'] * df['VOLUME']
    df['Turnover_SMA_50'] = df.groupby('SYMBOL')['Turnover'].transform(lambda x: x.rolling(window=50).mean())
    df['Daily_Turnover_Rank'] = df.groupby('DATE')['Turnover_SMA_50'].rank(pct=True)
    df['Is_Liquid'] = df['Daily_Turnover_Rank'] >= 0.85 
    
    df['Above_50EMA_Liquid'] = (df['CLOSE'] > df['EMA_50']) & df['Is_Liquid']
    df['Liquid_Market_Size'] = df.groupby('DATE')['Is_Liquid'].transform('sum')
    df['Uptrend_Count'] = df.groupby('DATE')['Above_50EMA_Liquid'].transform('sum')
    df['Market_Breadth'] = np.where(df['Liquid_Market_Size'] > 0, df['Uptrend_Count'] / df['Liquid_Market_Size'], 0)

    df['Prev_Close'] = df.groupby('SYMBOL')['CLOSE'].shift(1)
    df['Prev_High'] = df.groupby('SYMBOL')['HIGH'].shift(1)
    df['Prev_Volume'] = df.groupby('SYMBOL')['VOLUME'].shift(1)
    df['TR'] = np.maximum(df['HIGH'] - df['LOW'], np.maximum(abs(df['HIGH'] - df['Prev_Close']), abs(df['LOW'] - df['Prev_Close'])))
    df['ATR_14'] = df.groupby('SYMBOL')['TR'].transform(lambda x: x.rolling(window=14).mean())

    df['High_50d'] = df.groupby('SYMBOL')['HIGH'].transform(lambda x: x.rolling(window=50).max())
    cond_proximity = df['CLOSE'] >= (df['High_50d'] * 0.85) 

    delta = df.groupby('SYMBOL')['CLOSE'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ema_up_14 = up.groupby(df['SYMBOL']).transform(lambda x: x.ewm(com=13, adjust=False).mean())
    ema_down_14 = down.groupby(df['SYMBOL']).transform(lambda x: x.ewm(com=13, adjust=False).mean())
    df['RSI_14'] = 100 - (100 / (1 + (ema_up_14 / (ema_down_14 + 1e-8))))

    ema_up_3 = up.groupby(df['SYMBOL']).transform(lambda x: x.ewm(com=2, adjust=False).mean())
    ema_down_3 = down.groupby(df['SYMBOL']).transform(lambda x: x.ewm(com=2, adjust=False).mean())
    df['RSI_3'] = 100 - (100 / (1 + (ema_up_3 / (ema_down_3 + 1e-8))))

    df['RSI_above_70'] = (df['RSI_14'] >= 70).astype(int)
    df['Recent_Strong_Momentum'] = df.groupby('SYMBOL')['RSI_above_70'].transform(lambda x: x.rolling(window=15).max()) == 1
    df['RSI_3_Oversold'] = (df['RSI_3'] <= 30).astype(int)
    df['Recent_Pullback'] = df.groupby('SYMBOL')['RSI_3_Oversold'].transform(lambda x: x.rolling(window=4).max()) == 1

    cond_ignition = (df['CLOSE'] > df['Prev_High']) & (df['CLOSE'] > df['EMA_20']) & (df['VOLUME'] > df['Prev_Volume'])
    
    cond_liquidity = df['Is_Liquid']
    cond_volatility = (df['ATR_14'] / df['CLOSE']) < 0.05 
    cond_trend = (df['EMA_10'] > df['EMA_20']) & (df['EMA_20'] > df['EMA_50']) & (df['EMA_50'] > df['EMA_150'])
    cond_regime = df['Market_Breadth'] > 0.50 

    df['BB_Signal'] = (cond_liquidity & cond_volatility & cond_trend & 
                       df['Recent_Strong_Momentum'] & df['Recent_Pullback'] & 
                       cond_ignition & cond_proximity & (df['CLOSE'] >= 50) & cond_regime)

    df['Exhaustion_Flag'] = (df['RSI_3'] >= 85) & (df['CLOSE'] < df['Prev_Close'])

    with pd.option_context('future.no_silent_downcasting', True):
        df['BB_Enter_Today'] = df.groupby('SYMBOL')['BB_Signal'].shift(1).fillna(False).infer_objects(copy=False)
        df['BB_Exhaustion_Today'] = df.groupby('SYMBOL')['Exhaustion_Flag'].shift(1).fillna(False).infer_objects(copy=False)
        df['Target_ATR'] = df.groupby('SYMBOL')['ATR_14'].shift(1)
        
    return df

def apply_mean_reversion_logic(df):
    if 'Turnover' not in df.columns:
        df['Turnover'] = df['CLOSE'] * df['VOLUME']
        
    df['Avg_Turnover_20'] = df.groupby('SYMBOL')['Turnover'].transform(lambda x: x.rolling(20).mean())
    df['Daily_Turnover_Rank'] = df.groupby('DATE')['Avg_Turnover_20'].rank(pct=True)
    df['MR_Is_Liquid'] = df['Daily_Turnover_Rank'] >= 0.70
    
    df['Is_Quality'] = df['CLOSE'] >= 100
    df['SMA_200'] = df.groupby('SYMBOL')['CLOSE'].transform(lambda x: x.rolling(200).mean())
    df['SMA_5'] = df.groupby('SYMBOL')['CLOSE'].transform(lambda x: x.rolling(5).mean())

    delta = df.groupby('SYMBOL')['CLOSE'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.groupby(df['SYMBOL']).transform(lambda x: x.ewm(com=1, adjust=False).mean())
    ema_down = down.groupby(df['SYMBOL']).transform(lambda x: x.ewm(com=1, adjust=False).mean())
    df['RSI_2'] = 100 - (100 / (1 + (ema_up / (ema_down + 1e-8))))

    df['SMA_20'] = df.groupby('SYMBOL')['CLOSE'].transform(lambda x: x.rolling(20).mean())
    df['STD_20'] = df.groupby('SYMBOL')['CLOSE'].transform(lambda x: x.rolling(20).std())
    
    df['Lower_BB_3'] = df['SMA_20'] - (3.0 * df['STD_20'])
    df['IBS'] = (df['CLOSE'] - df['LOW']) / (df['HIGH'] - df['LOW'] + 1e-8)

    if 'Market_Breadth' in df.columns:
        df['Regime_Permitted'] = df['Market_Breadth'] < 0.50
    else:
        df['Regime_Permitted'] = True

    df['MR_Signal'] = (
        (df['MR_Is_Liquid'] == True) & (df['Is_Quality'] == True) & (df['Regime_Permitted'] == True) &
        (df['CLOSE'] > df['SMA_200']) & (df['RSI_2'] < 3) & 
        (df['CLOSE'] < df['Lower_BB_3']) & (df['IBS'] < 0.2)
    )
    
    signals = df[df['MR_Signal']].copy()
    signals['Daily_Rank'] = signals.groupby('DATE')['IBS'].rank(method='first', ascending=True)
    valid_signals_mask = signals['Daily_Rank'] <= 5
    df['MR_Signal'] = False
    df.loc[signals[valid_signals_mask].index, 'MR_Signal'] = True
    return df

# ==========================================
# 3. DATA INGESTION (HEAVY LIFTING CACHED)
# ==========================================
@st.cache_data
def load_and_prep_data(start_year):
    # Paste your NEW v1.1 GitHub Release URL here
    data_url = "https://github.com/sisodiyaatharva91-del/streamlit-test/releases/download/v1.1/NSE_15Y_Deployment_Ready.parquet"
    
    # Notice how we completely removed the apply_blue_box and mean_reversion functions here!
    df = pd.read_parquet(data_url)
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
        # Yield on Idle Cash
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
                bb_buy_signals = bb_buy_signals.sort_values(by='RSI_14', ascending=False)
                
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
    with st.spinner("Loading Parquet Data & Calculating Market Regimes... (This happens only once)"):
        # Note: Ensure the file name exactly matches the one in your root directory
        raw_df = load_and_prep_data(START_YEAR)

    with st.spinner("Running Unleveraged Master Ledger..."):
        combined_logs, merged_eq = run_sleeve_simulator(raw_df, BB_CONFIG, MR_CONFIG)

    # --- METRICS UI ---
    st.markdown("### 📊 Performance Metrics")
    bb_metrics = calculate_metrics(combined_logs[combined_logs['Strategy'] == 'Blue Box'], merged_eq, BB_CONFIG['initial_capital'], 'BB_Equity')
    mr_metrics = calculate_metrics(combined_logs[combined_logs['Strategy'] == 'Mean Reversion'], merged_eq, MR_CONFIG['initial_capital'], 'MR_Equity')
    comb_metrics = calculate_metrics(combined_logs, merged_eq, PORTFOLIO_CAPITAL, 'Combined_Equity')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Combined CAGR", f"{comb_metrics.get('CAGR %', 0)}%")
    col2.metric("Combined Max DD", f"{comb_metrics.get('Max DD %', 0)}%")
    col3.metric("Combined Win Rate", f"{comb_metrics.get('Win Rate %', 0)}%")
    col4.metric("Profit Factor", f"{comb_metrics.get('Profit Factor', 0)}")

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
    st.error(f"Error: {e}. Please check if 'NSE_EQ_2020_Fast.parquet' is correctly uploaded to your repository.")

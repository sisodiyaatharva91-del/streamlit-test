import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Backtest Engine", layout="wide")
st.title("Quantitative Strategy Dashboard")

# ---------------------------------------------------------
# 2. CACHED DATA LOADING
# ---------------------------------------------------------
@st.cache_data
def load_data(years):
    """
    Simulates loading market data. 
    Replace this with your actual pandas read_csv/SQL logic.
    """
    days = years * 252 # Trading days
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq='B')
    # Generate random price walk
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, days)))
    return pd.DataFrame({'Date': dates, 'Close': prices})

# ---------------------------------------------------------
# 3. SIDEBAR: INPUT PARAMETERS
# ---------------------------------------------------------
st.sidebar.header("Strategy Parameters")
capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=10000)
years = st.sidebar.slider("Backtest Period (Years)", min_value=1, max_value=15, value=5)
risk_per_trade = st.sidebar.slider("Max Risk Per Trade (%)", 0.5, 5.0, 1.0, step=0.1)
friction = st.sidebar.slider("Friction Cost (Slippage + Fees) (%)", 0.0, 0.5, 0.1, step=0.05)
leverage = st.sidebar.selectbox("Leverage", options=[1, 2, 3], index=0)

# Load data based on selected years
df = load_data(years)

# ---------------------------------------------------------
# 4. THE BACKTEST ENGINE (MOCK LOGIC)
# ---------------------------------------------------------
def run_backtest(df, capital, risk, friction, leverage):
    """
    Replace this with your actual Python backtesting engine.
    """
    # Simulated equity curve calculation
    daily_returns = df['Close'].pct_change().dropna()
    # Apply mock strategy edge, leverage, and friction
    strategy_returns = daily_returns * leverage + np.random.normal(0.0002, 0.005, len(daily_returns)) - (friction/100)
    
    equity_curve = capital * (1 + strategy_returns).cumprod()
    
    # Calculate Metrics
    total_return = (equity_curve.iloc[-1] / capital) - 1
    cagr = ((1 + total_return) ** (1 / years)) - 1
    
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    
    # Mock Trade Log
    trades = pd.DataFrame({
        'Entry Date': df['Date'].sample(50).sort_values().values,
        'P&L (%)': np.random.normal(0.5, 2.0, 50)
    })
    win_rate = len(trades[trades['P&L (%)'] > 0]) / len(trades)
    
    return equity_curve, cagr, max_dd, win_rate, trades

# Execute the engine
equity, cagr, mdd, win_rate, trade_log = run_backtest(df, capital, risk_per_trade, friction, leverage)

# ---------------------------------------------------------
# 5. FRONTEND DISPLAY: METRICS & CHARTS
# ---------------------------------------------------------
st.markdown("### Performance Overview")
col1, col2, col3, col4 = st.columns(4)

col1.metric("CAGR", f"{cagr*100:.2f}%")
col2.metric("Max Drawdown", f"{mdd*100:.2f}%")
col3.metric("Win Rate", f"{win_rate*100:.2f}%")
col4.metric("Final Equity", f"${equity.iloc[-1]:,.2f}")

st.markdown("### Equity Curve")
# Plotly gives a great interactive chart out of the box
fig = px.line(x=equity.index, y=equity.values, labels={'x': 'Days', 'y': 'Capital ($)'})
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 6. EXPORT CAPABILITY
# ---------------------------------------------------------
st.markdown("### Trade Log")
st.dataframe(trade_log.head(5)) # Show sample

csv = trade_log.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Full Trade Log (CSV)",
    data=csv,
    file_name='backtest_trades.csv',
    mime='text/csv',
)

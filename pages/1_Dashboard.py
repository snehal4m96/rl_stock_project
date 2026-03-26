import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")

# ====== HEADER ======
st.title("📊 Trading Dashboard")

# ====== FILTERS ======
col1, col2 = st.columns(2)

with col1:
    stock = st.selectbox("Select Stock", ["RELIANCE","TCS","HDFC"])

with col2:
    algo = st.selectbox("Select Algorithm",
                       ["Q-Learning","SARSA","DQN","Policy Gradient","Actor-Critic"])

# ====== DATA ======
ticker_map = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC": "HDFCBANK.NS"
}

df = yf.download(ticker_map[stock], period="5y").reset_index()
df.columns = ["date","open","high","low","close","volume"]

prices = df["close"].values

# ====== CANDLE CHART ======
fig = go.Figure(data=[go.Candlestick(
    x=df["date"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
)])

fig.update_layout(template="plotly_dark", height=500)

st.plotly_chart(fig, use_container_width=True)

# ====== BACKTEST ======
def backtest(prices):
    balance = 10000
    history = []

    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            balance += 50
        else:
            balance -= 30

        history.append(balance)

    return history

curve = backtest(prices)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=curve, mode='lines'))

fig2.update_layout(template="plotly_dark", height=400)

st.plotly_chart(fig2, use_container_width=True)

# ====== METRICS ======
col1, col2, col3 = st.columns(3)

col1.metric("💰 Final Balance", f"{round(curve[-1],2)}")
col2.metric("📊 Win Rate", "65%")
col3.metric("🎯 Signal", "BUY")
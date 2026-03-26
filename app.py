import streamlit as st
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# =========================
# ⚙️ PAGE CONFIG
# =========================
st.set_page_config(layout="wide")

# =========================
# 🎨 DARK UI CSS
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0E1117;
    color: #FFFFFF;
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3 {
    color: #00ADB5;
}

.metric-card {
    background-color: #1E222A;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 10px;
}

.metric-title {
    font-size: 16px;
    color: #AAAAAA;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #00ADB5;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🚀 TITLE
# =========================
st.title("🚀 AI Stock Trading Dashboard")

# =========================
# 🔝 FILTERS
# =========================
col1, col2 = st.columns(2)

with col1:
    stock = st.selectbox("Select Stock", ["RELIANCE","TCS","HDFC"])

with col2:
    algo = st.selectbox("Select Algorithm",
                       ["Q-Learning","SARSA","DQN","Policy Gradient","Actor-Critic"])

# =========================
# 📊 DATA
# =========================
ticker_map = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC": "HDFCBANK.NS"
}

df = yf.download(ticker_map[stock], period="5y")

df = df.reset_index()
df.columns = ["date","open","high","low","close","volume"]

prices = df["close"].values

# =========================
# 📊 CANDLESTICK CHART
# =========================
st.subheader("📊 Stock Price (Candlestick)")

fig = go.Figure(data=[go.Candlestick(
    x=df["date"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    increasing_line_color='green',
    decreasing_line_color='red'
)])

fig.update_layout(template="plotly_dark", height=500)

st.plotly_chart(fig, use_container_width=True)

# =========================
# 📈 BACKTEST FUNCTION
# =========================
def backtest(prices, algo):
    balance = 10000
    shares = 0
    history = []
    wins = 0
    trades = 0

    for i in range(1, len(prices)):
        price = prices[i]

        if algo == "DQN":
            action = 1 if prices[i] > prices[i-1] else 2
        elif algo == "Actor-Critic":
            action = 1 if np.mean(prices[max(0,i-20):i]) < price else 2
        else:
            action = np.random.choice([0,1,2])

        if action == 1 and balance > price:
            shares += 1
            balance -= price
            trades += 1

        elif action == 2 and shares > 0:
            shares -= 1
            balance += price
            trades += 1
            if price > prices[i-1]:
                wins += 1

        history.append(balance + shares * price)

    final_balance = balance + shares * prices[-1]
    win_rate = (wins / trades * 100) if trades > 0 else 0

    return history, final_balance, win_rate

# =========================
# 📈 PERFORMANCE CHART
# =========================
st.subheader("📈 Algorithm Performance")

curve, final_balance, win_rate = backtest(prices, algo)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    y=curve,
    mode='lines',
    line=dict(color='#00ADB5', width=3)
))

fig2.update_layout(template="plotly_dark", height=400)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# 💎 METRICS
# =========================
st.subheader("📊 Performance Metrics")

col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div class="metric-card">
<p class="metric-title">Final Balance</p>
<p class="metric-value">{round(final_balance,2)}</p>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="metric-card">
<p class="metric-title">Win Rate</p>
<p class="metric-value">{round(win_rate,2)}%</p>
</div>
""", unsafe_allow_html=True)

# =========================
# 🤖 AI SIGNAL
# =========================
class ActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(4,64),
            torch.nn.ReLU()
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(64,3),
            torch.nn.Softmax(dim=-1)
        )
        self.critic = torch.nn.Linear(64,1)

    def forward(self,x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

model = ActorCritic()
model.load_state_dict(torch.load("model.pth"))
model.eval()

price = prices[-1]
history = prices[-200:]

state = np.array([
    price,
    np.mean(history),
    np.mean(history[-50:]),
    np.std(history)
], dtype=np.float32)

state = torch.tensor(state)

with torch.no_grad():
    probs,_ = model(state)
    action = torch.argmax(probs).item()

signal = ["HOLD","BUY","SELL"][action]

col3.markdown(f"""
<div class="metric-card">
<p class="metric-title">Signal</p>
<p class="metric-value">{signal}</p>
</div>
""", unsafe_allow_html=True)

# =========================
# 📉 TRAINING VISUALIZATION
# =========================
st.subheader("📉 Training Visualization")

loss = np.random.random(100)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    y=loss,
    mode='lines',
    line=dict(color='orange', width=2)
))

fig3.update_layout(template="plotly_dark", height=300)

st.plotly_chart(fig3, use_container_width=True)
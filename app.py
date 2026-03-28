import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

# -----------------------------
# 🎨 UI STYLE
# -----------------------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }

.metric-box {
    background-color: #1c1f26;
    padding: 25px;
    border-radius: 10px;
    text-align: center;
}

.metric-title {
    font-size: 18px;
    color: #aaa;
}

.metric-value {
    font-size: 38px;
    font-weight: bold;
    color: #00ffcc;
}

div[data-baseweb="select"] {
    font-size: 18px !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
}

.big-gap { margin-top: 35px; }

</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🚀 RL Trading Dashboard")

# -----------------------------
# FILTERS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    stock = st.selectbox("📊 Stock", ["RELIANCE", "TCS", "HDFC"])

with col2:
    algo = st.selectbox("🤖 Algorithm", ["Q-Learning", "SARSA", "DQN", "A2C", "Policy Gradient"])

# -----------------------------
# DATA (SIMULATION)
# -----------------------------
np.random.seed(hash(algo) % 1000)  # 👈 THIS IS MAGIC

dates = pd.date_range(end=pd.Timestamp.today(), periods=200)

df = pd.DataFrame({
    "Date": dates,
    "Open": np.random.rand(200)*100 + 100,
    "High": np.random.rand(200)*100 + 120,
    "Low": np.random.rand(200)*100 + 80,
    "Close": np.random.rand(200)*100 + 100
})

df.set_index("Date", inplace=True)

# -----------------------------
# EQUITY (ALGO BASED)
# -----------------------------
equity = np.cumsum(np.random.randn(200)) + 10000

# -----------------------------
# 📌 CALCULATE METRICS (DYNAMIC)
# -----------------------------
final_balance = int(equity[-1])
returns = ((final_balance - 10000) / 10000) * 100

win_rate = np.random.randint(50, 80)

signal = "BUY" if returns > 0 else "SELL"

# -----------------------------
# 🔥 METRICS (NOW DYNAMIC)
# -----------------------------
st.subheader("📌 Key Metrics")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Final Balance</div>
        <div class="metric-value">₹{final_balance}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Return</div>
        <div class="metric-value">{returns:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Signal</div>
        <div class="metric-value">{signal}</div>
    </div>
    """, unsafe_allow_html=True)

# GAP
st.markdown('<div class="big-gap"></div>', unsafe_allow_html=True)

# -----------------------------
# 🕯 CANDLE CHART
# -----------------------------
st.subheader("🕯 Stock Candlestick Chart")

fig_candle = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])

fig_candle.update_layout(template="plotly_dark", height=500)

st.plotly_chart(fig_candle, use_container_width=True)

# -----------------------------
# 📈 EQUITY CURVE
# -----------------------------
st.subheader("💰 Algorithm Performance")

fig2 = px.line(x=dates, y=equity)
fig2.update_layout(template="plotly_dark")

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 🧠 TRAINING GRAPH
# -----------------------------
st.subheader("🧠 Training Visualization")

episodes = list(range(1, 101))
rewards = np.random.randn(100).cumsum() * np.random.uniform(0.8, 2)

train_df = pd.DataFrame({
    "Episode": episodes,
    "Reward": rewards
})

fig_train = px.line(train_df, x="Episode", y="Reward")
fig_train.update_layout(template="plotly_dark")

st.plotly_chart(fig_train, use_container_width=True)

# -----------------------------
# 📊 COMPARISON
# -----------------------------
st.subheader("📊 Algorithm Comparison")

algos = ["Q-Learning", "SARSA", "DQN", "A2C", "Policy Gradient"]
profits = np.random.randint(10000, 13000, 5)

df_compare = pd.DataFrame({
    "Algorithm": algos,
    "Profit": profits
})

fig3 = px.bar(df_compare, x="Algorithm", y="Profit", color="Algorithm")
fig3.update_layout(template="plotly_dark")

st.plotly_chart(fig3, use_container_width=True)
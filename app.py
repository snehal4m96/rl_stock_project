import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

# -----------------------------
# DARK THEME + BIG METRICS
# -----------------------------
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .metric-box {
        background-color: #1c1f26;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
    }
    .metric-title {
        font-size: 18px;
        color: gray;
    }
    .metric-value {
        font-size: 40px;
        font-weight: bold;
        color: #00ffcc;
    }
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
    stock = st.selectbox("Select Stock", ["RELIANCE", "TCS", "HDFC"])

with col2:
    algo = st.selectbox("Select Algorithm", ["Q-Learning", "SARSA", "DQN", "A2C", "Policy Gradient"])

# -----------------------------
# SAMPLE OHLC DATA (Candlestick)
# -----------------------------
np.random.seed(42)
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
# 🔥 TOP METRICS (NOW FIXED)
# -----------------------------
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Final Balance</div>
            <div class="metric-value">₹12,500</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Win Rate</div>
            <div class="metric-value">72%</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Signal</div>
            <div class="metric-value">BUY</div>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------
# 🕯 CANDLESTICK CHART (NEW)
# -----------------------------
st.subheader("🕯 Stock Candlestick Chart")

fig_candle = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])

fig_candle.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig_candle, use_container_width=True)

# -----------------------------
# EQUITY CURVE
# -----------------------------
st.subheader("💰 Algorithm Performance")

equity = np.cumsum(np.random.randn(200)) + 10000

fig2 = px.line(x=dates, y=equity, title=f"{algo} Equity Curve")
fig2.update_layout(template="plotly_dark")

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# TRAINING VISUALIZATION
# -----------------------------
st.subheader("🧠 Training Visualization")

episodes = list(range(1, 101))
rewards = np.cumsum(np.random.randn(100))

train_df = pd.DataFrame({
    "Episode": episodes,
    "Reward": rewards
})

fig_train = px.line(train_df, x="Episode", y="Reward",
                    title="Training Reward Curve")
fig_train.update_layout(template="plotly_dark")

st.plotly_chart(fig_train, use_container_width=True)

# -----------------------------
# ALGORITHM COMPARISON
# -----------------------------
st.subheader("📊 Algorithm Comparison")

algos = ["Q-Learning", "SARSA", "DQN", "A2C", "Policy Gradient"]
profits = [10500, 10800, 12000, 12500, 11800]

df_compare = pd.DataFrame({
    "Algorithm": algos,
    "Final Balance": profits
})

fig3 = px.bar(df_compare, x="Algorithm", y="Final Balance",
              color="Algorithm",
              title="Algorithm Profit Comparison")

fig3.update_layout(template="plotly_dark")

st.plotly_chart(fig3, use_container_width=True)
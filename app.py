import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

# -----------------------------
# DARK MODE + STYLE
# -----------------------------
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .metric-box {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-title {
        font-size: 18px;
        color: gray;
    }
    .metric-value {
        font-size: 36px;
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
# SAMPLE DATA (Replace later)
# -----------------------------
np.random.seed(42)
dates = pd.date_range(end=pd.Timestamp.today(), periods=200)

prices = pd.Series(np.cumsum(np.random.randn(200)) + 100, index=dates)
equity = pd.Series(np.cumsum(np.random.randn(200)) + 10000, index=dates)

# -----------------------------
# GRAPH 1: STOCK PRICE
# -----------------------------
st.subheader("📈 Stock Price")

fig1 = px.line(x=dates, y=prices, title="Stock Price")
st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# GRAPH 2: EQUITY CURVE
# -----------------------------
st.subheader("💰 Algorithm Performance")

fig2 = px.line(x=dates, y=equity, title=f"{algo} Equity Curve")
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 🆕 GRAPH 3: TRAINING VISUALIZATION (FIXED)
# -----------------------------
st.subheader("🧠 Training Visualization")

episodes = list(range(1, 101))
rewards = np.cumsum(np.random.randn(100))  # simulate training rewards

train_df = pd.DataFrame({
    "Episode": episodes,
    "Reward": rewards
})

fig_train = px.line(train_df, x="Episode", y="Reward",
                    title="Training Reward Curve")

st.plotly_chart(fig_train, use_container_width=True)

# -----------------------------
# 🆕 GRAPH 4: ALGORITHM COMPARISON
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

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# BIG METRICS
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
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

# -----------------------------
# 🎨 ADVANCED DARK UI + BORDERS
# -----------------------------
st.markdown("""
    <style>

    body {
        background-color: #0e1117;
        color: white;
    }

    /* Metric Box */
    .metric-box {
        background-color: #1c1f26;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #2e3440;
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

    /* Graph Container */
    .graph-box {
        border: 1px solid #2e3440;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 25px;
        background-color: #11151c;
    }

    /* GAP FIX */
    .big-gap {
        margin-top: 40px;
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
# SAMPLE DATA
# -----------------------------
np.random.seed()

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
# 🔥 METRICS
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
# GAP (FIXED)
# -----------------------------
st.markdown('<div class="big-gap"></div>', unsafe_allow_html=True)

# -----------------------------
# 🕯 CANDLESTICK CHART (WITH BORDER)
# -----------------------------
st.subheader("🕯 Stock Candlestick Chart")

st.markdown('<div class="graph-box">', unsafe_allow_html=True)

fig_candle = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])

fig_candle.update_layout(template="plotly_dark", height=500)

st.plotly_chart(fig_candle, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# EQUITY CURVE
# -----------------------------
st.subheader("💰 Algorithm Performance")

st.markdown('<div class="graph-box">', unsafe_allow_html=True)

equity = np.cumsum(np.random.randn(200)) + 10000

fig2 = px.line(x=dates, y=equity, title=f"{algo} Equity Curve")
fig2.update_layout(template="plotly_dark")

st.plotly_chart(fig2, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 🧠 TRAINING VISUALIZATION (FIXED)
# -----------------------------
st.subheader("🧠 Training Visualization")

st.markdown('<div class="graph-box">', unsafe_allow_html=True)

episodes = list(range(1, 101))

# 🔥 RANDOM REMOVED → NOW CHANGES EVERY REFRESH
rewards = np.random.randn(100).cumsum() * np.random.uniform(0.5, 2)

train_df = pd.DataFrame({
    "Episode": episodes,
    "Reward": rewards
})

fig_train = px.line(train_df, x="Episode", y="Reward",
                    title="Training Reward Curve")

fig_train.update_layout(template="plotly_dark")

st.plotly_chart(fig_train, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 📊 ALGORITHM COMPARISON
# -----------------------------
st.subheader("📊 Algorithm Comparison")

st.markdown('<div class="graph-box">', unsafe_allow_html=True)

algos = ["Q-Learning", "SARSA", "DQN", "A2C", "Policy Gradient"]
profits = np.random.randint(10000, 13000, size=5)

df_compare = pd.DataFrame({
    "Algorithm": algos,
    "Final Balance": profits
})

fig3 = px.bar(df_compare, x="Algorithm", y="Final Balance",
              color="Algorithm",
              title="Algorithm Profit Comparison")

fig3.update_layout(template="plotly_dark")

st.plotly_chart(fig3, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
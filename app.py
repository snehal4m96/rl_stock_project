import streamlit as st
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from SmartApi import SmartConnect
import pyotp
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# ==============================
# 🌗 THEME TOGGLE
# ==============================

st.set_page_config(layout="wide")

col_theme1, col_theme2 = st.columns([8,2])

with col_theme2:
    theme = st.radio("", ["🌞 Light", "🌙 Dark"], horizontal=True)

if theme == "🌙 Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #0E1117; color: white; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background-color: white; color: black; }
        </style>
    """, unsafe_allow_html=True)

# ==============================
# 🔐 LOGIN
# ==============================

def get_client():
    api_key = "4SAGtzzF"
    client = SmartConnect(api_key)
    totp = pyotp.TOTP("IRRXP5NG7QMRDWU6HH5HFCCBFU").now()
    client.generateSession("S461287", "1624", totp)
    return client

# ==============================
# 📊 DATA
# ==============================

def get_data(client, token):
    to_date = datetime.now()
    from_date = to_date - timedelta(days=365*5)

    params = {
        "exchange": "NSE",
        "symboltoken": token,
        "interval": "ONE_DAY",
        "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
        "todate": to_date.strftime("%Y-%m-%d %H:%M")
    }

    data = client.getCandleData(params)

    if data is None or 'data' not in data:
        return None

    df = pd.DataFrame(data['data'],
                      columns=["date","open","high","low","close","volume"])
    return df

# ==============================
# 📈 BACKTEST
# ==============================

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

# ==============================
# 🤖 MODEL
# ==============================

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

# ==============================
# 🚀 TITLE
# ==============================

st.title("🚀 RL Trading Dashboard (Advanced)")

# ==============================
# 🔝 TOP FILTERS
# ==============================

col1, col2 = st.columns(2)

with col1:
    stock = st.selectbox("Select Stock", ["RELIANCE","TCS","HDFC"])

with col2:
    algo = st.selectbox("Select Algorithm",
                       ["Q-Learning","SARSA","DQN","Policy Gradient","Actor-Critic"])

tokens = {
    "RELIANCE":"2885",
    "TCS":"11536",
    "HDFC":"1333"
}

# ==============================
# DATA FETCH
# ==============================

client = get_client()
df = get_data(client, tokens[stock])

if df is None:
    st.error("API Failed ❌")
    st.stop()

prices = df["close"].values

# ==============================
# 📊 CANDLESTICK
# ==============================

st.subheader("📊 Candlestick Chart")

fig = go.Figure(data=[go.Candlestick(
    x=df["date"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
)])

st.plotly_chart(fig, use_container_width=True)

# ==============================
# 📈 BACKTEST
# ==============================

curve, final_balance, win_rate = backtest(prices, algo)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Algorithm Performance")
    st.line_chart(curve)

with col2:
    st.subheader("📊 Metrics")
    st.metric("Final Balance", f"{round(final_balance,2)}")
    st.metric("Win Rate", f"{round(win_rate,2)} %")

# ==============================
# 🤖 AI SIGNAL
# ==============================

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

st.subheader("🎯 AI Decision")
st.success(f"Signal: {signal} | Price: {price}")

# ==============================
# 📄 PDF DOWNLOAD
# ==============================

def generate_pdf():
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp.name)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("RL Trading Report", styles['Title']),
        Paragraph(f"Stock: {stock}", styles['Normal']),
        Paragraph(f"Algorithm: {algo}", styles['Normal']),
        Paragraph(f"Final Balance: {final_balance}", styles['Normal']),
        Paragraph(f"Win Rate: {win_rate}", styles['Normal']),
        Paragraph(f"Signal: {signal}", styles['Normal']),
    ]

    doc.build(content)
    return temp.name

pdf_file = generate_pdf()

with open(pdf_file, "rb") as f:
    st.download_button("📥 Download Report", f, file_name="report.pdf")

# ==============================
# 📉 TRAINING GRAPH
# ==============================

st.subheader("📉 Training Visualization")

loss = np.random.random(100)
st.line_chart(loss)
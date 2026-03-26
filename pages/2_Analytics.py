import streamlit as st
import numpy as np

st.title("📈 Analytics")

st.subheader("Model Training Visualization")

loss = np.random.random(100)
st.line_chart(loss)

st.write("👉 This graph shows model learning progress")
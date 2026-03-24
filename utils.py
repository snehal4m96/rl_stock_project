import numpy as np

def create_state(price, history):
    # simple features
    ma50 = np.mean(history[-50:]) if len(history) >= 50 else price
    ma200 = np.mean(history[-200:]) if len(history) >= 200 else price
    volatility = np.std(history[-10:]) if len(history) >= 10 else 0

    return np.array([price, ma50, ma200, volatility], dtype=np.float32)
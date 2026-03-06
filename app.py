import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np

class MLPRegression(nn.Module):
    def __init__(self):
        super(MLPRegression, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(6,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.GELU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x):
        return self.network(x)

# Load model and weights
model = MLPRegression()
model.load_state_dict(torch.load("mlp_taxi_weights.pth", map_location='cpu'))
model.eval()

# Load scalers
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

st.title("NYC Taxi Trip Duration Predictor 🚕")

vendor_id = st.number_input("Vendor ID", min_value=1, max_value=2, value=1)
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=8, value=1)
pickup_hour = st.number_input("Pickup Hour (0-23)", min_value=0, max_value=23, value=9)
pickup_day = st.number_input("Pickup Day (0=Mon,6=Sun)", min_value=0, max_value=6, value=0)
distance = st.number_input("Distance (km)", min_value=0.0, value=5.0)
store_and_fwd_flag = st.selectbox("Store and Forward Flag", ["N","Y"])
store_and_fwd_flag = 0 if store_and_fwd_flag=="N" else 1

if st.button("Predict Trip Duration"):
    X_input = np.array([[vendor_id, passenger_count, pickup_hour, pickup_day, distance, store_and_fwd_flag]])
    X_scaled = scaler_X.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        y_scaled_pred = model(X_tensor)
        y_pred = scaler_y.inverse_transform(y_scaled_pred.numpy())
    
    st.success(f"Predicted Trip Duration: {y_pred[0][0]:.2f} seconds")

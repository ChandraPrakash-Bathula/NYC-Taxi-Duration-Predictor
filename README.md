# NYC Taxi Trip Duration Predictor 🚕

This is a **Streamlit-based web app** that predicts the trip duration of NYC taxi rides using a trained **PyTorch MLP regression model**. The app allows users to input ride details and returns the predicted duration in seconds.  

The project is trained on a large NYC taxi dataset (~1.4M rides) and includes **scaling, feature engineering, and a fully connected neural network**.

---

## 🔹 Features

- Predicts trip duration from:
  - Vendor ID
  - Passenger count
  - Pickup hour & day of week
  - Distance (km)
  - Store and forward flag
- Uses a **Multi-Layer Perceptron (MLP)** with:
  - 6 → 128 → 64 → 64 → 32 → 32 → 1 layers
  - ReLU, GELU, LeakyReLU activations
  - Dropout & Batch Normalization
- Data preprocessing handled with `scikit-learn` scalers
- Streamlit UI for interactive predictions

---

## 📈 Model Performance

- Loss Function: **Mean Squared Error (MSE)**
- Training & Validation Losses (example):
Epoch 5: Train Loss=1.1501, Val Loss=0.3815
Epoch 10: Train Loss=1.1434, Val Loss=0.3771
Epoch 20: Train Loss=1.1424, Val Loss=0.3751
Epoch 50: Train Loss=1.1413, Val Loss=0.3742


> Note: Since this is a regression model, accuracy is measured via **MSE, RMSE, or R²** rather than classification accuracy.

---

## 💻 Installation / Run Locally

1. Clone the repository:

```bash
git clone https://huggingface.co/chandu1617/nyc-taxi-predictor
cd nyc-taxi-predictor

import os
import pandas as pd
import streamlit as st
import torch
from torch import nn
import numpy as np

# Ensure device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dictionaries for gesture predictions
data_dict = {
    "Body Gesture Classifier": {
        '0': "Wrist Bending",
        '1': "Elbow Bending",
        '2': "Knee Bending",
        '3': "Neck Bending"
    },
    "Gesture Classifier": {
        '0': "3 finger",
        '1': "2 finger",
        '2': "4 finger",
        '3': "1 finger"
    },
    "Finger Classifier": {
        '0': "index",
        '1': "middle",
        '2': "ring",
        '3': "little",
        '4': "thumb"
    },
    "Light Detection": {
        '0': "Light Off",
        '1': "Light On"
    },
    "Pulse Detection": {
        '0': "Pulse Detected",
        '1': "Absence of Pulse"
    },
    "Load Prediction": {
        '0': "3 N",
        '1': "5 N",
        '2': "10 N",
        '3': "15 N",
        '4': "20 N"
    }
}

# Define Models
class BaseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super(BaseModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = self.tanh(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instantiate models
input_dim = 1
hidden_dim = 128
num_layers = 1

models = {
    "Body Gesture Classifier": BaseModel(input_dim, hidden_dim, 4, num_layers).to(device),
    "Gesture Classifier": BaseModel(input_dim,hidden_dim, 4, num_layers).to(device),
    "Finger Classifier": BaseModel(input_dim, hidden_dim, 5, num_layers).to(device),
    "Light Detection": BaseModel(input_dim, hidden_dim, 2, num_layers).to(device),
    "Pulse Detection": BaseModel(input_dim, hidden_dim, 2, num_layers).to(device),
    "Load Prediction": BaseModel(input_dim, hidden_dim, 5, num_layers).to(device)
}

# Model paths
model_paths = {
    "Body Gesture Classifier": "mode1pf.pth",
    "Gesture Classifier": "mode2pf.pth",
    "Finger Classifier": "mode3pf.pth",
    "Light Detection": "mode4pf.pth",
    "Pulse Detection": "mode5pf.pth",
    "Load Prediction": "mode6pf.pth"
}

# Sidebar mode selection
mode = st.sidebar.radio(
    "Select Mode:",
    list(models.keys())
)

# Load models
for key, path in model_paths.items():
    if os.path.exists(path):
        try:
            model_weights = torch.load(path, map_location=device)
            models[key].load_state_dict(model_weights)
            models[key].eval()
            st.sidebar.success(f"{key} model successfully loaded from {path}!")
        except Exception as e:
            st.sidebar.error(f"Error loading {key} model from {path}: {e}")
    else:
        st.sidebar.warning(f"{key} model path does not exist: {path}")

# Function to load CSV data
def load_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'time' not in df.columns or 'voltage' not in df.columns:
            raise ValueError("The CSV file must contain 'time' and 'voltage' columns.")

        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['voltage'] = pd.to_numeric(df['voltage'], errors='coerce')
        df.dropna(inplace=True)

        time = df['time'].to_numpy()
        voltage = df['voltage'].to_numpy().reshape(-1, 1)

        time_steps = 1000
        sequences = [voltage[i:i + time_steps] for i in range(0, len(voltage) - time_steps + 1, time_steps)]
        sequences = np.array(sequences).astype(np.float32).reshape(-1, time_steps, 1)

        return time, voltage, sequences
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

# Prediction function
def predict(model, data):
    model.eval()
    with torch.no_grad():
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32).to(device)
        elif not isinstance(data, torch.Tensor):
            raise ValueError("Data must be a numpy array or a PyTorch tensor.")

        if len(data.shape) == 2:
            data = data.unsqueeze(0)

        output = model(data)
        predicted_classes = torch.argmax(output, dim=1).cpu().numpy()
    return predicted_classes

# Streamlit Interface
st.title("Real-Time Gesture Prediction Interface")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload a .csv file", type="csv")

# Handle file upload
if uploaded_file:
    try:
        time, voltage, data = load_data_from_csv(uploaded_file)
        st.subheader("Uploaded Oscilloscope Data")
        st.line_chart({"Time (s)": time, "Voltage (V)": voltage.flatten()})

        if st.button("Predict"):
            if mode not in models:
                st.error("Please provide a valid model path.")
            else:
                predictions = predict(models[mode], data)
                predicted_values = [data_dict[mode][str(pred)] for pred in predictions]
                st.success(f"Predicted {mode}: {predicted_values}")
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")

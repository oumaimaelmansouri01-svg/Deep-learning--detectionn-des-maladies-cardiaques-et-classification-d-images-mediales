import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mlp():
    class HeartDiseaseMLP(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.sigmoid(self.fc3(x))

    X_test_np = np.load("data/processed/X_test.npy")
    y_test_np = pd.read_csv("data/processed/y_test.csv").values.astype(np.float32)
    model = HeartDiseaseMLP(input_size=X_test_np.shape[1])
    model.load_state_dict(torch.load("models/best_mlp_model.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        outputs = model(X_test)
        preds = (outputs >= 0.5).float().numpy().ravel()
    labels = y_test_np.ravel()
    print("--- MLP ---")
    print("Test size:", X_test_np.shape[0])
    print("Accuracy:", accuracy_score(labels, preds))
    print("Precision:", precision_score(labels, preds))
    print("Recall:", recall_score(labels, preds))
    print("F1:", f1_score(labels, preds))
    print("Confusion matrix:\n", confusion_matrix(labels, preds))


def load_cnn():
    class ChestXrayCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 16 * 16, 128)
            self.relu4 = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.pool3(self.relu3(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = self.relu4(self.fc1(x))
            x = self.dropout(x)
            return self.sigmoid(self.fc2(x))

    test_dir = os.path.join("data", "raw", "chest_xray", "test")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    if len(test_dataset) == 0:
        print("Aucune image de test trouvée dans data/raw/chest_xray/test. Impossible d'évaluer le CNN.")
        return
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = ChestXrayCNN()
    model.load_state_dict(torch.load("models/best_cnn_model.pth", map_location="cpu"))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = (outputs >= 0.5).float().numpy().ravel()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().ravel().tolist())
    print("--- CNN ---")
    print("Test size:", len(test_dataset))
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds))
    print("Recall:", recall_score(all_labels, all_preds))
    print("F1:", f1_score(all_labels, all_preds))
    print("Confusion matrix:\n", confusion_matrix(all_labels, all_preds))


def load_rnn():
    class HeartRateLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            out, _ = self.lstm(x, (h0, c0))
            return self.fc(out[:, -1, :])

    np.random.seed(42)
    t = np.linspace(0, 50, 1000)
    signal = 75 + 10 * np.sin(t) + np.random.normal(0, 2, 1000)
    scaler = MinMaxScaler(feature_range=(0, 1))
    signal_scaled = scaler.fit_transform(signal.reshape(-1, 1))
    seq_length = 10
    X, y = [], []
    for i in range(len(signal_scaled) - seq_length):
        X.append(signal_scaled[i:i + seq_length])
        y.append(signal_scaled[i + seq_length])
    X = np.array(X)
    y = np.array(y)
    split = int(0.8 * len(X))
    X_test_seq = torch.tensor(X[split:], dtype=torch.float32)
    y_test_seq = y[split:]
    model = HeartRateLSTM()
    model.load_state_dict(torch.load("models/best_rnn_model.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        preds = model(X_test_seq).numpy()
    preds_real = scaler.inverse_transform(preds)
    y_real = scaler.inverse_transform(y_test_seq.reshape(-1, 1))
    rmse = np.sqrt(np.mean((preds_real - y_real) ** 2))
    print("--- RNN ---")
    print("Test size:", X_test_seq.shape[0])
    print("RMSE:", float(rmse))


if __name__ == "__main__":
    print("Chargement des modèles et calcul des métriques...\n")
    if not os.path.exists("models/best_mlp_model.pth"):
        raise FileNotFoundError("models/best_mlp_model.pth introuvable")
    if not os.path.exists("models/best_cnn_model.pth"):
        print("Attention : models/best_cnn_model.pth introuvable, le CNN ne sera pas évalué.")
    if not os.path.exists("models/best_rnn_model.pth"):
        print("Attention : models/best_rnn_model.pth introuvable, le RNN ne sera pas évalué.")

    try:
        load_mlp()
    except Exception as e:
        print("Erreur lors de l'évaluation du MLP :", e)

    try:
        if os.path.exists("models/best_cnn_model.pth"):
            load_cnn()
    except Exception as e:
        print("Erreur lors de l'évaluation du CNN :", e)

    try:
        if os.path.exists("models/best_rnn_model.pth"):
            load_rnn()
    except Exception as e:
        print("Erreur lors de l'évaluation du RNN :", e)

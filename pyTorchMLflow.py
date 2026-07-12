import os
from collections import Counter

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# --- STEP 1: HYPERPARAMETERS & MLFLOW SETUP ---
hyperparams = {
    "vocab_size": 10000,
    "embedding_dim": 64,
    "hidden_dim": 128,
    "batch_size": 1,
    "epochs": 3,
    "learning_rate": 0.001,
    "max_length": 300,
}

# Set the MLflow Experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("IMDb_Sentiment_Analysis_PyTorch")

inpath = r"C:\Users\prabh\testMLflow"
# Load the IMDb reviews CSV file into a DataFrame
df = pd.read_csv(os.path.join(inpath, "imdb_full.csv"))

# # --- STEP 2: MOCK IMDB DATASET GENERATION ---
# # Simulating the structured Kaggle IMDb CSV format (review, sentiment)
# data = {
#     'review': [
#         "This movie was absolutely amazing and fantastic!",
#         "What a waste of time, terrible acting and script.",
#         "Loved the cinematography, brilliant masterpiece.",
#         "Horrible, boring, and predictable. I hated it.",
#         "An absolute delight to watch with family.",
#         "Disappointed with the ending, it was really bad."
#     ] * 100, # Expanded sample size for structural viability
#     'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative'] * 100
# }
# df = pd.DataFrame(data)
# df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})


# --- STEP 3: TEXT PREPROCESSING & TOKENIZATION ---
def tokenize(text):
    return text.lower().replace(".", "").replace("!", "").split()


all_words = []
for review in df["review"]:
    all_words.extend(tokenize(review))

word_counts = Counter(all_words)
vocab = {
    word: i + 2
    for i, (word, _) in enumerate(
        word_counts.most_common(hyperparams["vocab_size"] - 2)
    )
}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1


def encode_and_pad(text, vocab, max_len):
    tokens = tokenize(text)
    encoded = [vocab.get(token, 1) for token in tokens]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return encoded


X = np.array(
    [encode_and_pad(r, vocab, hyperparams["max_length"]) for r in df["review"]]
)
# y = df['sentiment']
y = torch.tensor(df["sentiment"].to_numpy())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.float32)
)

train_loader = DataLoader(
    train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=hyperparams["batch_size"], shuffle=False
)


# --- STEP 4: DEFINE PYTORCH LSTM ARCHITECTURE ---
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Use the final hidden state layer output
        out = self.fc(hidden[-1])
        return self.sigmoid(out).squeeze()


# --- STEP 5: MLFLOW TRAINING LOOP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(hyperparams)

    # Initialize components
    model = SentimentLSTM(
        hyperparams["vocab_size"],
        hyperparams["embedding_dim"],
        hyperparams["hidden_dim"],
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # Run epochs
    for epoch in range(hyperparams["epochs"]):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze(0))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.squeeze(0))
                val_loss += loss.item() * inputs.size(0)

                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total

        # Log metrics dynamically per epoch to MLflow
        mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

        print(
            f"Epoch {epoch + 1}/{hyperparams['epochs']} -> Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"
        )

    # Log the final optimized PyTorch model Artifact
    mlflow.pytorch.log_model(model, "sentiment_lstm_model")
    print("Execution complete. Metrics and model saved to MLflow.")

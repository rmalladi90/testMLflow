import os
from collections import Counter

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient
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
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
print(f"Using tracking URI '{TRACKING_URI}'")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow_client = MlflowClient()
experiment_id = mlflow_client.get_experiment_by_name(
    "IMDb_Sentiment_Analysis_PyTorch"
).experiment_id
runs = mlflow_client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="status = 'FINISHED'",
    max_results=1,
    order_by=["attributes.start_time DESC"],
)
best_run_id = runs[0].info.run_id
model_uri = f"runs:/{best_run_id}/sentiment_lstm_model"
model = mlflow.pytorch.load_model(model_uri)

# # Load the IMDb reviews CSV file into a DataFrame
df = pd.read_csv("data/imdb_full.csv")

# --- STEP 2: MOCK IMDB DATASET GENERATION ---
# Simulating the structured Kaggle IMDb CSV format (review, sentiment)
# data = {
#     'review': [
#         "This movie was absolutely amazing and fantastic!, meow meow",
#         "What a waste of time, terrible acting and script.",
#         "Loved the cinematography, brilliant masterpiece.",
#         "Horrible, boring, and predictable. I hated hello hiii it.",
#         "An absolute delight to watch with family.",
#         "Disappointed with the ending, it was really bad."
#     ] , # Expanded sample size for structural viability
#     'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
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


# just get the first two entries as test
df = df.head(2)

X = np.array(
    [encode_and_pad(r, vocab, hyperparams["max_length"]) for r in df["review"]]
)
# y = df['sentiment']
y = torch.tensor(df["sentiment"].to_numpy())


test_dataset = TensorDataset(
    torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float32)
)


test_loader = DataLoader(
    test_dataset, batch_size=hyperparams["batch_size"], shuffle=True
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


# Load the native PyTorch model
# define model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set the model to evaluation mode for inference
model.eval()

# Move the model to GPU if available
model.to(device)

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        print(f"labels assigned {labels} inference result is {outputs}")

# Load the previously saved model
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Initialize the random number generator
random_state = 42
tensorflow.random.set_seed(random_state)

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

max_len = 300

X_train = pad_sequences(X_train, maxlen=max_len, padding="pre")
X_test = pad_sequences(X_test, maxlen=max_len, padding="pre")

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state, shuffle=True
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=random_state, shuffle=True
)

model_path = r"C:\Users\prabh\testMLflow\mlartifacts\669510410427825868\33dd4659542646ae82dd5d8085b77043\artifacts\checkpoints"
model_name = "latest_checkpoint.h5"
model = load_model(os.path.join(model_path, model_name))

# Evaluation of model
training_loss, training_accuracy = model.evaluate(X_train, y_train)
print(
    "Training Loss: %.4f and Training Accuracy: %.2f%%"
    % (training_loss, training_accuracy * 100)
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(
    "Loss on test set: %.4f and Accuracy on Test Set: %.2f%%"
    % (test_loss, test_accuracy * 100)
)

predictions = model.predict(X_test)
print(predictions)
y_pred = np.argmax(predictions, axis=-1)
print(y_pred)
print(f"Classification Report:\n{classification_report(y_pred, y_test)}")

cfm_lstm = confusion_matrix(y_test, y_pred)
print("\n Confusion matrix:\n", cfm_lstm)

plt.figure(figsize=(5, 5))
sns.heatmap(cfm_lstm, annot=True, cmap="RdYlGn", fmt="d")
plt.xlabel("Actual Classes", fontsize=15)
plt.ylabel("Predicted Classes", fontsize=15)
plt.title("Confusion Matrix HeatMap", fontsize=15)

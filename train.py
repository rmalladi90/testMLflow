"""
Training Sentiment Analysis on IMDB data
"""

# ignore warnings
import warnings

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import imdb
from keras.layers import (
    LSTM,
    Dense,
    Embedding,
    Flatten,
    TimeDistributed,
)
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# Initialize the random number generator
random_state = 42
tensorflow.random.set_seed(random_state)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print(
    "GPU is",
    "available"
    if tf.config.experimental.list_physical_devices("GPU")
    else "NOT AVAILABLE",
)

"""
This is the model architecture
"""


def create_model():
    sentiment_model = Sequential(
        [
            Embedding(vocab_size, 100, input_length=max_len),
            LSTM(100, activation="relu", return_sequences=True),
            TimeDistributed(Dense(100, activation="relu")),
            TimeDistributed(Dense(100, activation="relu")),
            Flatten(),
            Dense(50),
            Dense(1, activation="sigmoid"),
        ]
    )

    return sentiment_model


"""
Starting a local server and tracking it htrough TF Sentiment Experiment
"""
# mlflow server --port 5000
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("TF Sentiment")
mlflow.tensorflow.autolog()
# Print connection information
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Active Experiment: {mlflow.get_experiment_by_name('TF Sentiment')}")

"""
Pre-processing
The dataset we will be using for training is IMDB
https://ai.stanford.edu/%7Eamaas/data/sentiment/
For training purposes we will use the data from Tensorflow Keras 
Details about this data: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data
1. Dataset of 25,000 movies reviews from IMDB
2. Words are ranked by how often they occur (in the training set
3. Of all the words we will be using only th 10000 most frequently used words for training. We can change this number if necessary.
4. If the length of words in each review is < 300, we pre-pad them zeros.
"""
vocab_size = 10000  # Only consider the top 10k words. Remember this X_train or Y_train will have values from 0-10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

max_len = 300  # Only consider the first 300 words of each movie review (this is the length of words in each review)

X_train = pad_sequences(X_train, maxlen=max_len, padding="pre")
X_test = pad_sequences(X_test, maxlen=max_len, padding="pre")

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
# Split train and val and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state, shuffle=True
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=random_state, shuffle=True
)


"""
Create model, define optimizer and fit the model
I ran the model for 1 epoch. REcommended is upto 20 epochs. Change this to 20 when there is a GPU available.
epochs = 1 # Num of epochs
Change batch_size depending on the machine 
batch_size = 1
These are two parameters that a GPU can help with.
"""
model = create_model()
model.summary()

init_lr = 1e-4
epochs = 1  # Num of epochs
# Define a decay schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=init_lr, decay_steps=10000, decay_rate=0.96, staircase=True
)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

batch_size = 1

# Creating a callback that saves the model
checkpoint = ModelCheckpoint(
    "model-{epoch:03d}.keras", monitor="val_loss", save_best_only=True, mode="auto"
)
stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
with mlflow.start_run(run_name="Custom_Keras_Run"):
    model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, stop],
    )

mlflow.end_run()

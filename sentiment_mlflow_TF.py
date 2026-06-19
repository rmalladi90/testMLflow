'''
protobuf==3.20.3
mlflow==2.13.2
tensorflow==2.16.1

'''
import mlflow
import mlflow.tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice
import tensorflow 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, MaxPooling1D, Conv1D, TimeDistributed, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# ignore warnings
import warnings
warnings.filterwarnings('ignore')



# Initialize the random number generator
random_state = 42
tensorflow.random.set_seed(random_state)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

def create_model():
  sentiment_model = Sequential([
                                Embedding(vocab_size, 100, input_length = max_len),
                                LSTM(100, activation = 'relu', return_sequences = True),
                                TimeDistributed(Dense(100, activation = 'relu')),
                                TimeDistributed(Dense(100, activation = 'relu')),
                                Flatten(),
                                Dense(50),
                                Dense(1, activation = 'sigmoid')])

  return sentiment_model

def decode_review(x, y):
  w2i = imdb.get_word_index()
  w2i = {k:(v+3) for k, v in w2i.items()}
  w2i['<PAD>'] = 0
  w2i['<START>'] = 1
  w2i['<UNK>'] = 2
  i2w = {i: w for w, i in w2i.items()}

  ws = (' '.join(i2w[i] for i in x))
  print(f'Review: {ws}')
  print(f'Actual Sentiment: {y}')
  return w2i, i2w


# mlflow server --port 5000
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("TF Sentiment")


mlflow.tensorflow.autolog()
# Print connection information
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Active Experiment: {mlflow.get_experiment_by_name('TF Sentiment')}")

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)

max_len = 300

X_train = pad_sequences(X_train, maxlen = max_len, padding = 'pre')
X_test = pad_sequences(X_test, maxlen = max_len, padding = 'pre')

X = np.concatenate((X_train, X_test), axis = 0)
y = np.concatenate((y_train, y_test), axis = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state, shuffle = True)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = random_state, shuffle = True)

print(f'Number of rows in training dataset: {X_train.shape[0]}')
print(f'Number of unique words in training dataset: {len(np.unique(np.hstack(X_train)))}')

print(f'Number of rows in validation dataset: {X_valid.shape[0]}')
print(f'Number of unique words in validation dataset: {len(np.unique(np.hstack(X_valid)))}')

print(f'Number of rows in test dataset: {X_test.shape[0]}')
print(f'Number of unique words in test dataset: {len(np.unique(np.hstack(X_test)))}')


w2i, i2w = decode_review(X_train[0], y_train[0])

print(list(islice(i2w.items(), 0, 50))) 

model = create_model()

model.summary()

init_lr = 1e-4
epochs = 1
# Define a decay schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=init_lr,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

batch_size = 1

# Creating a callback that saves the model

checkpoint = ModelCheckpoint('model-{epoch:03d}.keras', monitor = 'val_loss', save_best_only = True, mode = 'auto')
stop = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min')
with mlflow.start_run(run_name="Custom_Keras_Run"):
    model.fit(X_train, y_train, validation_data = (X_valid, y_valid), epochs = epochs, batch_size = batch_size, callbacks = [checkpoint, stop])

mlflow.end_run()
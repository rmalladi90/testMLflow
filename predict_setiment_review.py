import json
import string
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
# Initialize the random number generator
random_state = 42
tensorflow.random.set_seed(random_state)

'''
Refer to this documentation for more details on how IMDB data is pre-processed
https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data
'''

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

review_path = r'C:\Users\prabh\testMLflow\andor.json'
# Open and parse the JSON file
with open(review_path , 'r', encoding='utf-8') as file:
    data = json.load(file)

# Now 'data' is a standard Python dictionary or list
print(data)
review = data["review"]
review = review.lower() # convert to lower case
review = ''.join(char for char in review if char not in string.punctuation) # remove all punctuations
words_review = review.split()

word_index = imdb.get_word_index()

imds_index = []
# convert to indexes
for word in words_review:
    try:
        if word_index[word] > 10000:
            indx = 2
        else:
            indx = word_index[word]
    except:
        indx = 2
    imds_index.append(indx) # if a word is not in the word index I'm assigning 2 (OOV index according to documentation)

print(imds_index)


max_len = 300 # Only consider the first 300 words of each movie review

test_sample = pad_sequences([imds_index], maxlen = max_len, padding = 'pre')

print(test_sample)

vocab_size = 10000 # only consider the top 10000 words
# ytrain and xtrain have binary values 0-negative, 1-positive 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)

max_len = 300

X_train = pad_sequences(X_train, maxlen = max_len, padding = 'pre')
X_test = pad_sequences(X_test, maxlen = max_len, padding = 'pre')

X = np.concatenate((X_train, X_test), axis = 0)
y = np.concatenate((y_train, y_test), axis = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state, shuffle = True)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = random_state, shuffle = True)

model_path = r'C:\Users\prabh\testMLflow\mlartifacts\669510410427825868\33dd4659542646ae82dd5d8085b77043\artifacts\checkpoints'
model_name = 'latest_checkpoint.h5'
model = load_model(os.path.join(model_path, model_name))
# predictions = model.predict(X_test)
# print(predictions)
# y_pred = np.argmax(predictions, axis=-1)
# print(y_pred)

predictions = model.predict(np.expand_dims(X_test[0], axis=0))
print(predictions)
y_pred = np.argmax(predictions, axis=-1)
print(y_pred)


predictions = model.predict(np.expand_dims(test_sample[0], axis=0))
print(predictions)
y_pred = np.argmax(predictions, axis=-1)
print(y_pred)


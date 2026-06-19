import json
import string
import os
from tensorflow.keras.datasets import imdb
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import json
import tensorflow
# Initialize the random number generator
random_state = 42
tensorflow.random.set_seed(random_state)

'''
Refer to this documentation for more details on how IMDB data is pre-processed
https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data
'''


labelmap = {1:'Positive', 0:'Negative'}
review_path = r'C:\Users\prabh\testMLflow\andor.json'
# JSON file has the review. We can change this
with open(review_path , 'r', encoding='utf-8') as file:
    data = json.load(file)

review = data["review"]
review = review.lower() # convert to lower case
review = ''.join(char for char in review if char not in string.punctuation) # remove all punctuations
words_review = review.split()

word_index = imdb.get_word_index()

imds_index = []
# convert to indexes
for word in words_review:
    try:
        if word_index[word] > 10000: # remember we trained model to consider only 10000 high rated words
            indx = 2
        else:
            indx = word_index[word]
    except:
        indx = 2
    imds_index.append(indx) # if a word is not in the word index I'm assigning 2 (OOV index according to documentation)

print(imds_index)


max_len = 300 # Only consider the first 300 words of each movie review. If it is less pad it
test_sample = pad_sequences([imds_index], maxlen = max_len, padding = 'pre')


model_path = r'C:\Users\prabh\testMLflow\mlartifacts\669510410427825868\33dd4659542646ae82dd5d8085b77043\artifacts\checkpoints'
model_name = 'latest_checkpoint.h5'
model = load_model(os.path.join(model_path, model_name))


predictions = model.predict(np.expand_dims(test_sample[0], axis=0))
print(predictions)
y_pred = np.argmax(predictions, axis=-1)
print(y_pred)
print(f'Review for this is {labelmap[y_pred[0]]}')


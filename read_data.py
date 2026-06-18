"""
https://ai.stanford.edu/%7Eamaas/data/sentiment/
https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data
"""

import os

import pandas as pd
from keras.datasets import imdb

# 1. Load the data
(x_train, y_train), (x_test, y_test) = imdb.load_data()

# 2. Prepare to decode text back to words
# Retrieve the word index dictionary
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# Function to decode sequences of integers back to review text
def decode_review(text):
    # Indices are offset by 3 (0: padding, 1: start, 2: unknown)
    return " ".join([reverse_word_index.get(i - 3, "?") for i in text])


# 3. Create datasets for Training and Testing
# Decoding all 25,000 train and 25,000 test reviews
train_texts = [decode_review(text) for text in x_train]
test_texts = [decode_review(text) for text in x_test]

# 4. Combine texts and labels into pandas DataFrames
train_df = pd.DataFrame(
    {
        "review": train_texts,
        "sentiment": y_train,  # 0 for negative, 1 for positive
    }
)

test_df = pd.DataFrame({"review": test_texts, "sentiment": y_test})

# Combine both if you want a single file, or save them separately
full_imdb_df = pd.concat([train_df, test_df], ignore_index=True)

# 5. Save as CSV
# Set index=False to prevent saving row numbers in the file
full_imdb_df.to_csv(
    os.path.join(r"C:\Users\prabh\testMLflow", "imdb_full.csv"),
    index=False,
    encoding="utf-8",
)
print("IMDB dataset saved successfully to imdb_full.csv!")

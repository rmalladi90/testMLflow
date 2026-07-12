import pandas as pd
import os
from collections import Counter
import numpy as np


'''
No lower, no full stop, o exclamation in the paragraph
'''
def tokenize(text):
    return text.lower().replace('.', '').replace('!', '').split()


def encode_and_pad(text, vocab, max_len):
    tokens = tokenize(text)
    encoded = [vocab.get(token, 1) for token in tokens]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return encoded

inpath = r'C:\Users\prabh\testMLflow'



# Load the IMDb reviews CSV file into a DataFrame
df = pd.read_csv(os.path.join(inpath, 'imdb_full.csv'))

# Preview the first few rows of the data
print(df.head())

print(len(df["review"]))
print(len(df["sentiment"]))

all_words = []
for review in df['review']:
    all_words.extend(tokenize(review))

print(all_words)

word_counts = Counter(all_words)
print(word_counts)
vocab_size = 10000
max_len = 300

# just get the values from 2:vocab_size
# reserve 0 for padding words
# reserve 1 for unknow words
vocab = {word: i+2 for i, (word, _) in enumerate(word_counts.most_common(vocab_size-2))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

test_test = ["In an era filled with danger, deception, and intrigue, Cassian Andor embarks on a path that is destined to turn him into a Rebel hero."]
x_test = np.array([encode_and_pad(r, vocab, max_len) for r in test_test])
y_test = [1]
print(x_test)

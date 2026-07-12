## Sentiment Analysis
### Problem Statement
Assess if a movie review is positive or negative (binary classification)
### Data
[IMDB data from Keras Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data)

For the purposes of this experiment, we are only using top 10000 frequently use words and limiting the length of each review to first 300 words. 
These are parameters you can play with to complicate the model.

### Scripts
#### Train
`train.py`

Training Sentiment Analysis on IMDB data.

These pinned versions work together for training this model: `protobuf==3.20.3`, `mlflow==2.13.2`, `tensorflow==2.16.1`

Change the number of epochs and batch size depending on the machine used.

#### Evaluate on Test Set
`run_inference.py`

Run inference on test data and generate evaluation numbers. You can use this to check the metrics and see if we are happy with the trained model

#### Predict sentiment of custom movie review by user

`create_sample_review.py` 

To create review paragraph and save it in JSON file

`predict_sentiment_review.py`

Read the JSON file, use the trained model and print positve or negative review.

#### Sample results
mlartifacts folder

# Development

```
# Clone the repo and cd into the local repo dir
git clone <repo> <dir>
cd <dir>
# Install all the dependencies into the virtual environment
# and activate it
uv sync --all-groups
. .venv/bin/activate
# Install git hook in local repo (in <dir>) to run
# pre-commit checks on commit
pre-commit install
```

## Deactivate Virtual Env
```
deactivate
```








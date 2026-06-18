'''
We need to do this as the new ML flow version is not happy with this cachetools
pip uninstall -y cachetools mlflow
pip install cachetools==5.3.3 mlflow


See this for self hosting options
https://mlflow.org/docs/latest/self-hosting/
'''

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-first-experiment")


# Print connection information
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Active Experiment: {mlflow.get_experiment_by_name('my-first-experiment')}")

# Test logging
with mlflow.start_run():
    mlflow.log_param("test_param", "test_value")
    print("✓ Successfully connected to MLflow!")
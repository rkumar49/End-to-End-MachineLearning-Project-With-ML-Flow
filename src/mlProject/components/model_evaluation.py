import os
import pandas as pd
from urllib.parse import urlparse
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        # Load training data to retrain model
        train_data = pd.read_csv("artifacts/data_transformation/train.csv")
        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]

        # Initialize and train the ElasticNet model with params from config
        model = ElasticNet(
            alpha=self.config.all_params['alpha'],
            l1_ratio=self.config.all_params['l1_ratio']
        )
        model.fit(train_x, train_y)

        # Save the retrained model to the model path
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        joblib.dump(model, self.config.model_path)

        # Load test data and prepare features and labels
        test_data = pd.read_csv(self.config.test_data_path)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        # Predict using the retrained model
        predicted_qualities = model.predict(test_x)

        # Calculate evaluation metrics
        rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

        # Save metrics locally as JSON (optional)
        from src.mlProject.utils.common import save_json  # adjust import if needed
        save_json(path=Path(self.config.metric_file_name), data={
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })

        # Set MLflow tracking URI from config
        mlflow.set_registry_uri(self.config.mlflow_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(self.config.all_params)

            # Log evaluation metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Log the trained model to MLflow model registry if possible
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")

        print(f"Logged model and metrics to MLflow with alpha={self.config.all_params['alpha']}, "
              f"l1_ratio={self.config.all_params['l1_ratio']}")

    

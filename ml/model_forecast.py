import os
from transformers import DataFilter
import mlflow
import logging
import pandas as pd

from pathlib import Path
import yaml


# load config file
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = config["MLFLOW_S3_ENDPOINT_URL"]
os.environ["AWS_ACCESS_KEY_ID"] = config["S3_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = config["S3_SECRET_KEY"]

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

logger.info(config)



class ModelInference():
    def __init__(self):
        mlflow.set_tracking_uri(config["MLFLOW_URL"])
        logger.info('Tracking URL ' + config["MLFLOW_URL"])
        self.client = mlflow.MlflowClient()
        name = config["EXPIRIMENT_NAME"]
        logger.info('Searching for ' + f"name = '{name}'")
        experiment = self.client.search_experiments(filter_string=f"name = '{name}'")[0]
        self.experiment_id = experiment.experiment_id
        self.model = self.load_model()
        self.data_filter = DataFilter()


    def load_model(self):
        runs = self.client.search_runs(self.experiment_id, "", order_by=["metrics.cv_roc_auc DESC"], max_results=1)
        best_run = runs[0].info.run_id
        loaded_model = mlflow.sklearn.load_model(f's3://{config["SOURCE_BUCKET"]}/models/{self.experiment_id}/{best_run}/artifacts/{config["MODEL_NAME"]}')
        return loaded_model
   
    def get_forecast(self, msg):
        df = pd.DataFrame([msg.dict()])
        transformed_df = self.data_filter.transform(df)
        predictions_test = self.model.predict(transformed_df['text'])
        return pd.DataFrame(predictions_test, columns = ['label']).to_dict()


    
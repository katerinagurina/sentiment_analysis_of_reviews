import os
from datetime import datetime
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

import logging

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import transformers
import io
import boto3
import pandas as pd
from pathlib import Path
import yaml


# load config file
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

logger.info(config)



def get_data(file_name):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=config["SOURCE_BUCKET"], Key=f'data/{file_name}')
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return df


def main():    
    
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = config["MLFLOW_S3_ENDPOINT_URL"]
    os.environ["AWS_ACCESS_KEY_ID"] = config["S3_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = config["S3_SECRET_KEY"]
    
    mlflow.set_tracking_uri(config["MLFLOW_URL"])
    client = MlflowClient()
    experiment = client.search_experiments(filter_string=f"name = '{config["EXPIRIMENT_NAME"]}'")[0]
    experiment_id = experiment.experiment_id

    run_name = 'Run time: ' + ' ' + str(datetime.now())

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        pipeline = Pipeline(steps=[
            ('vectorizer', TfidfVectorizer()),
            ('model', XGBClassifier())
            ])
        
        old_train_df = get_data(config["FILE_NAME_FOR_TRAIN"])
        additional_train_df = get_data(config["FILE_NAME_FOR_REFFIT"])
        train_df = pd.concat([old_train_df, additional_train_df], ignore_index = True)

        data_filter = transformers.DataFilter()
        train_df = data_filter.transform(train_df)

        scores = cross_val_score(pipeline, train_df['text'], train_df['Label'], cv = 5, scoring = 'roc_auc')
        mean_roc_auc = scores.mean()
        
        mlflow.log_metrics({'cv_roc_auc' : mean_roc_auc})
        mlflow.log_params(pipeline.get_params()['steps'][1][1].get_params())
        pipeline.fit(train_df['text'], train_df['Label'])
        
        mlflow.sklearn.log_model(
                sk_model=pipeline, artifact_path=config["MODEL_NAME"])
        
        hold_out_df = get_data(config["FILE_NAME_FOR_HOLDOUT"])
        hold_out_df = data_filter.transform(hold_out_df)

        y_forecast = pipeline.predict(hold_out_df['text'])
        hold_out_metric = roc_auc_score(hold_out_df['Label'].values, y_forecast)

        mlflow.log_metrics({'hold_out_roc_auc' : hold_out_metric})
        

if __name__ == "__main__":
    main()
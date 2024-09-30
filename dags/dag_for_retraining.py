import uuid
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.secret import Secret
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.utils.dates import days_ago

aws_access_key_id = Secret('env', 'AWS_ACCESS_KEY_ID', 'ya-s3-secret', 'AWS_ACCESS_KEY_ID')
aws_secret_access_key = Secret('env', 'AWS_SECRET_ACCESS_KEY', 'ya-s3-secret', 'AWS_SECRET_ACCESS_KEY')


with DAG(dag_id="reffit_dag",
         start_date=days_ago(2),
         schedule="*/5 * * * *",
         catchup=False) as dag:
  
  filekey = str(uuid.uuid4())

  task1 = KubernetesPodOperator(
    task_id='sent_analysis_retraining',
    name='sent_analysis_retraining',
    namespace='default',
    image='katerinagurina/sentiment_analysis_project:latest',
    cmds = [
      "python", "ml/model_reffit.py"
    ],
    secrets=[aws_access_key_id, aws_secret_access_key],
    in_cluster=True
  )

task1
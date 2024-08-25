import asyncio
from pathlib import Path

import typer
import aiohttp


import io
import boto3
import pandas as pd
from pathlib import Path
import yaml
import pandas
import logging
import os


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




async def score(session, endpoint: str, joke_num: int, data: dict):
    async with session.post(endpoint, json=data) as response:
      try: 
        result = await response.json()
      except aiohttp.ContentTypeError as err:
        print(err)
      return result


async def run(endpoint: str, df: pandas.DataFrame):
  async with aiohttp.ClientSession() as session:
    def tasks():
      for review_id, review in enumerate(df['Review'].values):
        logger.info('SEND'+ str(review_id))
        yield asyncio.ensure_future(score(session, str(endpoint), review_id, review))
    await asyncio.gather(*tasks())


def main():
  os.environ["MLFLOW_S3_ENDPOINT_URL"] = config["MLFLOW_S3_ENDPOINT_URL"]
  os.environ["AWS_ACCESS_KEY_ID"] = config["S3_KEY_ID"]
  os.environ["AWS_SECRET_ACCESS_KEY"] = config["S3_SECRET_KEY"]
  
  df = get_data(config["FILE_NAME_FOR_TEST"])
  df = pd.concat([df] * 10, ignore_index=True)
  asyncio.run(run(f'http://test.ai/predict', df))


if __name__ == '__main__':
 typer.run(main)
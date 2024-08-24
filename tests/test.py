from fastapi.testclient import TestClient

import sys
 
# setting path
sys.path.append('../app/')

from app import app, load_model
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

client = TestClient(app)
load_model()


def test_healthcheck():
    response = client.get("/healthcheck")
    assert response.status_code == 200


def test_predict():
    data = '{"Review": "Everything is perfect"}'
    response = client.post("/predict", content = 'Content-Type: application/json', data = data)
    logger.info(response)
    assert response.json() == {'label': {'0': 1}}
    assert response.status_code == 200

if __name__ == "__main__":
    test_healthcheck()
    test_predict()

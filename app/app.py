
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import sys
sys.path.append('../ml/')
from model_forecast import ModelInference
from starlette_exporter import PrometheusMiddleware, handle_metrics
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
import logging
from typing import List

class ModelHandler:
    def __init__(self):
        self.model = None

class Review(BaseModel):
    Review: str

class ListOfReviews(BaseModel):
    Reviews: List[Review]

MODEL = ModelHandler()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    MODEL.model = ModelInference()
    yield
    print('Bye!')

def load_model():
    global MODEL
    MODEL.model = ModelInference()

app = FastAPI(lifespan=lifespan)
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)
   
    
@app.get("/healthcheck")
def read_healthcheck():
    return Response(status_code=status.HTTP_200_OK)

@app.post("/predict")
def predict(msg:Review):
    if MODEL.model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    try:
        result = MODEL.model.get_forecast(msg)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
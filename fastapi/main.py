from celery.result import AsyncResult
from fastapi import Body, FastAPI, Form, Request, File, UploadFile, Path
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Union
from worker import create_task
import shutil

tags_metadata = [
    {
        "name": "prediction",
        "description": "Post de dataSet schema and send them to the prediction model training the model",
    },
    {
        "name": "result",
        "description": "Based on the taskID returns the prediction model result.",
    },
]


class FilePrediction(BaseModel):
    predictionType: int
    file: Union[bytes, None] = None
    
app = FastAPI(openapi_tags=tags_metadata)
app.mount("/static", StaticFiles(directory="/"), name="static")

templates = Jinja2Templates(directory="html")

@app.post("/task/prediction", tags=["prediction"], status_code=201)
def run_task(predictionType: int = Form(...),
             predictionFile: UploadFile = File(...)):
    task_type = payload["predictionType"]
    task = create_task(filePrediction.predictionType,filePrediction.file)
    return JSONResponse({"task_id": task.id})


@app.get("/task/result/{task_id}", tags=["result"])
def get_status(task_id):
    task_result = AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return JSONResponse(result)

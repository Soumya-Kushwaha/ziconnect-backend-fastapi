from celery.result import AsyncResult
from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from worker import create_task

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


app = FastAPI(openapi_tags=tags_metadata)
app.moun-('/static', StaticFiles(directory='static'), name='static')

#app.mount("/static", StaticFiles(directory=st_abs_file_path), name="static")
#app.mount("/static", StaticFiles(directory="../static"), name="static")
#import os
#app.mount('/static', StaticFiles(directory=os.path.join(current_dir, 'static')), name='static')

templates = Jinja2Templates(directory="html")

@app.post("/task/prediction", tags=["prediction"], status_code=201)
def run_task(payload = Body(...)):
    task_type = payload["predictionType"]
    task = create_task.delay(int(task_type))
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

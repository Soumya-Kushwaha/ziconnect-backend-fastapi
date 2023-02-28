import os
import json
from fastapi import FastAPI, Body, Form, Request, File, UploadFile, Path
import worker 
from datetime import datetime
from time import mktime
from typing import Union
from celery.result import AsyncResult
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from celery import Celery,uuid


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
    task_type = predictionType
    task_name = "uploadFile_task"

    """ CheckFile """
    taskId = uuid()
    fileName = taskId + "_" + predictionFile.filename
    targetDir = '/usr/src/app/files/'
    filePath = os.path.join(targetDir, fileName)
    with open(filePath, mode='wb+') as f:
        f.write(predictionFile.file.read())
     
    result = worker.app.send_task(task_name, args=[filePath], kwargs={},queue='celery', routing_key='key_result_processing')
    return JSONResponse({"task_id": result.id})

@app.get("/task/result/{task_id}", tags=["result"])
def get_status(task_id):
    task_result =  worker.app.AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return JSONResponse(result)

class EncoderObj(json.JSONEncoder):   
    def default(self, obj):
        if isinstance(obj, datetime):
            return {
                '__type__': '__datetime__', 
                'epoch': int(mktime(obj.timetuple()))
            }
        else:
            return json.JSONEncoder.default(self, obj)

def decoderObj(obj):
    if '__type__' in obj:
        if obj['__type__'] == '__datetime__':
            return datetime.fromtimestamp(obj['epoch'])
    return obj

# Encoder function      
def dumpEncode(obj):
    return json.dumps(obj, cls=EncoderObj)

# Decoder function
def loadDecode(obj):
    return json.loads(obj, object_hook=decoderObj)



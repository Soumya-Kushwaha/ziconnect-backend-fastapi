import base64
import os
import json
from fastapi import FastAPI, Body, Form, Request, File, UploadFile, Path
import worker 
from worker import app as celery_app
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
def run_task(localityFile: UploadFile = File(...),
             schoolFile: UploadFile = File(...)):
    try:

        task_name = "uploadFile_task"
        taskId = uuid()
        
        """ Import Files """
        
        csvLocalityFile = taskId + "_" + localityFile.filename
        localityFileName = taskId + "_" + csvLocalityFile.filename
        targetDir = '/var/lib/docker/volumes/fastapi-storage/_data/'
        filePath = os.path.join(targetDir, localityFileName)
        with open(filePath, mode='wb+') as f:
            f.write(localityFile.file.read())
        
        
        csvSchoolFile = taskId + "_" + schoolFile.filename
        schoolFileName = taskId + "_" + csvSchoolFile.filename
        targetDir = '/var/lib/docker/volumes/fastapi-storage/_data/'
        filePath = os.path.join(targetDir, schoolFileName)
        with open(filePath, mode='wb+') as f:
            f.write(schoolFile.file.read())
        
        result = celery_app.send_task(task_name, args=[filePath], kwargs=None)
        return JSONResponse({"task_id": result.id})

    except Exception as ex:
        return JSONResponse(content=ex,status_code=400)


@app.get("/task/result/{task_id}", tags=["result"])
def get_status(task_id):
    try:
        task_result =  worker.app.AsyncResult(task_id)
        result = {
            "task_id": task_id,
            "task_status": task_result.status,
            "task_result": task_result.result
        }
        return JSONResponse(result)

    except Exception as ex:
        return JSONResponse(content=ex,status_code=400)

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



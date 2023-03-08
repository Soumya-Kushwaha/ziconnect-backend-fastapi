import os
import json
from fastapi import FastAPI, Body, Form, Request, File, UploadFile, Path
import worker 
from worker import app as celery_app
from datetime import datetime, time
from time import mktime
from typing import Union
from celery import Celery,uuid
from celery.result import AsyncResult
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from celery.exceptions import TimeoutError


tags_metadata = [
    {
        "name": "prediction",
        "description": "Post de dataSet(Locality and Schools schema) and send them to the prediction model training the model",
    },
    {
        "name": "result",
        "description": "Based on the taskID returns the prediction model result.",
    },
    {
        "name": "healthCheck",
        "description": "Check application availability.",
    }
]


class FilePrediction(BaseModel):
    predictionType: int
    file: Union[bytes, None] = None
    
app = FastAPI(openapi_tags=tags_metadata)
app.mount("/static", StaticFiles(directory="/"), name="static")

templates = Jinja2Templates(directory="html")

@app.get('/health', tags=["healthCheck"], status_code=201)
async def service_health():
    """Return service health"""
    return JSONResponse(content='ok', status_code=200)
    

@app.post("/task/prediction", tags=["prediction"], status_code=201)
def run_task(localityFile: UploadFile = File(...),
             schoolFile: UploadFile = File(...)):
    try:
        task_name = "uploadFile_task"
        targetFilePath = '/var/lib/docker/volumes/fastapi-storage/_data/'
        taskId = uuid()
        
        """ Import Files (Locality/School) """
        csvLocalityFile = taskId + "_" + localityFile.filename
        localityFileName = taskId + "_" + csvLocalityFile
        localityLocalFilePath = os.path.join(targetFilePath, localityFileName)
        with open(localityLocalFilePath, mode='wb+') as f:
            f.write(localityFile.file.read())
        
        csvSchoolFile = taskId + "_" + schoolFile.filename
        schoolFileName = taskId + "_" + csvSchoolFile
        schoolLocalFilePath = os.path.join(targetFilePath, schoolFileName)
        with open(schoolLocalFilePath, mode='wb+') as f:
            f.write(schoolFile.file.read())
        
        result = celery_app.send_task(task_name, args=[localityLocalFilePath,schoolLocalFilePath], kwargs=None)
        return JSONResponse({"task_id": result.id})

    except Exception as ex:
        return JSONResponse(content=ex,status_code=400)


@app.get("/task/result/{task_id}", tags=["result"])
def get_status(task_id):
    try:
        urlReq =  'http://dashboard:5555/api/task/info/' + task_id

        getRespTask = requests.get(urlReq).text

        if (getRespTask == ''):
             return JSONResponse(content="TaskID not found", status_code=400)
        
        parsed_json = json.loads(getRespTask)
        taskState = parsed_json['state']
        taskTimestamp = None
        taskSucceeded = None
        taskFailed = None

        if (taskState == 'STARTED' or taskState == 'PENDING'):
            taskdateStarted = datetime.fromtimestamp(parsed_json['started']).strftime('%Y-%m-%dT%H:%M:%S.%f%z')
            taskReceivedDate = datetime.fromtimestamp(parsed_json['received']).strftime('%Y-%m-%dT%H:%M:%S.%f%z')
            taskTimestamp = datetime.fromtimestamp(parsed_json['timestamp']).strftime('%Y-%m-%dT%H:%M:%S.%f%z')

            response = {
                    "taskID" : parsed_json['uuid'],
                    "taskName" : parsed_json['name'],
                    "taskState" : parsed_json['state'],
                    "taskStartedDate" : taskdateStarted,     
                    "taskReceivedDate" : taskReceivedDate,
                    "taskFailed" : taskFailed,
                    "taskResult" : parsed_json['result'],
                    "taskTimestamp" : taskTimestamp,
                    "taskRejected" : parsed_json['rejected'],
                    "taskSucceeded" : taskSucceeded,
                    "taskException" : parsed_json['exception']
            }
            return JSONResponse(content=response, status_code=200)
        

        taskTimestamp = datetime.fromtimestamp(parsed_json['timestamp']).strftime('%Y-%m-%dT%H:%M:%S.%f%z')
        taskSucceeded = datetime.fromtimestamp(parsed_json['succeeded']).strftime('%Y-%m-%dT%H:%M:%S.%f%z')
        taskFailed = datetime.fromtimestamp(parsed_json['failed']).strftime('%Y-%m-%dT%H:%M:%S.%f%z')

        response = {
                    "taskID" : parsed_json['uuid'],
                    "taskName" : parsed_json['name'],
                    "taskState" : parsed_json['state'],
                    "taskStartedDate" : taskdateStarted,     
                    "taskReceivedDate" : taskReceivedDate,
                    "taskFailed" : taskFailed,
                    "taskResult" : parsed_json['result'],
                    "taskTimestamp" : taskTimestamp,
                    "taskRejected" : parsed_json['rejected'],
                    "taskSucceeded" : taskSucceeded,
                    "taskException" : parsed_json['exception']
            }
        return JSONResponse(content=response, status_code=200)

    except HTTPException as exGet:
        return JSONResponse(status_code=400,detail=exGet)

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



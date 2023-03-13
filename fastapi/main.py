import os
import json
import worker 
import requests

from worker import app as celery_app
from datetime import datetime
from pydantic import BaseModel
from time import mktime
from typing import Union

from celery import Celery, uuid
from celery.result import AsyncResult
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, HTTPException


tags_metadata = [
    {
        "name": "prediction",
        "description": "Post the dataset (Locality and Schools schema) and" \
                       + " send them to the prediction model training the model",
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

@app.get('/health', tags=["healthCheck"], status_code=200)
async def service_health() -> JSONResponse:
    try:
        """Return service health"""        
        return JSONResponse(content='ok', status_code=200)
    except Exception as ex:
        return JSONResponse(content=ex, status_code=404)

@app.post("/task/prediction", tags=["prediction"], status_code=200)
def run_task(locality_file: UploadFile = File(...),
             school_file: UploadFile = File(...)
             ) -> JSONResponse:
    try:
        task_name = "uploadFile_task"
        target_dirpath = '/var/lib/docker/volumes/fastapi-storage/_data/'
        task_id = uuid()

        # Import Files (Locality / School)
        locality_filename = f'{task_id}_{locality_file.filename}'
        locality_local_filepath = os.path.join(target_dirpath, locality_filename)
        with open(locality_local_filepath, mode='wb+') as f:
            f.write(locality_file.file.read())

        school_filename = f'{task_id}_{school_file.filename}'
        school_local_filepath = os.path.join(target_dirpath, school_filename)
        with open(school_local_filepath, mode='wb+') as f:
            f.write(school_file.file.read())

        args = [locality_local_filepath, school_local_filepath]
        result = celery_app.send_task(task_name, args=args, kwargs=None)
        return JSONResponse({"task_id": result.id})

    except Exception as ex:
        return JSONResponse(content=ex, status_code=404)


@app.get("/task/result/{task_id}", tags=["result"])
def get_status(task_id: Union[int, str]) -> JSONResponse:
    try:
        request_url = f'http://dashboard:5555/api/task/info/{task_id}'

        response = requests.get(request_url)
        if response.text == '':
            return JSONResponse(content="TaskID not found", status_code=404)

        parsed_json = json.loads(response.text)
        def get_date_field(field: str) -> str:
            if field not in parsed_json:
                return ''
            value = parsed_json[field]
            date_format = '%Y-%m-%dT%H:%M:%S.%f%z'
            return datetime.fromtimestamp(value).strftime(date_format)

        task_started = None
        task_received = get_date_field('received')
        task_timestamp = get_date_field('timestamp')
        task_succeeded = None
        task_failed = None

        task_state = parsed_json['state']
        if task_state not in ['STARTED', 'PENDING','RECEIVED','SUCCESS']:
            task_failed = get_date_field('failed')
        
        if task_state in ['FAILURE']:
            task_succeeded = None
            task_started = get_date_field('started') 
        
        if task_state in ['SUCCESS']:
            task_succeeded = get_date_field('succeeded')
        
        if task_state in ['RECEIVED']:
            task_started = None

        if task_state in ['STARTED', 'PENDING','SUCCESS']:           
            task_started = get_date_field('started') 
            task_failed = None
            
        response = {
            "taskID" : parsed_json['uuid'],
            "taskName" : parsed_json['name'],
            "taskState" : parsed_json['state'],
            "taskTimestamp" : task_timestamp,
            "taskStartedDate" : task_started,     
            "taskReceivedDate" : task_received,
            "taskFailedDate" : task_failed,
            "taskSucceededDate" : task_succeeded,
            "taskResult" : parsed_json['result'],
            "taskRejected" : parsed_json['rejected'],
            "taskException" : parsed_json['exception']
        }
        return JSONResponse(content=response, status_code=200)

    except HTTPException as ex:
        return JSONResponse(content=ex, status_code=404)

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

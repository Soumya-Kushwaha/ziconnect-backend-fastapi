import os
import json
import worker
import requests

from worker import app as celery_app
from datetime import datetime
from pydantic import BaseModel
from time import mktime
from typing import Union, Dict

from celery import Celery, uuid
from celery.result import AsyncResult
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware


tags_metadata = [
    {
        "name": "prediction",
        "description": "Post the dataset (Locality and Schools schema) and" \
                       + " send them to the prediction model training the model",
    },
    {
        "name": "socialimpact",
        "description": "Post the dataset for social impact analysis and" \
                       + " send them to the prediction model training the model and present the results",
    },
    {
        "name": "result",
        "description": "Based on the taskID returns the prediction model result.",
    },
    {
        "name": "info",
        "description": "Based on the taskID returns the task's informations.",
    },
    {
        "name": "healthCheck",
        "description": "Check application availability.",
    }
]


class FilePrediction(BaseModel):
    predictionType: int
    file: Union[bytes, None] = None


urlFlowerApi = 'http://0.0.0.0:5556/api'
target_dirpath = '/var/lib/docker/volumes/fastapi-storage/_data/'

origins = ["*"]
app = FastAPI(openapi_tags=tags_metadata)
app.add_middleware(
                    CORSMiddleware,
                    allow_origins=origins,
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                  )

app.mount("/static", StaticFiles(directory="/"), name="static")

templates = Jinja2Templates(directory="html")

@app.get("/health", tags=["healthCheck"], status_code=200)
def service_health() -> JSONResponse:
    try:
        """Return service health"""
        return JSONResponse(content='ok', status_code=200)
    except Exception as ex:
        return JSONResponse(content=ex, status_code=400)


@app.post("/task/prediction", tags=["prediction"], status_code=200)
def run_task(locality_file: UploadFile = File(...),
             school_file: UploadFile = File(...)
             ) -> JSONResponse: # pragma: no cover
    try:
        task_name = "uploadFile_task"
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
        return JSONResponse(content=ex, status_code=500)


@app.post("/task/socialimpact", tags=["socialimpact"], status_code=200)
def run_socialimpact_task(locality_history: UploadFile = File(...),
                          school_history: UploadFile = File(...),
                          homogenize_columns: list[str] = None
                         ) -> JSONResponse: # pragma: no cover
    try:
        task_name = "uploadSocialImpactFile_task"
        task_id = uuid()

        # Import Files (Locality / School)
        locality_filename = f'{task_id}_{locality_history.filename}'
        locality_local_filepath = os.path.join(target_dirpath, locality_filename)
        with open(locality_local_filepath, mode='wb+') as f:
            f.write(locality_history.file.read())

        # Import Files (School)
        school_filename = f'{task_id}_{school_history.filename}'
        school_local_filepath = os.path.join(target_dirpath, school_filename)
        with open(school_local_filepath, mode='wb+') as f:
            f.write(school_history.file.read())

        args = [locality_local_filepath, school_local_filepath]
        result = celery_app.send_task(task_name, args=args, kwargs=None)
        return JSONResponse({"task_id": result.id})

    except Exception as ex:
        return JSONResponse(content=ex, status_code=500)

def parse_failure_exception(exception: str) -> Dict:
    """Parse the exception message and return the error message"""
    try:
        exception = exception.split('(', 1)[1].rsplit(')', 1)[0]
        return json.loads(exception)
    except:
        return {"error": exception}

@app.get("/task/result/{task_id}", tags=["result"])
def get_result(task_id: Union[int, str]) -> JSONResponse: # pragma: no cover
    try:
        request_url = f'{urlFlowerApi}/task/result/{task_id}'
        response = requests.get(request_url).json()
        task_result = None

        task_state = response['state']
        if task_state == 'SUCCESS':
            task_result = response['result']
        elif task_state == 'FAILURE':
            task_result = parse_failure_exception(response['result'])

        response = {
            "taskResult" : task_result
        }
        return JSONResponse(content=response, status_code=200)

    except HTTPException as ex:
        return JSONResponse(content=ex, status_code=500 )

@app.get("/task/info/{task_id}", tags=["info"])
def get_status(task_id: Union[int, str]) -> JSONResponse: # pragma: no cover
    try:
        request_url = f'{urlFlowerApi}/task/info/{task_id}'

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
        task_exception = None

        task_state = parsed_json['state']
        if task_state not in ['STARTED', 'REJECT', 'PENDING','RECEIVED','SUCCESS']:
            task_failed = get_date_field('failed')

        if task_state in ['FAILURE']:
            task_succeeded = None
            task_started = get_date_field('started')
            task_exception = parse_failure_exception(parsed_json['exception'])

        if task_state in ['SUCCESS']:
            task_succeeded = get_date_field('succeeded')

        if task_state in ['RECEIVED']:
            task_started = None

        if task_state in ['STARTED', 'REJECT', 'PENDING', 'SUCCESS']:
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
            "taskException" : task_exception
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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

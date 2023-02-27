import os
import base64
from time import sleep
from celery import Celery
from fastapi import  FastAPI, File, UploadFile
import csv

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")


@celery.task(name="create_task")
def create_task(predictionType,predictionFile):
    idTaskCelery = celery.AsyncResult.task_id
    uploadFile_task(idTaskCelery,predictionFile)   
    return True


#@celery.task(name="uploadfile_task")
def uploadFile_task(idTaskCelery, predictionFile):
    
    """ CheckFile """
    fileName = predictionFile.filename
    targetDir = '/usr/src/app/files/'
    filePath = os.path.join(targetDir, fileName)
    with open(filePath, mode='wb+') as f:
        f.write(predictionFile.file.read())
    
    """ Convert to JsonFormat """
    data = {}
    with open(filePath, mode='r', encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        for rows in csvReader:             
            key = rows['school_name']
            data[key] = rows    
    return {data}


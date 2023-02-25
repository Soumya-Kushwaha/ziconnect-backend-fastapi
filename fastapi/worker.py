import os
import time
import shutil

from celery import Celery
from fastapi import  File, UploadFile

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")


@celery.task(name="create_task")
def create_task(predictionType,predictionFile):
    idTaskCelery = celery.AsyncResult.task_id
    file_location = f"files/{idTaskCelery}"
    with open(file_location, "wb+") as file_object:
        file_object.write(predictionFile.file.read())
    return True

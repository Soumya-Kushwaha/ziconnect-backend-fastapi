import os
from time import sleep
from celery import Celery
from celery.utils.log import get_task_logger
import traceback
import subprocess

app = Celery(__name__, include=['worker', 'celery.app.builtins'])
app.conf.broker_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
app.conf.result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
app.conf.update(result_extended=True)
celery_log = get_task_logger(__name__)

@app.task(name="uploadFile_task")
def uploadFile_task(localityLocalFilePath,schoolLocalFilePath):
    try:

        """ Send files to predict """ 
        filePrediction = 'services/internetConnectivityService.py'     
        pipe = subprocess.run(["python3.9", filePrediction , localityLocalFilePath, schoolLocalFilePath,],stdout=subprocess.PIPE,text=True)
        return pipe.stdout

    except Exception as ex:
        meta = {
            'exc_type': type(ex).__name__,
            'exc_message': traceback.format_exc().split('\n')
        }
    raise meta

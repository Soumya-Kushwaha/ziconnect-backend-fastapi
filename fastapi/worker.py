import os
from time import sleep
from celery import Celery
from celery.utils.log import get_task_logger
import traceback
import subprocess
import json
from services.internetConnectivityService import *
import pandas as pd

app = Celery(__name__, include=['worker'])
app.conf.broker_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
app.conf.result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
app.conf.update(result_extended=True,task_track_started=True)
celery_log = get_task_logger(__name__)

@app.task(name="uploadFile_task")
def uploadFile_task(localityLocalFilePath,schoolLocalFilePath):
    try:

        """ Send files to predict """ 
        connectivity_dl = InternetConnectivityDataLoader(pd.read_csv(localityLocalFilePath), pd.read_csv(schoolLocalFilePath))
        connectivity_dl.setup()

        # Train the model
        model = InternetConnectivityModel()
        result = model.fit(connectivity_dl.train_dataset)
        import json
        return json.dumps(result, indent=4)

        #filePrediction = 'services/internetConnectivityService.py'     
        #pipe = subprocess.run(["python3.9", filePrediction , localityLocalFilePath, schoolLocalFilePath,],stdout=subprocess.PIPE,text=True)
        #return pipe.stdout

    except Exception as ex:
        meta = {
            'exc_type': type(ex).__name__,
            'exc_message': traceback.format_exc().split('\n')
        }
    raise meta

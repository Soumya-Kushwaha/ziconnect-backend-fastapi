import os
from time import sleep
from celery import Celery
from celery.utils.log import get_task_logger
import csv
import traceback

app = Celery(__name__,include=['worker', 'celery.app.builtins'])
app.conf.broker_url = os.getenv("CELERY_BROKER_URL", "amqp://localhost:5672")
app.conf.result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379")
app.conf.update(result_extended=True)
celery_log = get_task_logger(__name__)

@app.task(name="uploadFile_task", bind=True)
def uploadFile_task(filePath):
    try:        
        """ Convert to JsonFormat """
        data = {}
        with open(filePath, mode='r', encoding='utf-8') as csvf:
            csvReader = csv.DictReader(csvf)
            for rows in csvReader:             
                key = rows['school_name']
                data[key] = rows    
        return {"result": "Escolas {}".format(data)}
    except Exception as ex:
            meta={
                'exc_type': type(ex).__name__,
                'exc_message': traceback.format_exc().split('\n')
                 }
    raise meta





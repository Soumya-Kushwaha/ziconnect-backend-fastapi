import os
from time import sleep
from celery import Celery
from celery.utils.log import get_task_logger
import csv
import traceback

app = Celery(__name__,include=['worker', 'celery.app.builtins'])
app.conf.broker_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
app.conf.result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
app.conf.update(result_extended=True)
celery_log = get_task_logger(__name__)
rowsLimit = 100

@app.task(name="uploadFile_task", bind=True)
def uploadFile_task(filePath):
    try:        
        """ Convert to JsonFormat """
        data = {}
        with open(filePath, mode='r', encoding='utf-8') as csvf:
            csvReader = csv.DictReader(csvf)
            for idx, rows in enumerate(csvReader, 1):
                for row in rows:             
                    key = row['school_name']
                    data[key] = row['school_name']
                    if idx == rowsLimit:
                        break
        return {"result": "Escolas {}".format(data.key['school_name'])}
        
    except Exception as ex:
            meta={
                'exc_type': type(ex).__name__,
                'exc_message': traceback.format_exc().split('\n')
                 }
    raise meta





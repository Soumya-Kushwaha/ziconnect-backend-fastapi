import importlib
import sys
import logging
from celery import Task

from .celery import worker


class PredictTask(Task):
    #Celery's Task class Abstraction

    abstract = True

    def __init__(self):
        super().__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        
       # Load model on first call
       # Avoids the need to load model on each task request
        
        if not self.model:
            logging.info("Loading connectivityModel...")
            sys.path.append("..")
            module_import = importlib.import_module(self.path[0])
            model_obj = getattr(module_import, self.path[1])
            self.model = model_obj()
            logging.info("Prediction model ready!")
        return self.run(*args, **kwargs)


@worker.task(
    ignore_result=False,
    bind=True,
    base=PredictTask,
    path=("logic.model", "ConnectivityModel"),
    name="{}.{}".format(__name__, "Connectivity"),
)

def predict(self, x):
    return self.model.predict(x)
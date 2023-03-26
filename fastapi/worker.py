import json
import os
import traceback
import pandas as pd
import pytest

from celery import Celery
from celery.utils.log import get_task_logger
from services.internetConnectivityService import (
    InternetConnectivityDataLoader,
    InternetConnectivityModel,
    InternetConnectivitySummarizer
)

app = Celery(__name__, include=['worker'])
app.conf.broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
app.conf.result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
app.conf.update(result_extended=True, task_track_started=True)
celery_log = get_task_logger(__name__)

@app.task(name="uploadFile_task")
def uploadFile_task(locality_local_filepath: str, school_local_filepath: str) -> str:
    try:

        # Convert raw tables to dataset
        locality_df = pd.read_csv(locality_local_filepath, sep=',', encoding='utf-8')
        school_df = pd.read_csv(school_local_filepath, sep=',', encoding='utf-8')
        connectivity_dl = InternetConnectivityDataLoader(locality_df, school_df)
        connectivity_dl.setup()

        # Train the model
        model = InternetConnectivityModel()
        model_metrics = model.fit(connectivity_dl.train_dataset)

        # Predict
        full_dataset = pd.concat([connectivity_dl.train_dataset,
                                  connectivity_dl.test_dataset])
        predictions = model.predict(full_dataset)

        # Connectivity summary
        full_dataset['internet_availability_prediction'] = predictions
        summarizer = InternetConnectivitySummarizer()
        result_summary = summarizer.compute_statistics_by_locality(full_dataset)
        print(json.dumps(result_summary, indent=4))

        response = {
            'model_metrics': model_metrics,
            'result_summary': result_summary
        }
        return response

    except Exception as ex:
        raise RuntimeError({
            'exception_type': type(ex).__name__,
            'exception_message': traceback.format_exc().split('\n')
        })


@app.task(name="uploadSocialImpactFile_task")
def uploadSocialImpactFile_task(localityHistory_local_filepath: str, schoolHistory_local_filepath: str) -> str:
    try:

        # Convert raw tables to dataset
        locality_df = pd.read_csv(localityHistory_local_filepath, sep=',', encoding='utf-8')
        school_df = pd.read_csv(schoolHistory_local_filepath, sep=',', encoding='utf-8')
        connectivity_dl = InternetConnectivityDataLoader(locality_df, school_df)
        connectivity_dl.setup()

        # Train the model
        model = InternetConnectivityModel()
        model_metrics = model.fit(connectivity_dl.train_dataset)

        # Predict
        full_dataset = pd.concat([connectivity_dl.train_dataset,
                                  connectivity_dl.test_dataset])
        predictions = model.predict(full_dataset)

        # Connectivity summary
        full_dataset['internet_availability_prediction'] = predictions
        summarizer = InternetConnectivitySummarizer()
        result_summary = summarizer.compute_statistics_by_locality(full_dataset)
        print(json.dumps(result_summary, indent=4))

        response = {
            'model_metrics': model_metrics,
            'result_summary': result_summary
        }
        return response

    except Exception as ex:
        raise RuntimeError({
            'exception_type': type(ex).__name__,
            'exception_message': traceback.format_exc().split('\n')
        })

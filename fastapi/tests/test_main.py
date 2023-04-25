import json
import pytest
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from fastapi.testclient import TestClient
import requests
from pytest_mock import MockerFixture
from unittest.mock import patch, call
import unittest 
from worker import *
from main import app
from starlette.testclient import TestClient
import pandas as pd
from celery import Celery, uuid
from main import *

client = TestClient(app)


def test_getHealthCheck():
    response = client.get("/health")
    assert response.status_code == 200

def route_with_http_exception():
    response = client.get("/health")
    raise HTTPException(status_code=400)

def test_postTaskPrediction():

    df = pd.DataFrame({
        'state_code': ['CA', 'CA', 'NY', 'NY', 'NY'],
        'municipality_code': ['LA', 'LA', 'NYC', 'NYC', 'BUF'],
        'school_code': ['1', '2', '3', '4', '5'],
        'school_type': ['State', 'Local', 'Federal', 'State', 'State'],
        'school_region': ['Urban', 'Rural', 'Urban', 'Urban', 'Urban'],
        'student_count': [100, 200, 150, 300, 250],
        'internet_availability': ['Yes', 'No', 'Yes', 'NA', 'NA'],
        'internet_availability_prediction': ['Yes', 'No', 'Yes', 'No', 'No'],
    })

    task_name = "uploadFile_task"
    target_dirpath = '/var/lib/docker/volumes/fastapi-storage/_data/'
    task_id = uuid()

    assert task_name == "uploadFile_task"
    assert target_dirpath == '/var/lib/docker/volumes/fastapi-storage/_data/'
    assert task_id is not None

    data_modeling = { "locality_file": df, "school_file": df }
    header = { 'Content-Type': 'application/x-www-form-urlencoded' }
    response = client.post("/task/prediction", params=data_modeling, headers=header)
    assert response.status_code != 200


def test_postTaskEmployabilityImpact():

    df = pd.DataFrame({
        'state_code': ['CA', 'CA', 'NY', 'NY', 'NY'],
        'municipality_code': ['LA', 'LA', 'NYC', 'NYC', 'BUF'],
        'school_code': ['1', '2', '3', '4', '5'],
        'school_type': ['State', 'Local', 'Federal', 'State', 'State'],
        'school_region': ['Urban', 'Rural', 'Urban', 'Urban', 'Urban'],
        'student_count': [100, 200, 150, 300, 250],
        'internet_availability': ['Yes', 'No', 'Yes', 'NA', 'NA'],
        'internet_availability_prediction': ['Yes', 'No', 'Yes', 'No', 'No'],
    })

    task_name = "uploadEmployabilityImpactFile_task"
    assert task_name is not None

    target_dirpath = '/var/lib/docker/volumes/fastapi-storage/_data/'
    assert target_dirpath is not None

    data_modeling = {
        "employability_history_file": df,
        "school_history_file": df,
        "homogenize_columns": "state_code,hdi,population_size"
    }
    header = { 'Content-Type': 'application/x-www-form-urlencoded'}
    response = client.post("/task/employability-impact", params=data_modeling, headers=header)

    assert response.status_code != 200

def test_getTaskResult():
    url_request = "/task/result/5984d769-7805-4fdb-81fc-da68fad134fe"
    response = client.get(url_request)
    assert response.status_code == 200


def test_getTaskInfo():
    url_request = "/task/info/5984d769-7805-4fdb-81fc-da68fad134fe"
    response = client.get(url_request)
    print(response)
    assert response.status_code != 200


def test_worker_throws_exception():
    try:
        df = pd.DataFrame({
            'state_code': ['CA', 'CA', 'NY', 'NY', 'NY'],
            'municipality_code': ['LA', 'LA', 'NYC', 'NYC', 'BUF'],
            'school_code': ['1', '2', '3', '4', '5'],
            'school_type': ['State', 'Local', 'Federal', 'State', 'State'],
            'school_region': ['Urban', 'Rural', 'Urban', 'Urban', 'Urban'],
            'student_count': [100, 200, 150, 300, 250],
            'internet_availability': ['Yes', 'No', 'Yes', 'NA', 'NA'],
            'internet_availability_prediction': ['Yes', 'No', 'Yes', 'No', 'No'],
         })
        uploadFile_task(df, df)
    except RuntimeError:
        pass
    except Exception as ex:
         raise RuntimeError({
            'exception_type': type(ex).__name__,
            'exception_message': traceback.format_exc().split('\n'),
            'schema_eror': None
        })

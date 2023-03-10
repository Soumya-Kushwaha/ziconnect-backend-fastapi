import requests
import csv
from unittest.mock import patch, call
from unittest import TestCase
from worker import uploadFile_task
from main import app
from starlette.testclient import TestClient

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_taskPrediction():

    row_list = [["id", "country_code", "country_name"],
                [1, "BR", "Brasil"]
                [2, "BR", "Brazil"]]
    with open('locality.csv', 'w', newline='') as locality:
        writer = csv.writer(locality)
        writer.writerows(row_list)

    form_data = {
        "locality_file": locality,
        "school_file": locality
    }

    response = client.post("/task/prediction", 
                            json=form_data,
                            headers={ 'Content-Type': 'application/x-www-form-urlencoded'})

    result = response.json()
    assert result.status_code == 200

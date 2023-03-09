import json
import os
import requests
from unittest.mock import patch, call
from unittest import TestCase
from worker import uploadFile_task
from main import * 
import pytest


endpointDashFlower = 'https://fastapi-homolog.jobzi.com'

def test_service_health_ok():      
     response = requests.get(endpointDashFlower + '/health')
     assert response.status_code == 200
    
def test_service_health_nok():
     response = requests.get(endpointDashFlower + '/health')
     assert response.status_code == 404

def test_service_health_server_nok():
     response = requests.get(endpointDashFlower + '/health')
     assert response.status_code == 500
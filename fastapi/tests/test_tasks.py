import json
import os
import requests
from unittest.mock import patch, call
from unittest import TestCase
from worker import uploadFile_task
from main import * 
import pytest


endpointDashFlower = 'http://dashboard:5555/'

def test_service_health_ok():      
        result = service_health() 
        dataLoad = json.loads(result.content)    
        assert  int(dataLoad['status_code']) == 200 
    
def test_service_health_nok():
        result = service_health()
        dataLoad = json.loads(result.content)      
        assert  int(dataLoad['status_code']) == 404  
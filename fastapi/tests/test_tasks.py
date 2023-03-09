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
        dataLoad = result.status_code    
        assert  dataLoad == 200 
    
def test_service_health_nok():
        result = service_health()
        dataLoad = result.status_code       
        assert  dataLoad == 404  
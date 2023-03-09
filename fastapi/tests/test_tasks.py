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
        result = JSONResponse(content='ok', status_code=200)      
        assert service_health() == result
    
def test_service_health_nok():
        ex = Exception
        result = JSONResponse(content=ex, status_code=404)      
        assert service_health() == result
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
        response = service_health()
        assert (response,JSONResponse(content=response, status_code=200))
    
def test_service_health_nok():
        response = service_health()
        assert (response,JSONResponse(content=response, status_code=404))
import json
import os
from unittest.mock import patch, call
from worker import uploadFile_task
from main import *


def test_service_health_ok():
    from main import service_health 
    with service_health.raises(JSONResponse(content='ok', status_code=200)):
        service_health()

def test_service_health_nok():
    from main import service_health 
    with service_health.raises(Exception):
        service_health()


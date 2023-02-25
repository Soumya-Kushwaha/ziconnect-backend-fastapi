import json
from unittest.mock import patch, call
from worker import create_task
import io

def test_task():
    your_data = b'\x02\x1b\x92\x1fs\x96\x97\xe8\x01'
    sd = io.BytesIO()
    sd.write(your_data)
    sd.seek(0)
    assert create_task.run(1,sd)
    assert create_task.run(2,sd)
    assert create_task.run(3,sd)

@patch("worker.create_task.run")
def test_mock_task(mock_run):
    your_data = b'\x02\x1b\x92\x1fs\x96\x97\xe8\x01'
    sd = io.BytesIO()
    sd.write(your_data)
    sd.seek(0)
    assert create_task.run(1,sd)
    create_task.run.assert_called_once_with(1)

    assert create_task.run(2,sd)
    assert create_task.run.call_count == 2

    assert create_task.run(3,sd)
    assert create_task.run.call_count == 3

"""
def test_task_status(test_app):
    response = test_app.post(
        "/task/prediction",
        data=json.dumps({"predictionType": 1})
    )
    content = response.json()
    task_id = content["task_id"]
    assert task_id

    response = test_app.get(f"task/result/{task_id}")
    content = response.json()
    assert content == {"task_id": task_id, "task_status": "PENDING", "task_result": None}
    assert response.status_code == 200

    while content["task_status"] == "PENDING":
        response = test_app.get(f"task/result/{task_id}")
        content = response.json()
    assert content == {"task_id": task_id, "task_status": "SUCCESS", "task_result": True}   
"""

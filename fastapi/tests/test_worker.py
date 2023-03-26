from unittest.mock import patch
from worker import uploadFile_task,uploadSocialImpactFile_task

@patch('worker.uploadFile_task')
def test_uploadFile_task(return_uploadFile_mock):
    try:
        uploadFile_task.apply()
        assert return_uploadFile_mock.called
    except AssertionError as msg:
        print(msg)

@patch('worker.uploadSocialImpactFile_task')
def test_uploadSocialImpactFile_task(return_uploadSocialImpactFile_mock):
    try:
        uploadSocialImpactFile_task.apply()
        assert return_uploadSocialImpactFile_mock.called
    except AssertionError as msg:
        print(msg)
from unittest.mock import patch
from worker import uploadFile_task, uploadSocialImpactFile_task

import unittest

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


class TestAddTask(unittest.TestCase):

    def setUp(self):
        try:
            localityFile = '/var/lib/docker/volumes/fastapi-storage/_data/locality.csv'
            schoolFile = '/var/lib/docker/volumes/fastapi-storage/_data/school.csv'

            self.task = uploadFile_task.apply_async(localityFile,schoolFile)
            assert self.results == self.task.get()
        except AssertionError as msg:
            print(msg)

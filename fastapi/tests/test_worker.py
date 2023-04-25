from unittest.mock import patch
from worker import uploadFile_task, uploadEmployabilityImpactFile_task

import unittest

@patch('worker.uploadFile_task')
def test_uploadFile_task(return_uploadFile_mock):
    try:
        uploadFile_task.apply()
        assert return_uploadFile_mock.called
    except AssertionError as msg:
        print(msg)

@patch('worker.uploadEmployabilityImpactFile_task')
def test_uploadEmployabilityImpactFile_task(return_uploadEmployabilityImpactFile_mock):
    try:
        uploadEmployabilityImpactFile_task.apply()
        assert return_uploadEmployabilityImpactFile_mock.called
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

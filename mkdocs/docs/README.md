[![codecov](https://codecov.io/gh/Jobzi-Artificial-Intelligence/ziconnect-backend-fastapi/branch/master/graph/badge.svg)](https://codecov.io/gh/Jobzi-Artificial-Intelligence/ziconnect-backend-fastapi)

# Predicting school connectivity and employability in Brazil #

Jobzi has a considerable knowledge and experience to provide AI NLP solution designed to guide, contribute and merge the Job and Education market. As a part of this experience we are helping the Unicef and Giga a platform 
and AI prediction models to identify Internet availability in Brazilian schools using geographical data and information on employability and internet connectivity for the schools' region.

As a result of this Unicef project, we have a platform featured for connectivity dashboard provided through the 
AI Models for predicting Internet availability in Brazilian schools using geographical data and information on employability for the schools' region.

### What is this repository for? ###

This repository presents the process to work with the async requests for the prediction model int the platform based on pre defined connectivity data schema.

The repository is organized as follows:

* `FastAPI`: library used for the users send a request with the payload data to the API implementation.
* `Celery`: library for queuing and async execution. The tasks will be executed asynchronously by the worker in Celery available. The user doen't have to wait for the task finished its process.
* `Redis`: backEnd library responsible to store the results performed with the taskID.

Additional guidance files or procedures can be find to explain how to set up each component of the project.

### Who do I talk to? ###

* Jobzi support team: [support@jobzi.com](mailto:support@jobzi.com)

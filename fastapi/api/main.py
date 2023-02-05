from fastapi import FastAPI
from fastapi.responses import JSONResponse
from celery.result import AsyncResult

from worker.task import predict
from .model import ModelInput, TaskTicket, ModelPrediction

app = FastAPI()

@app.post("/connectivity/prediction", 
          response_model=TaskTicket, 
          status_code=202)
async def schedule_taskPrediction(model_input: ModelInput):
    task_id = predict.delay(dict(model_input).get("x"))
    return {
            "task_id": str(task_id), 
            "status": "Processing"
           }


@app.get("/connectivity/result/{task_id}", 
         response_model=ModelPrediction, 
         status_code=200,
         responses={202: {"model": TaskTicket, "description": "Accepted: Not Ready"}})
async def get_prediction_result(task_id):
    task = AsyncResult(task_id)
    if not task.ready():
        print(app.url_path_for("schedule_taskPrediction"))
        return JSONResponse(
                            status_code=202, content={
                                                      "task_id": str(task_id), 
                                                      "status": "Processing"}
                            )
    result = task.get()
    return {"task_id": task_id, "status": "Success", "result": str(result)}
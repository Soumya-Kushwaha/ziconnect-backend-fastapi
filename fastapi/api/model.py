from pydantic import BaseModel

class TaskTicket(BaseModel):
    task_id: str
    status: str

# X

class ModelInput(BaseModel):
    x: float

# Y

class ModelPrediction(BaseModel):
    #Final result
    task_id: str
    status: str
    result: float
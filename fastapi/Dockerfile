FROM python:3.9-slim-bullseye AS builder

WORKDIR /fastapi
COPY . .

RUN apt update -y && apt upgrade -y && apt install curl -y
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
RUN $HOME/.poetry/bin/poetry config virtualenvs.create false 
RUN $HOME/.poetry/bin/poetry install --no-dev 
RUN $HOME/.poetry/bin/poetry export -f requirements.txt >> requirements.txt

# Prod image
FROM python:3.9-slim-bullseye AS runtime

WORKDIR /fastapi
COPY --from=builder /fastapi/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
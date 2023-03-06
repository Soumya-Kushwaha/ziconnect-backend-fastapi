# Oficial image
FROM python:3.9.5-slim-buster

WORKDIR /usr/src/app/
RUN chmod 777 /usr/src/app/

# Environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Dependencies installation
RUN pip3 install --upgrade pip
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

# copy project
COPY . .

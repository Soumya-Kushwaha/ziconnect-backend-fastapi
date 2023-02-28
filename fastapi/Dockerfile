# Oficial image
FROM python:3.9.5-slim-buster

# Creation and Directory base
#RUN mkdir -p /usr/src/app/files
WORKDIR /usr/src/app

#Permission
RUN chown -R 8004:8004 /usr/src/app
RUN chmod 777 /usr/src/app

#RUN chown -R root:root /usr/src/app/files
#RUN chmod 777 /usr/src/app/files

# Environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Dependencies installation
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . .

FROM python:3.12.6

WORKDIR /opt/mount/

COPY requirements.txt requirements.txt

RUN pip install  --no-cache-dir  -r requirements.txt

COPY . /opt/mount/
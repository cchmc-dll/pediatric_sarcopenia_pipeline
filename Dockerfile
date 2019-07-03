#FROM python:3.6.4
#FROM continuumio/miniconda3

FROM tensorflow/tensorflow:1.13.1-gpu-py3

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt
COPY . /opt/app

#FROM continuumio/miniconda3
#COPY environment_linux.yml /opt/app/
#
#WORKDIR /opt/app
##RUN ["conda", "env", "create", "-f", "environment_linux.yml"]
##RUN ["conda activate tensorflow"]
#
COPY . /opt/app
CMD [ "python", "./run_preprocess.py" ]
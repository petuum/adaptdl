FROM nvcr.io/nvidia/pytorch:20.10-py3
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/yolov3/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY adaptdl adaptdl
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/yolov3 yolov3
ENV PYTHONPATH=/root/yolov3:$PYTHONPATH
WORKDIR /root/yolov3

ENV PYTHONUNBUFFERED=true

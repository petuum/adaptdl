FROM nvcr.io/nvidia/pytorch:20.10-py3
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/imagenet/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY adaptdl adaptdl
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/imagenet imagenet
ENV PYTHONPATH=/root/imagenet:$PYTHONPATH
WORKDIR /root/imagenet

ENV PYTHONUNBUFFERED=true

FROM nvcr.io/nvidia/pytorch:20.03-py3
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/cifar10/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY adaptdl adaptdl
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/ncf ncf
ENV PYTHONPATH=/root/ncf:$PYTHONPATH
WORKDIR /root/ncf

ENV PYTHONUNBUFFERED=true

FROM nvcr.io/nvidia/pytorch:20.10-py3
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/bert/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY adaptdl adaptdl
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/bert bert
ENV PYTHONPATH=/root/bert:$PYTHONPATH
WORKDIR /root/bert

ENV PYTHONUNBUFFERED=true

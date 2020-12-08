FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /root
copy examples/BERT/requirements.txt /root
RUN python3 -m pip install -r /root/requirements.txt

COPY examples/BERT/* /root/
COPY adaptdl adaptdl

ARG ADAPTDL_VERSION=0.0.0
RUN cd adaptdl && python3 setup.py bdist_wheel
RUN ADAPTDL_VERSION=${ADAPTDL_VERSION} pip install adaptdl/dist/*.whl

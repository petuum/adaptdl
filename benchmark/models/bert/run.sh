#!/usr/bin/env bash
export ADAPTDL_MASTER_ADDR=10.117.1.18
export ADAPTDL_MASTER_PORT=12345
export ADAPTDL_NUM_REPLICAS=$1
export ADAPTDL_REPLICA_RANK=$2
#export TARGET_BATCH_SIZE=384
export SQUAD_DIR=./data

python3 run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --per_gpu_train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./logs/debug_squad_$1/ \
    --logging_steps 1

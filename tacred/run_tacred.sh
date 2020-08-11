#!/bin/bash
# This script is for running our GLUE approach.
# Usage: bash run_glue.sh ${TASK} ${GLUEDATA} ${SEED} ${GPU_ID}
#   - TASK: task name list delimited by ",". Defaults to all.
#   - GLUEDATA: GLUE data directory. Defaults to "data".
#   - LOGPATH: log directory. Defaults to "logs".
#   - SEED: random seed. Defaults to 111.
#   - GPU_ID: GPU to use, or -1 for CPU. Defaults to 0.

TASK=${1:-TACRED}
DATA=${2:-tacred}
VOCAB=${3:-vocab}
LOGPATH=${4:-logs}
SEED=${5:-1}

python run.py \
  --seed ${SEED} \
  --log_path ${LOGPATH} \
  --id 00 \
  --emb_dim 300 \
  --ner_dim 30 \
  --pos_dim 30 \
  --hidden_dim 200 \
  --num_layers 2 \
  --attn_dim 200 \
  --pe_dim 30 \
  --n_epoch 30 \
  --batch_size 50 \
  --log_step 20 \
  --save_epoch 5

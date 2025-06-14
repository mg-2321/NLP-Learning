#!/bin/bash
# This script is used to run the hw6 code for the hw.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/dropbox/24-25/574/env
# Run the test script
pytest test_all.py
#Vanilla RNN, default parameters
python run_sst.py
# Vanilla RNN with L2 and dropout
python run_sst.py --l2 1e-4 --dropout 0.5
#LSTM, default parameters
python run_sst.py --lstm
#LSTM with L2 and dropout
python run_sst.py --lstm --l2 1e-4 --dropout 0.5
#Default parameters
python run_lm.py
#New parameters
python run_lm.py \
  --embedding_dim 120 \
  --hidden_dim 100 \
  --num_epochs 30 \
  --lr 0.0008 \
  --dropout 0.3 \
  --l2 0.001 \
  --temp 1.8


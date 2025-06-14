#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/dropbox/24-25/574/env

cd ~/my_hw4
# Comands to run the script with different parameters
python run.py
printf "Running with default parameters...\n"
python run.py --l2 1e-5
printf "Running with L2 regularization...\n"
python run.py --l2 1e-5 --word_dropout 0.3
printf "Running with L2 regularization and word dropout...\n"
#!/bin/bash

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/dropbox/24-25/574/env

# Run the model with default settings
python run.py

# Run the model with modified hyperparameter (Q2 in §3)
python run.py --hidden_size 150
python run.py --embedding_size 100
python run.py --num_prev_chars 8

# run tests 
pytest test_all.py

#!/bin/sh

# (1) if you are using miniconda: 
# source ~/miniconda3/etc/profile.d/conda.sh
# if you install miniconda in a different directory, try the following command
# source path_to_anaconda3/miniconda3/etc/profile.d/conda.sh
# if you install the full anaconda package instead of just miniconda, try:
# source ~/anaconda3/etc/profile.d/conda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/dropbox/24-25/574/env
# (2)conda activate /dropbox/23-24/574/env/

# include your commands here, invoking python with /mnt/dropbox/24-25/env/bin/python
python main.py --text_file /mnt/dropbox/24-25/574/data/sst/train-reviews.txt --output_file train_vocab_base.txt
python main.py --text_file /mnt/dropbox/24-25/574/data/sst/train-reviews.txt --output_file train_vocab_freq5.txt --min_freq 5


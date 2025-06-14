# Train and save vectors
#!/bin/bash

python word2vec.py \
    --training_data /mnt/dropbox/24-25/574/data/sst/train-reviews.txt \
    --num_epochs 6 \
    --embedding_dim 15 \
    --learning_rate 0.2 \
    --min_freq 5 \
    --num_negatives 15 \
    --save_vectors vectors.txt


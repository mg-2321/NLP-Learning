#To acticate the conda environment and run the script, use the following commands:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/dropbox/24-25/574/env
# To run the Running the Translation Model with full parameters , use the following command:
python run.py \
  --train_source /mnt/dropbox/24-25/574/data/europarl-v7-es-en/train.en.txt \
  --train_target /mnt/dropbox/24-25/574/data/europarl-v7-es-en/train.es.txt \
  --output_file test.en.txt.es \
  --num_epochs 8 \
  --embedding_dim 16 \
  --hidden_dim 64 \
  --num_layers 2 \
  --generate_every 1

#To evaluate translations (Character F-score), using the following command:
python chrF++.py -nw 0 \
  -R /mnt/dropbox/24-25/574/data/europarl-v7-es-en/test.es.txt \
  -H test.en.txt.es > test.en.txt.es.score

#To run the test, use the following command:
pytest test_all.py


#!/bin/bash

if [ ! -f v0.9.2.zip ]; then
  wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
fi
if [ ! -d fastText-0.9.2 ]; then
  unzip v0.9.2.zip
  cd fastText-0.9.2
  make
  cd ..
fi

mkdir -p output
for SETTING in {,no}variants.prune_{more,less}; do
  for dim in {16,32,64,128,256}; do
    ./fastText-0.9.2/fasttext cbow -input ../data/$SETTING.txt -output output/fasttext.cbow.$SETTING.$dim \
      -lr 0.025 -dim $dim -epoch 25000 -minCount 0 -minn 0 -maxn 0 -ws 15
    ./fastText-0.9.2/fasttext skipgram -input ../data/$SETTING.txt -output output/fasttext.skip.$SETTING.$dim \
      -lr 0.025 -dim $dim -epoch 25000 -minCount 0 -minn 0 -maxn 0 -ws 15
  done
done

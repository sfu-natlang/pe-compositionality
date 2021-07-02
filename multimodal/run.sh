#!/bin/bash

mkdir -p output

for SETTING in {,no}variants.prune_{more,less}; do
  python3 src/main.py --cuda --epochs 100 --batch_size 100 --nhid 64 --emsize 64  --data ../data/$SETTING.txt --images ./data/images/pe_64 --train --export --textual --save output/lm.text.64.$SETTING.pt --bidirectional
  python3 src/main.py --cuda --epochs 130 --batch_size 100 --nhid 64 --emsize 64  --data ../data/$SETTING.txt --images ./data/images/pe_64 --train --export --visual --imgdim 64 --save output/lm.image.64.$SETTING.pt --bidirectional
  python3 src/main.py --cuda --epochs 130 --batch_size 100 --nhid 64 --emsize 64  --data ../data/$SETTING.txt --images ./data/images/pe_64 --train --export --textual --visual --imgdim 64 --save output/lm.image+text.64.$SETTING.pt --bidirectional
  python3 src/main.py --cuda --epochs 130 --batch_size 100 --nhid 64 --emsize 128 --data ../data/$SETTING.txt --images ./data/images/pe_64 --train --export --textual --visual --imgdim 64 --save output/lm.image+text.128.$SETTING.pt --bidirectional
done

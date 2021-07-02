#!/bin/bash

mkdir -p output

git clone https://github.com/stanfordnlp/glove
cd glove && make

for SETTING in {,no}variants.prune_{more,less}; do
  for VECTOR_SIZE in {16,32,64,128,256}; do
    CORPUS=../../data/$SETTING.txt
    VOCAB_FILE=vocab.txt
    COOCCURRENCE_FILE=cooccurrence.bin
    COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
    BUILDDIR=build
    SAVE_FILE=../output/glove.$SETTING.$VECTOR_SIZE
    VERBOSE=0
    MEMORY=4.0
    VOCAB_MIN_COUNT=0
    # There is no test set so we don't care about 
    # overfitting: just want to get representations 
    # which are good for this data in particular:
    MAX_ITER=3000
    WINDOW_SIZE=15
    BINARY=2
    NUM_THREADS=8
    X_MAX=10
    if hash python 2>/dev/null; then
      PYTHON=python
    else
      PYTHON=python3
    fi

    $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
    $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE 2&> /dev/null
  done
done

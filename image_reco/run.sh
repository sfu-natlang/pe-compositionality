#!/bin/bash

mkdir -p output

python pe_mnist.py
python get_embeddings_from_trained_model.py

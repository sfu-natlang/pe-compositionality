from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import argparse

import load_images
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np

import pe_mnist

parser = argparse.ArgumentParser(description='')
parser.add_argument('--sign_name', type=str, default=None,
                    help='the sign embedding to output. if not specified, all sign embeddings will be dumped')
parser.add_argument('--variants', action='store_true',
                    help='if true, treat sign variants as distinct classes')
parser.add_argument('--model', default='trained.h5',
                    help='saved model to get embeddings from')
args = parser.parse_args()

model = pe_mnist.model
model.load_weights(args.model)

inp = model.input
outputs = [layer.output for layer in model.layers]
functor = K.function([inp], outputs)

if args.variants:
  num_classes = 1506
else:
  num_classes = 432

outfile = open("output/embeddings.txt", "w")
if args.sign_name is None:
    batchno = 0
    data_generator = load_images.get_traingen()
    sign2i = data_generator.class_indices
    i2sign = {sign2i[sign]:sign for sign in sign2i}
    for xs,ys in data_generator:
        for i, x in enumerate(xs):
            x=x.reshape( (1, 64, 64,1) )
            label = np.argmax(ys[i])
            hid = map(str,functor([x])[3].reshape( (-1,) ))
            outfile.write(str(i2sign[int(label)]) + " " + (" ".join(hid)) + "\n")
        if batchno * load_images.batch_size + i + 1 >= num_classes:
            break
        batchno += 1
        print("Done batch",batchno)
else:
    x = Image.open(args.sign_name)
    x = x.resize( (64, 64) )
    x = np.array(x.getdata())
    x = x.reshape( (1, 64, 64, 1) ) / 255.
    print(f"getting embedding for {args.sign_name}")
    hid  = map(str,functor([x])[3].reshape( (-1,) ))
    outfile.write(args.sign_name + " " + (" ".join(hid)) + "\n")
outfile.close()

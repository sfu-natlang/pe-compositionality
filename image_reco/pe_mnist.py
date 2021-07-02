import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import load_images
train_generator = load_images.get_traingen()

import argparse
parser = argparse.ArgumentParser(description='Image Recognition for Proto-Elamite sign images')
parser.add_argument('--variants', action='store_true',
                    help='if true, treat sign variants as distinct classes')
parser.add_argument('--epochs', default=6000, type=int,
                    help='number of training epochs')
parser.add_argument('--model', default='trained.h5',
                    help='path to save model')
args = parser.parse_args()

if args.variants:
  num_classes = 1506
else:
  num_classes = 432

# Reproduce the image embedding component from the multimodal LM:
model = Sequential()
model.add(Conv2D(96, kernel_size=(10, 10), strides=(2,2),
                 activation='relu',
                 input_shape=( 64, 64, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

if __name__ == '__main__':
    model.fit(train_generator,
        steps_per_epoch = num_classes // load_images.batch_size,
        epochs=args.epochs,
    )

    model.save_weights(args.model)

    # There is only one image for each sign, thus there is no separate test_generator
    # Later work might consider using tablet lineart to get multiple images per sign,
    # but this will require image segmentation which will have to account for breaks in
    # tablets and other damage.
    score = model.evaluate_generator(train_generator)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

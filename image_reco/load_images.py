import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

batch_size = 128
def get_traingen():
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
    )
    train_generator = train_datagen.flow_from_directory(
        "data/train/",
        target_size = (64, 64),
        color_mode = "grayscale",
        batch_size = 128,
        class_mode = 'categorical'
        )
    return train_generator

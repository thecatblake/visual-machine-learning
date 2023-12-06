import tensorflow as tf
import numpy as np
import keras
from keras import layers

BATCH_SIZE = 128
NB_CLASSES = 10
N_HIDDEN = 128
RESHAPED = 784

GEN_DIM = 100

def create_model_simple():
    model = keras.models.Sequential([
        keras.layers.Dense(NB_CLASSES,
                           input_shape=(RESHAPED,),
                           name="dense_layer",
                           activation="softmax"),
    ])
    return model

def create_model():
    input_shape = (28, 28, 1)

    # Create the MobileNet-V2 model
    model = keras.Sequential()

    # First Convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())

    # Depthwise Separable Convolution layers
    model.add(layers.DepthwiseConv2D((3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.DepthwiseConv2D((3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.DepthwiseConv2D((3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.DepthwiseConv2D((3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (1, 1), activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.DepthwiseConv2D((3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (1, 1), activation='relu'))
    model.add(layers.BatchNormalization())

    # Global Average Pooling and Output layer
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(10, activation='softmax'))

    return model


def create_gen_model():
    model = keras.models.Sequential([
        keras.layers.Dense(RESHAPED, input_shape=(GEN_DIM,), activation="relu")
    ])

    return model

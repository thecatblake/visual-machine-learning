import tensorflow as tf
import numpy as np
import keras
from model import create_model, create_model_simple
import matplotlib.pyplot as plt


EPOCHS = 1
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

mnist = keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.astype('float32').reshape((X_train.shape[0], -1,))
X_test = X_test.astype('float32').reshape((X_test.shape[0], -1,))

X_train /= 255
X_test /= 255

Y_train = keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = keras.utils.to_categorical(Y_test, NB_CLASSES)

model = create_model_simple()

model.compile(optimizer="SGD", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

test_loss, test_acc = model.evaluate(X_test, Y_test)

print(vars(model.layers[0].weights[0]))

plt.plot(range(1, EPOCHS+1), history.history["loss"])
plt.show()

model.save("model.h5")

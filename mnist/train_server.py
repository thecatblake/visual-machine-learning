import tensorflow as tf
import numpy as np
from tensorflow import keras
from fastapi import FastAPI
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import json

origins = [
    "*"
]

middleware = [
    Middleware(CORSMiddleware, allow_origins=origins)
]

app = FastAPI(middleware=middleware)

EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

mnist = keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

RESHAPED = 784
X_train = X_train.reshape(-1, RESHAPED)
X_test = X_test.reshape(-1, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = keras.utils.to_categorical(Y_test, NB_CLASSES)

optimizer = keras.optimizers.SGD(learning_rate=1e-3)

loss_fn = keras.losses.CategoricalCrossentropy()

model = keras.models.Sequential([
    keras.layers.Dense(NB_CLASSES,
                       input_shape=(RESHAPED,),
                       name="dense_layer",
                       activation="softmax")
])

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

train_data_iter = enumerate(train_dataset)


@app.get("/")
async def root():
    try:
        i, data = next(train_data_iter)
    except StopIteration:
        return {"msg": "end"}
    x, y = data

    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(y, logits)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    i = np.random.randint(x.shape[0], size=1)

    return {
        "X": json.dumps(x.numpy()[i, :][0].tolist()),
        "Y": json.dumps(y.numpy()[i, :][0].tolist()),
        "W": json.dumps(model.get_weights()[0].flatten().tolist()),
        "output": json.dumps(logits.numpy()[i, :][0].tolist())
    }


@app.get("/next")
async def update():
    global train_data_iter
    train_data_iter = enumerate(train_dataset)

    return {"msg": "ok"}

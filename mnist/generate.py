import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from model import create_gen_model, GEN_DIM

trained_model = keras.models.load_model('model.h5')
model = create_gen_model()

EPOCHS = 10000

optimizer = keras.optimizers.SGD(learning_rate=1e-3)

loss_fn = keras.losses.CategoricalCrossentropy()

BATCH_SIZE = 200

y = np.zeros((BATCH_SIZE, 10))
y[:, 1] = 1


for epoch in range(EPOCHS):
    r = np.random.rand(BATCH_SIZE, GEN_DIM)
    sample = tf.Variable(tf.convert_to_tensor(r))
    with tf.GradientTape() as tape:
        x = model(sample)
        pred = trained_model(x)
        loss = loss_fn(y, pred)

    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))

    if epoch % 1000 == 0:
        print(loss)
        img = model(sample).numpy()[0].reshape((28, 28))
        plt.imshow(img, cmap='gray')
        plt.show()

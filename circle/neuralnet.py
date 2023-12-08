import matplotlib
import matplotlib.cm as cmx
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation

x,y = np.meshgrid(np.linspace(-5,5,500),np.linspace(-5,5,500))

X = tf.convert_to_tensor(np.column_stack((x.flatten(),y.flatten())))
Y = tf.reduce_sum(tf.square(X), axis=1)
Y = tf.logical_or(tf.logical_and(Y < 2, Y > 1), tf.logical_and(Y < 4, Y > 3))
Y = tf.cast(Y, tf.float32)

model = keras.Sequential([
    keras.layers.Dense(10000, input_shape=(2,), activation='relu'),
    keras.layers.Dense(1)
])

model.compile("adam", "MSE")

EPOCHS = 10
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam()

artists = []
filenames = []

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)
test_x, test_y = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
points = np.column_stack((test_x.flatten(), test_y.flatten()))
predictions = model.predict(points).reshape(-1)

cm = plt.get_cmap("jet")
cNorm = matplotlib.colors.Normalize(vmin=min(predictions), vmax=max(predictions))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
ax.scatter(test_x, test_y, predictions, c=scalarMap.to_rgba(predictions), s=1)

plt.show()
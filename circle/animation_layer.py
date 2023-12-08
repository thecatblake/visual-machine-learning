import matplotlib
import matplotlib.cm as cmx
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation

x,y = np.meshgrid(np.linspace(-10,10,500),np.linspace(-10,10,500))
X = tf.convert_to_tensor(np.column_stack((x.flatten(),y.flatten())))

Y1 = [tf.reduce_sum(tf.square(X+(i, 0)), axis=1) for i in np.linspace(-10, 10, 5)]
Y = Y1[0] < 1
for i in range(1, len(Y1)):
    Y = tf.logical_or(Y, Y1[i] < 1)

Y = tf.cast(Y, tf.float32)

EPOCHS = 50
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam()

artists = []
filenames = []

fig, ax = plt.subplots()

for layer_num in range(100, 5100, 500):
    model = keras.Sequential([
        keras.layers.Dense(layer_num, input_shape=(2,), activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile("adam", "MSE")
    model.fit(X, Y, epochs=10, batch_size=256, verbose=1)
    test_x, test_y = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
    points = np.column_stack((test_x.flatten(), test_y.flatten()))
    predictions = model.predict(points).reshape(-1)
    cm = plt.get_cmap("jet")
    cNorm = matplotlib.colors.Normalize(vmin=min(predictions), vmax=max(predictions))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    g = ax.scatter(test_x, test_y,  c=scalarMap.to_rgba(predictions))
    artists.append([g])

ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=500)
ani.save(filename="layer_100.gif", writer="pillow")

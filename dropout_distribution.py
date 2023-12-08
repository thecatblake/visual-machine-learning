import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from statsmodels.graphics.api import qqplot

N = 1000
rate = 0.4

hist = []

num = 10000
artists = []

fig, ax = plt.subplots()

for j in range(10, 250, 5):
    hist = []
    for i in range(num):
        x = np.random.uniform(size=(j,))
        x = x > 0.95
        hist.append(np.sum(x))

    hist = np.array(hist)
    #qqplot(hist, line="s", ax=ax[1])
    ax.set_ylim((0, 3000))
    n, bins, g = ax.hist(hist, bins=100, color="blue")
    artists.append(g)

ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=150)
ani.save(filename="dropout_distribution.gif", writer="pillow")

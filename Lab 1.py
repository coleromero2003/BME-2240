import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
xx, yy = np.meshgrid(x,y)

g = xx
h = yy

# yp = np.sqrt(x*x)

plt.plot(x)
plt.quiver(xx, yy, h, g)

plt.show()
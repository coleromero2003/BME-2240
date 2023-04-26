import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 17)
y = np.linspace(-2, 2, 17)
xx, yy = np.meshgrid(x,y)

g = yy+xx
h = 1
e=np.e
yp = (-x -1 + (e*e**(x)/3))

plt.plot(x,yp)
plt.quiver(xx, yy, h, g)
plt.ylim(-2,2)
plt.show()
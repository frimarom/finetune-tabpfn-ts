import numpy as np, matplotlib.pyplot as plt
x = np.arange(-1.3, 1.3, .01)
plt.plot(x, x**2)
plt.savefig("bilder/pyplot1.png")

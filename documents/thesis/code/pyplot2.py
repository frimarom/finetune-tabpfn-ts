import numpy as np, matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Linux Libertine O", "sans-serif"]
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (4,3) # inches
x = np.arange(-1.3, 1.3, .01)
plt.plot(x, x**2)
plt.tight_layout()
plt.savefig("bilder/pyplot2.pdf")

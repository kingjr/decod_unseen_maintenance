import numpy as np
import matplotlib.pyplot as plt


bottom = 8
max_height = 4

hist, bin_edges = np.histogram(np.array(predAngle[20,20,:]),bins=80)
N = len(hist)
radii = ones(theta.shape)
width = (2*np.pi) / N

ax = plt.subplot(111, polar=True)
bars = ax.bar(bin_edges[0:N], hist, width=width, bottom=bottom)


xT=plt.xticks()[0]
xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
   r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
plt.xticks(xT, xL)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(0.8)

plt.show()

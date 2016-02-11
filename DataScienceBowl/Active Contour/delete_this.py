import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab as pl
import skfmm

circ = np.zeros((50, 50))

for i, j in np.ndindex(np.shape(circ)):
    dist = np.array([i, j]) - np.array([25, 25])
    norm_dist = np.linalg.norm(dist)
    if norm_dist < 20:
        circ[i, j] = 1


def get_contour(s):
    bound = np.zeros(np.shape(s))
    for k, l in np.ndindex(np.shape(s)):
        if 0 < k < (np.shape(s)[0]-1) and 0 < l < (np.shape(s)[1]-1):
            if s[k, l] == 1 and \
                    (s[k-1, l] == 0 or s[k+1, l] == 0 or s[k, l-1] == 0 or s[k, l+1] == 0):
                bound[k, l] = 1
    return bound

circ[circ == 1] = -1
circ[circ == 0] = 1
plt.imshow(circ)
plt.show()
# ring = get_contour(circ)
# plt.imshow(ring)
# plt.show()
# ring[ring == 1] = -1
# ring[ring == 0] = 1
D = -skfmm.distance(circ, dx=1e-2)
plt.imshow(D)
plt.show()


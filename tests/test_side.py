"""
Tests the side detection between a point and a vector
"""
import numpy as np
import matplotlib.pyplot as plt
from policy.utils.overlap_detection import Tangent


point = (1, -1)
normal1 = np.array([1., 1.])
normal1 /= np.linalg.norm(normal1)
normal1 = tuple(normal1)
tangent1 = Tangent((1.,0.), normal1)

fig, ax = plt.subplots(figsize=(9,6), layout='tight')
ax.set_aspect('equal')
tangent1.plot(ax, length=5)
ax.scatter(point[0], point[1], s=25, c='red')
print(tangent1.side(point))

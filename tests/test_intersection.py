"""
Tests the intersection point detection between two lines
"""

import numpy as np
import matplotlib.pyplot as plt
from policy.utils.overlap_detection import Tangent


fig, ax = plt.subplots(figsize=(9,6), layout='tight')
ax.set_aspect('equal')

normal1 = np.array([1., 1.])
normal1 /= np.linalg.norm(normal1)
normal1 = tuple(normal1)
tangent1 = Tangent((0.,0.), normal1)

normal2 = np.array([-1., 1.])
normal2 /= np.linalg.norm(normal2)
normal2 = tuple(normal2)
tangent2 = Tangent((0, 1.), normal2)

inter_point = tangent1.intersect(tangent2)

tangent1.plot(ax, length=10)
tangent2.plot(ax, length=10)
ax.scatter(inter_point[0], inter_point[1], s=25, c='blue')

print(inter_point)

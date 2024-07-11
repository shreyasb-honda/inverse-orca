"""
Tests the overlap detection algorithm
"""

import os
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from policy.utils.overlap_detection import Circle, VelocityObstacle, Tangent, Point
sns.set_theme(context='talk', style='dark')


def plot_and_test(tau: float, relative_position: Point, radius_a: float, radius_b: float,
                  vb_max: float, v_a: Point):
    """
    Plots the velocity obstacle and the velocity circle.
    Determines and outputs the overlap between them
    """

    fig, ax = plt.subplots(layout='tight', figsize=(9, 9))
    ax.set_aspect('equal')
    ax.axhline(color='black')
    ax.axvline(color='black')
    cutoff_center = tuple(np.array(relative_position) / tau)
    cutoff_radius = (radius_a + radius_b) / tau
    cutoff_circle = Circle(cutoff_center, cutoff_radius)
    velocity_obstacle = VelocityObstacle(cutoff_circle)
    ax = velocity_obstacle.plot(ax)
    velocity_circle = Circle(v_a, vb_max)
    ax = velocity_circle.plot(ax)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

    return ax, velocity_obstacle.overlap(velocity_circle)


def first_quadrant():
    """
    Performs the test in the first quadrant
    """
    # Constants
    tau = 2.0
    relative_position = (1., 1.)
    radius_a = 0.3
    radius_b = 0.3
    vb_max = 1.0

    home = expanduser('~')
    out_directory = os.path.join(home, 'OneDrive', 'Documents', 'Notes', 'Plots', 'overlap-testing')
    # Intersection with the cutoff circle
    v_a = (-0.3, -0.3)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '1.png'))

    # Intersection with the left leg
    v_a = (0.3, 2.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '2.png'))

    # Intersection with the right leg
    v_a = (2.0, 0.3)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '3.png'))

    # No overlap (below right leg)
    v_a = (2.0, -0.8)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '4.png'))

    # No overlap (above left leg)
    v_a = (-0.3, 3.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '5.png'))


    # No overlap (in third quadrant)
    v_a = (-0.5, -0.5)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '6.png'))


    # No overlap - intersects tangent line on the wrong side
    v_a = (-1.0, 0.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '7.png'))

    plt.show()


def second_quadrant():
    """
    Performs the test in the second quadrant
    """

    # Constants
    tau = 2.0
    radius_a = 0.3
    radius_b = 0.3
    vb_max = 1.0
    relative_position = (-1., 1.)

    home = expanduser('~')
    out_directory = os.path.join(home, 'OneDrive', 'Documents', 'Notes', 'Plots', 'overlap-testing')

    # Intersection with the cutoff circle
    v_a = (0.3, -0.3)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '1.png'))

    # Intersection with the left leg
    v_a = (-3.0, 0.7)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '2.png'))

    # Intersection with the right leg
    v_a = (-0.2, 2.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '3.png'))

    # No overlap (above right leg)
    v_a = (0.2, 3.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '4.png'))

    # No overlap (below left leg)
    v_a = (-3.0, -0.5)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '5.png'))

    # No overlap (in fourth quadrant)
    v_a = (0.5, -0.5)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '6.png'))

    plt.show()


def third_quadrant():
    """
    Performs the test in the third quadrant
    """

    # Constants
    tau = 2.0
    relative_position = (-1., -1.)
    radius_a = 0.3
    radius_b = 0.3
    vb_max = 1.0

    home = expanduser('~')
    out_directory = os.path.join(home, 'OneDrive', 'Documents', 'Notes', 'Plots', 'overlap-testing')

    # Intersection with the cutoff circle
    v_a = (0.3, 0.3)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '1.png'))

    # Intersection with the left leg
    v_a = (-0.8, -3.6)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '2.png'))

    # Intersection with the right leg
    v_a = (-1.2, -0.2)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '3.png'))

    # No overlap (above right leg)
    v_a = (-3.5, 0.4)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '4.png'))

    # No overlap (below left leg)
    v_a = (-0.1, -4.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '5.png'))

    # No overlap (in first quadrant)
    v_a = (0.5, 1.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '6.png'))

    plt.show()


def fourth_quadrant():
    """
    Performs the test in the fourth quadrant
    """

    # Constants
    tau = 2.0
    relative_position = (1., -1.)
    radius_a = 0.3
    radius_b = 0.3
    vb_max = 1.0

    home = expanduser('~')
    out_directory = os.path.join(home, 'OneDrive', 'Documents', 'Notes', 'Plots', 'overlap-testing')

    # Intersection with the cutoff circle
    v_a = (0.3, -0.3)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '1.png'))

    # Intersection with the left leg
    v_a = (3.0, -0.7)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '2.png'))

    # Intersection with the right leg
    v_a = (1.0, -2.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '3.png'))

    # No overlap (below right leg)
    v_a = (0.0, -3.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '4.png'))

    # No overlap (above left leg)
    v_a = (3.0, 1.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '5.png'))

    # No overlap (in second quadrant)
    v_a = (-0.5, 0.5)
    ax, overlap = plot_and_test(tau, relative_position, radius_a, radius_b, vb_max, v_a)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '6.png'))

    plt.show()


def line_circle_overlap():
    """
    Tests the overlap detection between a line
    and a circle
    """
    circ = Circle((1.,1.), 1.)

    normal = np.array([-1., 1.])
    normal /= np.linalg.norm(normal)
    point = np.array(circ.center) + 0.5 * normal + np.array([1., 1.])
    point = tuple(point)

    line = Tangent(point, normal)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    circ.plot(ax)
    line.plot(ax)

    print(circ.line_overlap(line))

    plt.show()

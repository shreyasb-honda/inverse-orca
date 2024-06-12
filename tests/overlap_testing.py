import os
from os.path import expanduser
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from policy.utils.OverlapDetection import Circle, VelocityObstacle
sns.set_theme(context='talk', style='dark')


def plot_and_test(tau: float, relative_position: Tuple[float, float], radius_A: float, radius_B: float, 
                  vB_max: float, v_A: Tuple[float, float]):

    fig, ax = plt.subplots(layout='tight', figsize=(9, 9))
    ax.set_aspect('equal')
    ax.axhline(color='black')
    ax.axvline(color='black')
    cutoff_center = tuple(np.array(relative_position) / tau)
    cutoff_radius = (radius_A + radius_B) / tau
    cutoff_circle = Circle(cutoff_center, cutoff_radius)
    velocity_obstacle = VelocityObstacle(cutoff_circle)
    ax = velocity_obstacle.plot(ax)
    velocity_circle = Circle(v_A, vB_max)
    ax = velocity_circle.plot(ax)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

    return ax, velocity_obstacle.overlap(velocity_circle)


def first_quadrant():

    # Constants
    tau = 2.0
    relative_position = (1., 1.)
    radius_A = 0.3
    radius_B = 0.3
    vB_max = 1.0


    home = expanduser('~')
    out_directory = os.path.join(home, 'OneDrive', 'Documents', 'Notes', 'Plots', 'overlap-testing')
    # Intersection with the cutoff circle
    v_A = (-0.3, -0.3)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '1.png'))

    # Intersection with the left leg
    v_A = (0.3, 2.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '2.png'))

    # Intersection with the right leg
    v_A = (2.0, 0.3)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '3.png'))

    # No overlap (below right leg)
    v_A = (2.0, -0.8)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '4.png'))

    # No overlap (above left leg)
    v_A = (-0.3, 3.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '5.png'))


    # No overlap (in third quadrant)
    v_A = (-0.5, -0.5)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '6.png'))


    # No overlap - intersects tangent line on the wrong side
    v_A = (-1.0, 0.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '7.png'))

    plt.show()

def second_quadrant():
    # Constants
    tau = 2.0
    relative_position = (-1., 1.)
    radius_A = 0.3
    radius_B = 0.3
    vB_max = 1.0

    home = expanduser('~')
    out_directory = os.path.join(home, 'OneDrive', 'Documents', 'Notes', 'Plots', 'overlap-testing')

    # Intersection with the cutoff circle
    v_A = (0.3, -0.3)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '1.png'))

    # Intersection with the left leg
    v_A = (-3.0, 0.7)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '2.png'))

    # Intersection with the right leg
    v_A = (-0.2, 2.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '3.png'))

    # No overlap (above right leg)
    v_A = (0.2, 3.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '4.png'))

    # No overlap (below left leg)
    v_A = (-3.0, -0.5)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '5.png'))

    # No overlap (in fourth quadrant)
    v_A = (0.5, -0.5)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '6.png'))

    plt.show()

def third_quadrant():
    # Constants
    tau = 2.0
    relative_position = (-1., -1.)
    radius_A = 0.3
    radius_B = 0.3
    vB_max = 1.0

    home = expanduser('~')
    out_directory = os.path.join(home, 'OneDrive', 'Documents', 'Notes', 'Plots', 'overlap-testing')

    # Intersection with the cutoff circle
    v_A = (0.3, 0.3)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '1.png'))

    # Intersection with the left leg
    v_A = (-0.8, -3.6)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '2.png'))

    # Intersection with the right leg
    v_A = (-1.2, -0.2)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '3.png'))

    # No overlap (above right leg)
    v_A = (-3.5, 0.4)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '4.png'))

    # No overlap (below left leg)
    v_A = (-0.1, -4.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '5.png'))

    # No overlap (in first quadrant)
    v_A = (0.5, 1.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '6.png'))

    plt.show()

def fourth_quadrant():
    # Constants
    tau = 2.0
    relative_position = (1., -1.)
    radius_A = 0.3
    radius_B = 0.3
    vB_max = 1.0

    home = expanduser('~')
    out_directory = os.path.join(home, 'OneDrive', 'Documents', 'Notes', 'Plots', 'overlap-testing')

    # Intersection with the cutoff circle
    v_A = (0.3, -0.3)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '1.png'))

    # Intersection with the left leg
    v_A = (3.0, -0.7)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '2.png'))

    # Intersection with the right leg
    v_A = (1.0, -2.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '3.png'))

    # No overlap (below right leg)
    v_A = (0.0, -3.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '4.png'))

    # No overlap (above left leg)
    v_A = (3.0, 1.0)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '5.png'))

    # No overlap (in second quadrant)
    v_A = (-0.5, 0.5)
    ax, overlap = plot_and_test(tau, relative_position, radius_A, radius_B, vB_max, v_A)
    ax.text(0.01, 0.99, f"Overlap {overlap}", ha='left', va='top', transform=ax.transAxes)
    print(f"Overlap: {overlap}")
    plt.savefig(os.path.join(out_directory, '6.png'))

    plt.show()

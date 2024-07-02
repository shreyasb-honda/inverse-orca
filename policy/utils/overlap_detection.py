"""
Classes to help build the velocity obstacle and determine whether the possible
relative velocities circle overlaps the velocity obstacle
"""

from typing import Tuple
from math import sin, cos
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


Point = Tuple[float, float]

class Tangent:
    """
    A class representing a tangent line, with the point of contact and the normal direction
    """

    def __init__(self, point: Point, normal: Point):
        """
        :param point - the point of contact of the tangent
        :param normal - the unit normal to the tangent line
        """
        self.point = point
        self.normal = normal

    def plot(self, ax: plt.Axes, length: float = 3., color: str = 'black'):
        """
        Plots the ray
        :param length - then length of the ray to be plotted
        :param color - the color of the ray
        """
        ax.scatter(self.point[0], self.point[1], color=color, s=9)
        arrow_direction = np.array([-self.normal[1], self.normal[0]])
        arrow_length = arrow_direction * length
        ax.arrow(
            self.point[0], self.point[1], arrow_length[0], arrow_length[1], color=color
        )

        return ax

    def intersect(self, other: "Tangent"):
        """
        Returns the intersection point between this line and other.
        (Considering the rays as lines extending both sides)
        """
        x1, y1 = self.point
        n1x, n1y = self.normal

        x2, y2 = other.point
        n2x, n2y = other.normal

        denom = n1x*n2y - n2x*n1y

        x = (x1*n1x*n2y - x2*n1y*n2x + n1y*n2y*(y1-y2)) / denom
        y = (n2y*n1x*y2 - n1y*n2x*y1 + n1x*n2x*(x2-x1)) / denom

        return x, y

    def dist(self, point: Point):
        """
        Returns the distance of a point from this line
        """
        _point_to_point = np.array(point) - np.array(self.point)
        return abs(np.dot(self.normal, _point_to_point))

    def side(self, point: Point):
        """
        Returns 
            -1 if the point is on the right side of the line
            0 if the point is on the line
            1 if the point is on the left side of the line
        """
        _point_to_point = np.array(point) - np.array(self.point)
        det = self.normal[0] * _point_to_point[1] - self.normal[1] * _point_to_point[0]

        if det < 0:
            return -1
        if det > 0:
            return 1

        return 0


class Circle:
    """
    A class representing a circle. Implements overlap checks with rays and other circles
    """

    def __init__(self, center: Point, radius: float):
        """
        :param center - the center of the circle
        :param radius - the radius of the circle
        """
        self.center = center
        self.radius = radius
        self.circ = None
        self.left_tangent = None
        self.right_tangent = None

    def circle_overlap(self, other: "Circle"):
        """
        Checks for overlap with another circle
        :param other - the other circle
        """
        distance = np.linalg.norm(np.array(self.center) - np.array(other.center))
        if distance < self.radius + other.radius:
            return True
        return False

    def line_overlap(self, line: Tangent):
        """
        Checks for overlap with a line
        :param line - the line with which to evaluate the overlap
        """
        point_to_center = np.array(self.center) - np.array(line.point)
        # If the point is inside the circle, return true
        if norm(point_to_center) < self.radius:
            return True

        # If the point is outside the circle, but
        # the distance of the ray from the center is less than the radius AND
        # the line from the point to the center makes a positive angle with
        # the normal, then return true
        normal = line.normal
        det = normal[0] * point_to_center[1] - normal[1] * point_to_center[0]
        # distance_from_line = point_to_center.dot(np.array(normal))
        distance_from_line = line.dist(self.center)
        if abs(distance_from_line) < self.radius and det > 0:
            return True

        return False

    def plot(self, ax: plt.Axes, fill: bool = False, edgecolor: str = 'red',
             color: str | None = None):
        """
        Plots the circle on the axis
        :param ax - the pyplot axis on which to plot this circle
        :param fill - whether or not to fill the circle (default: False)
        :param edgecolor - the edge color of the circle (default: red)
        :param color - the fill color of the circle (default: None)
        """
        if color is None:
            self.circ = plt.Circle(self.center, self.radius, fill=fill, edgecolor=edgecolor)
        else:
            self.circ = plt.Circle(self.center, self.radius, fill=fill, color=color)        
        # ax.scatter(self.center[0], self.center[1], s=9, color=edgecolor)
        ax.add_patch(self.circ)

        return ax

class VelocityObstacle:
    """
    The velocity obstacle object. Consists of a cutoff circle and two tangents to the circle 
    from the origin
    """

    def __init__(self, cutoff_circle: Circle):
        """
        Initializes the velocity obstacle
        :param cutoff_circle - the circle at which the cone of the velocity obstacle gets truncated
        """
        self.cutoff_circle = cutoff_circle
        self.left_tangent = None
        self.right_tangent = None

        # Angle of the center from the origin
        self.alpha = np.arctan2(self.cutoff_circle.center[1], self.cutoff_circle.center[0])

        # Length of the tangent segment from the origin to the point of contact
        center = self.cutoff_circle.center
        radius = self.cutoff_circle.radius
        self.leg_length = np.sqrt(norm(np.array(center))**2 - radius ** 2)
        self.compute_tangents()

        # Angle from the origin to the right tangent
        self.theta = None

        # Angle from the origin to the left tangent
        self.beta = None

    def compute_tangents(self):
        """
        Computes tangents to the circle from the origin
        """
        radius = self.cutoff_circle.radius
        direction = np.array([self.leg_length * sin(self.alpha) - radius * cos(self.alpha),
                              self.leg_length * cos(self.alpha) + radius * sin(self.alpha)])
        self.theta = np.arctan2(*direction)

        # print(f"Alpha: {np.rad2deg(self.alpha)}")
        # print(f"Theta: {np.rad2deg(self.theta)}")

        right_point = (self.leg_length * cos(self.theta), self.leg_length * sin(self.theta))
        right_normal = (sin(self.theta), -cos(self.theta))
        self.right_tangent = Tangent(right_point, right_normal)

        self.beta = 2 * self.alpha - self.theta
        left_point = (self.leg_length * cos(self.beta), self.leg_length * sin(self.beta))
        left_normal = (sin(self.beta), -cos(self.beta))
        self.left_tangent = Tangent(left_point, left_normal)


    def plot(self, ax: plt.Axes):
        """
        Plots the velocity obstacle
        :param ax - the matplotlib.Axes object used for plotting
        """
        ax = self.cutoff_circle.plot(ax, edgecolor='black')
        ax = self.left_tangent.plot(ax)
        ax = self.right_tangent.plot(ax)

        return ax

    def overlap(self, velocity_circle: Circle):
        """
        Checks the overlap between the possible relative velocities circle and the velocity obstacle
        :param velocity_circle - the circle of possible relative velocities
        """
        if velocity_circle.circle_overlap(self.cutoff_circle):
            return True

        if velocity_circle.line_overlap(self.left_tangent):
            return True

        if velocity_circle.line_overlap(self.right_tangent):
            return True

        return False

    def find_dmax(self, velocity_circle: Circle, which: str):
        """
        Finds the maximum possible distance d of the projection line from the tangent line
        If the projection line is further than this distance, 
        the projection will be switched to the other leg
        
        :param velocity_circle: The relative velocities circle
        :param which: whether dmax should be computed from the right or left leg
        """
        chosen_normal = self.right_tangent.normal
        if which == 'left':
            chosen_normal = self.left_tangent.normal

        va_x, va_y = velocity_circle.center
        vb_max = velocity_circle.radius

        # slope of line from origin to cutoff circle center
        m = self.cutoff_circle.center[1] / self.cutoff_circle.center[0]

        point = self.cutoff_circle.center
        normal = np.array([m, -1.])
        normal /= norm(normal)

        line = Tangent(point, normal)

        if not velocity_circle.line_overlap(line):
            # If there is no overlap between the center line and the velocity circle,
            # dmax is the radius of the velocity circle - distance of center from tangent
            return velocity_circle.radius - abs(np.dot(chosen_normal, velocity_circle.center))

        point = self.cutoff_circle.center
        normal = np.array([m, -1.])
        normal /= norm(normal)

        line = Tangent(point, normal)

        if not velocity_circle.line_overlap(line):
            # If there is no overlap between the center line and the velocity circle,
            # dmax is the radius of the velocity circle - distance of center from tangent
            return velocity_circle.radius - abs(np.dot(chosen_normal, velocity_circle.center))

        # Terms in the quadratic equation ax^2 - 2bx + c to get the point of intersection
        # of y=mx and the velocity circle
        a = 1 + m**2
        b = va_x + m * va_y
        c = va_x**2 + va_y**2 - vb_max**2
        # print(a, b, c)

        x1 = (b + np.sqrt(b**2 - a*c)) / (2*a)
        x2 = (b - np.sqrt(b**2 - a*c)) / (2*a)

        y1 = m * x1
        y2 = m * x2

        point1 = (x1, y1)
        point2 = (x2, y2)

        cross1 = chosen_normal[0] * y1 - chosen_normal[1] * x1
        cross2 = chosen_normal[0] * y2 - chosen_normal[0] * x2

        if cross1 > 0 and cross2 > 0:
            # Both points to the left of the normal at origin
            if norm(point1) > norm(point2):
                return abs(np.dot(chosen_normal, point1))

            return abs(np.dot(chosen_normal, point2))

        if cross1 > 0:
            return abs(np.dot(chosen_normal, point1))

        if cross2 > 0:
            return abs(np.dot(chosen_normal, point2))

        # print("Both intersection points are to the right of the normal. Impossibe case...")
        # print("Both intersection points are to the right of the normal. Impossibe case...")
        # print('Returning dmax = 0')
        return 0

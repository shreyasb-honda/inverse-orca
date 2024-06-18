"""
Steps: 
    1. based on vA, determine whether the relative velocity circle overlaps the velocity obstacle
    2a. If it does, then find the best projection direction based on vA_d and magnitude. 
        Find vB if it exists. 
    2b. If there is no overlap or if vB does not exist, choose the best possible vB that tries to 
        get a solution in the next step
"""

import numpy as np
from numpy.linalg import norm
from policy.utils.overlap_detection import Circle, VelocityObstacle, Point, Tangent

EPSILON = 1e-6

class InverseORCA:
    """
    Class implementing the inverse ORCA algorithm
    """
    def __init__(self, vo: VelocityObstacle | None = None,
                 epsilon: float = 1e-5,
                 vB_max: float = 1.0, 
                 collision_responsibility: float = 0.5):
        """
        Initializes this class
        :param vo - the velocity obstacle object between the target human and the robot
        :param epsilon - the small error added to reduce projection ambiguity
        :param vB_max - the maximum possible speed of the robot
        :param collision_responsibility - the share of collision avoidance done by the human
        """
        self.vo = vo
        self.velocity_circle = None
        self.epsilon = epsilon
        self.vB_max = vB_max
        self.solution_exists = None
        self.overlap = None
        self.u = np.array([0., 0.])
        self.u_hat = None
        self.vB = None
        self.vA = None
        self.vA_d = None
        self.vA_new = None
        self.collision_responsibility = collision_responsibility

    def handle_right_leg(self, current_to_desired):
        self.u_hat = np.array(self.vo.right_tangent.normal)
        d = abs(self.u_hat.dot(current_to_desired))
        dmax = self.vo.find_dmax(self.velocity_circle, 'right')
        print(dmax, d)
        d = min(dmax, d)
        self.u = d * self.u_hat

        point = (self.vo.right_tangent.point[0] - self.u[0],
                 self.vo.right_tangent.point[1] - self.u[1])
        normal = self.vo.right_tangent.normal
        proj_line = Tangent(point, normal)
        # There is no overlap between the relative velocity circle and
        # the line at a distance d from the tangent
        if not self.velocity_circle.line_overlap(proj_line):
            print("No overlap of right projection line and velocity circle")
            self.solution_exists = False

        if self.solution_exists:
            # Compute the intersection point of the center line and the projection line
            center_line_normal = np.array([self.vo.cutoff_circle.center[1], -self.vo.cutoff_circle.center[0]])
            center_line_normal /= norm(center_line_normal)
            center_line_normal = tuple(center_line_normal)
            center_line = Tangent((0., 0.), center_line_normal)
            inter_point = center_line.intersect(proj_line)
            # dist_from_proj_line = proj_line.dist(self.vA)

            # If the intersection point is to the left of the normal,
            # the minimum vB would be such that the relative velocity
            # is at vA - dist_from_proj_line * u
            side = self.vo.right_tangent.side(inter_point)
            if side < 0:
                # print("right")
                relative_velocity = np.array(self.vo.right_tangent.point) - d * self.u_hat
                self.vB = self.vA - relative_velocity
                self.vA_new = tuple(np.array(self.vA) + self.collision_responsibility * self.u)
                self.vB = tuple(self.vB)
                return self.vB, self.u

            # dist_along_line = np.sqrt(self.vB_max ** 2 - dist_from_proj_line ** 2)
            # u_hat_perp = np.array([-self.u_hat[1], self.u_hat[0]])
            # relative_velocity = np.array(self.vA) - dist_from_proj_line * self.u_hat + dist_along_line * u_hat_perp
            # self.vB = tuple(np.array(self.vA) - relative_velocity)
            # self.vA_new = tuple(np.array(self.vA) + self.collision_responsibility * self.u)

            # If the intersection point is to the right of the normal,
            # then the minimum vB would be such that the relative velocity is at
            # the intersection point (plus epsilon in the u_perp direction to
            # avoid ambiguity of projection)
            if side > 0:
                # print("left")
                u_perp = np.array([-self.u_hat[1], self.u_hat[0]])
                # Move a bit away from the intersection point to avoid ambiguity
                relative_velocity = np.array(inter_point) + self.epsilon * u_perp
                self.vB = np.array(self.vA) - relative_velocity
                self.vB = tuple(self.vB)
                self.vA_new = tuple(np.array(self.vA) + self.collision_responsibility * self.u)
                return self.vB, self.u

        return None, None

    def handle_left_leg(self, current_to_desired):

        self.u_hat = -np.array(self.vo.left_tangent.normal)
        d = abs(self.u_hat.dot(current_to_desired))
        dmax = self.vo.find_dmax(self.velocity_circle, 'left')
        d = min(dmax, d)
        self.u = d * self.u_hat

        point = (self.vo.left_tangent.point[0] - self.u[0], 
                 self.vo.left_tangent.point[1] - self.u[1])
        normal = self.vo.left_tangent.normal
        proj_line = Tangent(point, normal)
        if not self.velocity_circle.line_overlap(proj_line):
            # print("No overlap of left projection line and velocity circle")
            self.solution_exists = False

        if self.solution_exists:
            center_line_normal = np.array([self.vo.cutoff_circle.center[1], -self.vo.cutoff_circle.center[0]])
            center_line_normal /= norm(center_line_normal)
            center_line_normal = tuple(center_line_normal)
            center_line = Tangent((0., 0.), center_line_normal)
            inter_point = center_line.intersect(proj_line)
            # dist_from_proj_line = proj_line.dist(self.vA)

            side = self.vo.left_tangent.side(inter_point)
            if side < 0:
                # print("right")
                relative_velocity = np.array(self.vo.left_tangent.point) - d * self.u_hat
                self.vB = self.vA - relative_velocity
                self.vA_new = tuple(np.array(self.vA) + self.collision_responsibility * self.u)
                self.vB = tuple(self.vB)
                return self.vB, self.u

            if side > 0:
                # print("left")
                u_perp = np.array([-self.u_hat[1], self.u_hat[0]])
                # Move a bit away from the intersection point to avoid ambiguity
                relative_velocity = np.array(inter_point) - self.epsilon * u_perp
                self.vB = np.array(self.vA) - relative_velocity
                self.vB = tuple(self.vB)
                self.vA_new = tuple(np.array(self.vA) + self.collision_responsibility * self.u)
                return self.vB, self.u

            # dist_from_proj_line = line.dist(self.vA)
            # dist_along_line = np.sqrt(self.vB_max ** 2 - dist_from_proj_line ** 2)
            # u_hat_perp = np.array([self.u_hat[1], -self.u_hat[0]])
            # relative_velocity = np.array(self.vA) - dist_from_proj_line * self.u_hat + dist_along_line * u_hat_perp
            # self.vB = tuple(np.array(self.vA) - relative_velocity)
            # self.vA_new = tuple(np.array(self.vA) + self.collision_responsibility * self.u)

        return None, None

    def handle_cutoff_circle(self, current_to_desired):

        l = norm(current_to_desired)
        self.u_hat = current_to_desired / l
        d = min(l, self.vo.cutoff_circle.radius - self.epsilon)
        self.u = d * self.u_hat
        self.vB = np.array(self.vA) - np.array(self.vo.cutoff_circle.center) + (d - self.vo.cutoff_circle.radius) * self.u_hat
        self.vB = tuple(self.vB)
        self.vA_new = tuple(np.array(self.vA) + self.collision_responsibility * self.u)

        return self.vB, self.u

    def handle_getting_closer_to_right_leg(self):

        self.u_hat = -np.array(self.vo.right_tangent.normal)
        right_point_to_center = np.array(self.vA) - np.array(self.vo.right_tangent.point)
        dist_from_right_leg = abs(self.u_hat.dot(right_point_to_center))
        d = min(self.vB_max, dist_from_right_leg - self.epsilon)
        self.u = d * self.u_hat
    
    def handle_getting_closer_to_left_leg(self):

        self.u_hat = np.array(self.vo.left_tangent.normal)
        left_point_to_center = np.array(self.vA) - np.array(self.vo.left_tangent.point)
        dist_from_left_leg = abs(self.u_hat.dot(left_point_to_center))
        d = min(self.vB_max, dist_from_left_leg - self.epsilon)
        self.u = d * self.u_hat

    def handle_getting_closer_to_cutoff_circle(self):

        self.u_hat = np.array(self.vo.cutoff_circle.center) - np.array(self.vA)
        l = norm(self.u_hat)
        self.u_hat /= l
        d = min(self.vB_max, l - self.epsilon)
        self.u = d * self.u_hat

    def handle_getting_closer_to_vo(self):

        cutoff_center_to_current = tuple(-np.array(self.vo.cutoff_circle.center) + np.array(self.vA))
        angle_right = get_angle(self.vo.right_tangent.normal, cutoff_center_to_current)
        angle_left = get_angle(self.vo.left_tangent.normal, cutoff_center_to_current)
        # print(f"Angle left {np.rad2deg(angle_left):.2f}")
        # print(f"Angle right {np.rad2deg(angle_right):.2f}")

        # Find the vector that takes the relative velocity closest to the velocity obstacle, but not in it.
        if np.pi/2 > angle_right > 0:
            # print("Getting closer to right leg")
            self.handle_getting_closer_to_right_leg()
        elif angle_left > np.pi / 2:
            # print("Getting closer to left leg")
            self.handle_getting_closer_to_left_leg()
        elif angle_right < 0 and angle_left < 0:
            # print("Getting closer to the cutoff circle")
            self.handle_getting_closer_to_cutoff_circle()

        self.vB = -self.u
        self.vB = tuple(self.vB)
        self.vA_new =  self.vA

        return self.vB, self.u

    def compute_velocity(self, vA: Point, vA_d: Point):
        """
        Implements the inverse ORCA algorithm
        Steps include:
            1. Checking for overlap between relative velocities and velocity obstacle
            2. Computing the optimal projection direction and magnitude
            3. Checking for the existence of solutions
            4. Returning the optimal velocity for B to influence vA towards vA_d
        """
        self.vA = vA
        self.vA_d = vA_d
        self.velocity_circle = Circle(vA, self.vB_max)
        self.overlap = self.vo.overlap(self.velocity_circle)
        self.solution_exists = False

        if self.overlap:
            self.solution_exists = True
            current_to_desired = tuple(np.array(vA_d) - np.array(vA))

            # Angle with the left tangent's normal
            angle_left = get_angle(self.vo.left_tangent.normal, current_to_desired)

            # Angle with the right tangent's normal
            angle_right = get_angle(self.vo.right_tangent.normal, current_to_desired)

            if 0 < angle_right < np.pi / 2:
                # print("Projecting on right leg")
                self.vB, self.u = self.handle_right_leg(current_to_desired)
                if self.solution_exists:
                    return self.vB, self.u

            if np.pi/2 < angle_left < np.pi:
                # print("Projecting on left leg")
                self.vB, self.u = self.handle_left_leg(current_to_desired)
                if self.solution_exists:
                    return self.vB, self.u

            if angle_right < 0 and angle_left < 0:
                # print("Projecting on cutoff circle")
                return self.handle_cutoff_circle(current_to_desired)

            if self.solution_exists:
                self.solution_exists = False
                # This means that there is an overlap between the velocity obstacle and the velocity
                # circle. However, the current conditions dictate that nudging the human towards
                # the desired velocity is not possible
                # Currently treating this case as the same as the one with no overlap

                # print("Something has gone terribly wrong....")
                # print('None of the conditions for getting best projection direction are satisfied')
                # print(f"Angle with right leg: {np.rad2deg(angle_right):.2f}")
                # print(f"Angle with left leg: {np.rad2deg(angle_left):.2f}")
                # return None, None

        if not self.solution_exists:
            return self.handle_getting_closer_to_vo()

def get_angle(vec1: Point, vec2: Point):
    """
    Gets the counterclockwise angle between vec1 and vec2
    """
    dot = np.dot(vec1, vec2)
    det = vec1[0] * vec2[1] - vec1[1] * vec2[0]

    return np.arctan2(det, dot)

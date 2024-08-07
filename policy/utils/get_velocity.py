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

class OptimalInfluence:
    """
    Class implementing the inverse ORCA algorithm
    """
    def __init__(self, vo: VelocityObstacle,
                 epsilon: float = 1e-5,
                 vr_max: float = 1.0,
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
        self.vr_max = vr_max
        self.solution_exists = None
        self.overlap = None
        self.u = np.array([0., 0.])
        self.u_hat = None
        self.vr = None
        self.vh = None
        self.vh_d = None
        self.vh_new = None
        self.collision_responsibility = collision_responsibility

    def handle_right_leg(self, current_to_desired):
        """
        Finds the optimal influence velocity for the robot and the optimal 
        projection vector when the desired projection is on the right leg
        """

        self.u_hat = np.array(self.vo.right_tangent.normal)
        d = abs(self.u_hat.dot(current_to_desired))
        # print("d:", d)
        dmax = self.vo.find_dmax(self.velocity_circle, 'right')
        # print(dmax, d)
        d = min(dmax, d)
        # print("dmax:", dmax)
        self.u = d * self.u_hat

        tangent = self.vo.right_tangent
        point = (tangent.point[0] - self.u[0], tangent.point[1] - self.u[1])
        normal = tangent.normal
        proj_line = Tangent(point, normal)
        # There is no overlap between the relative velocity circle and
        # the line at a distance d from the tangent
        if not self.velocity_circle.line_overlap(proj_line):
            # print("No overlap of right projection line and velocity circle")
            self.solution_exists = False

        if self.solution_exists:
            # Compute the intersection point of the center line and the projection line
            center = self.vo.cutoff_circle.center
            center_line_normal = np.array([center[1], -center[0]])
            center_line_normal /= norm(center_line_normal)
            center_line_normal = tuple(center_line_normal)
            center_line = Tangent((0., 0.), center_line_normal)
            inter_point = center_line.intersect(proj_line)

            # # If the intersection point is to the left of the normal,
            # # the minimum vB would be such that the relative velocity
            # # is at vA - dist_from_proj_line * u
            side = tangent.side(inter_point)
            if side < 0:
                # print("right")
                relative_velocity = np.array(tangent.point) - d * self.u_hat
                self.vr = self.vh - relative_velocity
                # self.vh_new = tuple(np.array(self.vh) + self.collision_responsibility * self.u)
                self.vr = tuple(self.vr)
                return self.vr, self.u

            # dist_from_proj_line = proj_line.dist(self.vA)
            # dist_along_line = np.sqrt(self.vB_max ** 2 - dist_from_proj_line ** 2)
            # u_hat_perp = np.array([-self.u_hat[1], self.u_hat[0]])
            # relative_velocity = np.array(self.vA) - dist_from_proj_line * self.u_hat +
            # dist_along_line * u_hat_perp
            # self.vB = tuple(np.array(self.vA) - relative_velocity)
            # self.vA_new = tuple(np.array(self.vA) + self.collision_responsibility * self.u)
            # return self.vB, self.u

            # If the intersection point is to the right of the normal,
            # then the minimum vB would be such that the relative velocity is at
            # the intersection point (plus epsilon in the u_perp direction to
            # avoid ambiguity of projection)
            if side >= 0:
                # print("left")
                u_perp = np.array([-self.u_hat[1], self.u_hat[0]])
                # Move a bit away from the intersection point to avoid ambiguity
                relative_velocity = np.array(inter_point) + self.epsilon * u_perp
                self.vr = np.array(self.vh) - relative_velocity
                self.vr = tuple(self.vr)
                return self.vr, self.u

        return None, None

    def compute_vh(self, u):
        """
        Computes the new human velocity given the projection
        """
        vh_new = np.array(self.vh) + self.collision_responsibility * u
        if norm(vh_new) > 1.0:
            vh_new /= norm(vh_new)
        return tuple(vh_new)


    def handle_left_leg(self, current_to_desired):
        """
        Finds the optimal influence velocity for the robot and the optimal 
        projection vector when the desired projection is on the left leg
        """

        self.u_hat = -np.array(self.vo.left_tangent.normal)
        d = abs(self.u_hat.dot(current_to_desired))
        # print("d:", d)
        dmax = self.vo.find_dmax(self.velocity_circle, 'left')
        d = min(dmax, d)
        # print("dmax:", dmax)
        self.u = d * self.u_hat

        tangent = self.vo.left_tangent
        point = (tangent.point[0] - self.u[0], tangent.point[1] - self.u[1])
        normal = tangent.normal
        proj_line = Tangent(point, normal)
        if not self.velocity_circle.line_overlap(proj_line):
            # print("No overlap of left projection line and velocity circle")
            self.solution_exists = False

        if self.solution_exists:
            center = self.vo.cutoff_circle.center
            center_line_normal = np.array([center[1], -center[0]])
            center_line_normal /= norm(center_line_normal)
            center_line_normal = tuple(center_line_normal)
            center_line = Tangent((0., 0.), center_line_normal)
            inter_point = center_line.intersect(proj_line)
            # dist_from_proj_line = proj_line.dist(self.vA)

            side = tangent.side(inter_point)
            if side < 0:
                # print("right")
                relative_velocity = np.array(tangent.point) - d * self.u_hat
                self.vr = self.vh - relative_velocity
                # self.vh_new = tuple(np.array(self.vh) + self.collision_responsibility * self.u)
                self.vr = tuple(self.vr)
                return self.vr, self.u

            if side >= 0:
                # print("left")
                u_perp = np.array([-self.u_hat[1], self.u_hat[0]])
                # Move a bit away from the intersection point to avoid ambiguity
                relative_velocity = np.array(inter_point) - self.epsilon * u_perp
                self.vr = np.array(self.vh) - relative_velocity
                self.vr = tuple(self.vr)
                # self.vh_new = tuple(np.array(self.vh) + self.collision_responsibility * self.u)
                return self.vr, self.u

            # dist_from_proj_line = proj_line.dist(self.vA)
            # dist_along_line = np.sqrt(self.vB_max ** 2 - dist_from_proj_line ** 2)
            # u_hat_perp = np.array([self.u_hat[1], -self.u_hat[0]])
            # relative_velocity = np.array(self.vA) - dist_from_proj_line * self.u_hat +
            # dist_along_line * u_hat_perp
            # self.vB = tuple(np.array(self.vA) - relative_velocity)
            # self.vA_new = tuple(np.array(self.vA) + self.collision_responsibility * self.u)
            # return self.vB, self.u

        return None, None

    def handle_cutoff_circle(self, current_to_desired):
        """
        Finds the optimal influence velocity for the robot and the optimal 
        projection vector when the desired projection is on the cutoff circle
        """

        l = norm(current_to_desired)
        self.u_hat = current_to_desired / l
        d = min(l, self.vo.cutoff_circle.radius - self.epsilon)
        self.u = d * self.u_hat
        center = self.vo.cutoff_circle.center
        radius = self.vo.cutoff_circle.radius
        self.vr = np.array(self.vh) - np.array(center) + (d - radius) * self.u_hat
        self.vr = tuple(self.vr)
        self.vh_new = tuple(np.array(self.vh) + self.collision_responsibility * self.u)

        return self.vr, self.u

    def handle_getting_closer_to_right_leg(self):
        """
        Handles the case when the closest possible relative velocity
        to the VO is near the right leg
        """

        self.u_hat = -np.array(self.vo.right_tangent.normal)
        # right_point_to_center = np.array(self.vA) - np.array(self.vo.right_tangent.point)
        # dist_from_right_leg = abs(self.u_hat.dot(right_point_to_center))
        dist_from_right_leg = self.vo.right_tangent.dist(self.vh)
        d = min(self.vr_max, dist_from_right_leg - self.epsilon)
        self.u = d * self.u_hat

    def handle_getting_closer_to_left_leg(self):
        """
        Handles the case when the closest possible relative velocity
        to the VO is near the left leg
        """

        self.u_hat = np.array(self.vo.left_tangent.normal)
        # left_point_to_center = np.array(self.vA) - np.array(self.vo.left_tangent.point)
        # dist_from_left_leg = abs(self.u_hat.dot(left_point_to_center))
        dist_from_left_leg = self.vo.left_tangent.dist(self.vh)
        d = min(self.vr_max, dist_from_left_leg - self.epsilon)
        self.u = d * self.u_hat

    def handle_getting_closer_to_cutoff_circle(self):
        """
        Handles the case when the closest possible relative velocity
        to the VO is near the cutoff circle
        """

        self.u_hat = np.array(self.vo.cutoff_circle.center) - np.array(self.vh)
        l = norm(self.u_hat)
        self.u_hat /= l
        # d = min(self.vB_max, l - self.epsilon)
        # We need to subtract the radius to ensure that we do not end up inside the VO
        d = min(self.vr_max, l - self.epsilon - self.vo.cutoff_circle.radius)
        self.u = d * self.u_hat

    def handle_getting_closer_to_vo(self):
        """
        Handles the case when there is no solution to influence the human's
        velocity in the desired direction. 
        In this case, we find the direction that is perpendicular to the VO
        from the current velocity and set the relative velocity to be 
        the point as close as possible to the VO
        """

        cutoff_center_to_current = tuple(-np.array(self.vo.cutoff_circle.center)
                                         + np.array(self.vh))
        angle_right = get_angle(self.vo.right_tangent.normal, cutoff_center_to_current)
        angle_left = get_angle(self.vo.left_tangent.normal, cutoff_center_to_current)
        # print(f"Angle left {np.rad2deg(angle_left):.2f}")
        # print(f"Angle right {np.rad2deg(angle_right):.2f}")

        # Find the vector that is closest to the velocity obstacle, but not in it.
        if np.pi/2 > angle_right > 0:
            # print("Getting closer to right leg")
            self.handle_getting_closer_to_right_leg()
        elif angle_left > np.pi / 2:
            # print("Getting closer to left leg")
            self.handle_getting_closer_to_left_leg()
        elif angle_right < 0 and angle_left < 0:
            # print("Getting closer to the cutoff circle")
            self.handle_getting_closer_to_cutoff_circle()
        else:  # This condition should not happen
            # TODO: Currently just setting it to the max goal-directed speed
            # print("Here?!")
            self.u = -np.array([self.vr_max, 0])

        self.vr = -self.u
        self.vr = tuple(self.vr)
        self.vh_new =  self.vh

        return self.vr, self.u

    def compute_velocity(self, vh: Point, vh_d: Point):
        """
        Implements the inverse ORCA algorithm
        Steps include:
            1. Checking for overlap between relative velocities and velocity obstacle
            2. Computing the optimal projection direction and magnitude
            3. Checking for the existence of solutions
            4. Returning the optimal velocity for B to influence vA towards vA_d
        """

        if norm(np.array(vh) - np.array(vh_d)) < self.epsilon:
            return self.vr, self.u

        self.vh = vh
        self.vh_d = vh_d

        self.velocity_circle = Circle(vh, self.vr_max)
        self.overlap = self.vo.overlap(self.velocity_circle)
        self.solution_exists = False

        if self.overlap:
            self.solution_exists = True
            current_to_desired = tuple(np.array(vh_d) - np.array(vh))

            # vr_r, u_r = self.handle_right_leg(current_to_desired)
            # vh_r = self.compute_vh(u_r)
            # dist_r = 1e5
            # right_sol_exists = self.solution_exists
            # if self.solution_exists:
            #     dist_r = norm(np.array(vh_r) - np.array(self.vh_d))

            # vr_l, u_l = self.handle_left_leg(current_to_desired)
            # vh_l = self.compute_vh(u_l)
            # dist_l = 1e5
            # left_sol_exits = self.solution_exists
            # if self.solution_exists:
            #     dist_l = norm(np.array(vh_l) - np.array(self.vh_d))

            # if left_sol_exits and right_sol_exists:
            #     if dist_r < dist_l:
            #         print("Selecting right leg")
            #         self.vr, self.u, self.vh_new = vr_r, u_r, vh_r
            #     else:
            #         print("Selecting left leg")
            #         self.vr, self.u, self.vh_new = vr_l, u_l, vh_l
            #     return self.vr, self.u

            # if left_sol_exits:
            #     print("Selecting left leg. right no exist")
            #     self.vr, self.u, self.vh_new = vr_l, u_l, vh_l
            #     return self.vr, self.u

            # if right_sol_exists:
            #     print("Selecting right leg. left no exist")
            #     self.vr, self.u, self.vh_new = vr_r, u_r, vh_r
            #     return self.vr, self.u

            dot_left = -np.dot(self.vo.left_tangent.normal, current_to_desired)
            dot_right = np.dot(self.vo.right_tangent.normal, current_to_desired)

            # print("dot_left:", dot_left)
            # print("dot_right:", dot_right)

            if dot_left < dot_right:
                # print("Projecting on right leg")
                self.vr, self.u = self.handle_right_leg(current_to_desired)
                # self.vr, self.u = self.handle_left_leg(current_to_desired)
                if self.u is not None:
                    self.vh_new = self.compute_vh(self.u)
                if self.solution_exists:
                    return self.vr, self.u
            else:
                # print("Projecting on left leg")
                # self.vr, self.u = self.handle_right_leg(current_to_desired)
                self.vr, self.u = self.handle_left_leg(current_to_desired)
                if self.u is not None:
                    self.vh_new = self.compute_vh(self.u)
                if self.solution_exists:
                    return self.vr, self.u


            # # Angle with the left tangent's normal
            # angle_left = get_angle(self.vo.left_tangent.normal, current_to_desired)

            # # Angle with the right tangent's normal
            # angle_right = get_angle(self.vo.right_tangent.normal, current_to_desired)

            # if 0 < angle_right < np.pi / 2:
            #     print("Projecting on right leg")
            #     self.vr, self.u = self.handle_right_leg(current_to_desired)
            #     if self.solution_exists:
            #         return self.vr, self.u

            # if np.pi/2 < angle_left < np.pi:
            #     print("Projecting on left leg")
            #     self.vr, self.u = self.handle_left_leg(current_to_desired)
            #     if self.solution_exists:
            #         return self.vr, self.u

            # if angle_right < 0 and angle_left < 0:
            #     print("Projecting on cutoff circle")
            #     return self.handle_cutoff_circle(current_to_desired)

            # if not right_sol_exists and not left_sol_exits:
            #     self.solution_exists = False
                # This means that there is an overlap between the velocity obstacle and the velocity
                # circle. However, the current conditions dictate that nudging the human towards
                # the desired velocity is not possible
                # Currently treating this case as the same as the one with no overlap

                # print("Something has gone terribly wrong....")
                # print('None of the conditions for getting best',
                #        'projection direction are satisfied')
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

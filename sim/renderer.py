"""
Renders the simulation
"""

from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from policy.utils.overlap_detection import Circle, VelocityObstacle, Point

class Renderer:
    """
    Class to help with the rendering of the Hallway Scene
    """

    def __init__(self, hallway_dimensions: Dict,
                 robot_parameters: Dict,
                 human_parameters: Dict,
                 time_step: float,
                 display_tails: bool = True) -> None:

        self.hallway_dimensions = hallway_dimensions
        self.robot_params = robot_parameters
        self.human_params = human_parameters

        self.robot_circ = None
        self.human_circ = None
        self.observations = None
        self.num_frames = None
        self.time_step = time_step
        self.ax = None
        self.goal_height = None
        self.goal_dist = None
        self.virtual_goal = None
        self.goal_frame = None

        self.human_positions = []
        self.robot_positions = []
        self.alpha = None
        self.tails = display_tails

        self.vel_arrow_robot = None
        self.vel_arrow_human = None

        # Debugging
        self.debug_vel_ax = None

    def generate_background(self, ax: plt.Axes, robot_pos: Tuple[float, float],
                            human_pos: Tuple[float, float]):
        """
        Creates the hallway background (walls, initial positions of the human and the robot)
        """
        # Add the hallway boundaries
        ax.axhline(y=0,
                   c='black', lw=3)
        ax.axhline(y=self.hallway_dimensions['width'],
                   c='black', lw=3)
        # ax.set_ylim([-0.05, self.hallway_dimensions['width'] + 0.05])

        # Add the human goal
        ax.axvline(x=0, ls='--', lw=1, c='red')

        # Add the robot goal
        ax.axvline(x=self.hallway_dimensions['length'],
                   c='blue', lw=1, ls='--')

        # Add the robot
        self.robot_circ = plt.Circle(robot_pos, self.robot_params['radius'],
                                     color=self.robot_params['color'])
        ax.add_patch(self.robot_circ)

        # Add the human
        self.human_circ = plt.Circle(human_pos, self.human_params['radius'],
                                     color=self.human_params['color'])
        ax.add_patch(self.human_circ)

        # Add the robot's virtual goal for the human
        x = self.human_circ.center[0] - self.goal_dist
        ymax = self.goal_height #/ self.hallway_dimensions['width']
        ymin = 0
        # self.virtual_goal = ax.axvline(x=x, ymax=ymax, lw=1, ls='--', c='gold')
        self.virtual_goal, = ax.plot([x, x], [ymin, ymax], lw=1, ls='--', color='gold')

        return ax

    def set_observations(self, observations: List[Dict], goal_frame: int):
        """
        Adds the observations and the frame in which the goal was reached
        :param observations - the list of observations in the simulation
        :param goal_frame - the index of the frame in which the robot's goal 
                            for the human was reached
        """
        self.goal_frame = goal_frame
        self.num_frames = len(observations)
        self.observations = observations

    def set_goal_params(self, goal_dist: float, goal_height: float):
        """
        Sets the parameters for the virtual goal line of the robot for the human
        :param goal_dist - the x-distance of the goal line from the human's current position
        :param goal_height - the y-coordinate of the goal-line's topmost point
        """
        self.goal_dist = goal_dist
        self.goal_height = goal_height

    def update(self, frame_num, show_velocities: bool=True):
        """
        The update function used in the animation to update a frame
        :param frame_num - the index of the frame to plot
        :param show_velocities - whether to plot the velocity vectors
        """
        observation = self.observations[frame_num]
        if self.robot_circ is None or self.human_circ is None:
            self.generate_background(self.ax, tuple(observation['robot pos']),
                                     tuple(observation['human pos']))

        self.robot_circ.set_center(tuple(observation['robot pos']))
        self.human_circ.set_center(tuple(observation['human pos']))

        if show_velocities:
            robot_vel = observation['robot vel']
            human_vel = observation['human vel']
            x_start, y_start = self.robot_circ.get_center()
            x_end, y_end = robot_vel[0], robot_vel[1]

            if self.vel_arrow_robot is None:
                self.vel_arrow_robot = plt.arrow(x_start, y_start, x_end, y_end,
                                                 color='black', width=0.02)
            else:
                self.vel_arrow_robot.set_data(x=x_start, y=y_start, dx=x_end, dy=y_end)

            x_start, y_start = self.human_circ.get_center()
            x_end, y_end = human_vel[0], human_vel[1]

            if self.vel_arrow_human is None:
                self.vel_arrow_human = plt.arrow(x_start, y_start, x_end, y_end,
                                                 color='black', width=0.02)
            else:
                self.vel_arrow_human.set_data(x=x_start, y=y_start, dx=x_end, dy=y_end)


        if self.tails:
            self.ax.scatter(*tuple(observation['robot pos']),
                            alpha=max(0.1, frame_num/self.num_frames), c=self.robot_params['color'])
            self.ax.scatter(*tuple(observation['human pos']),
                            alpha=max(0.1, frame_num/self.num_frames), c=self.human_params['color'])

        # Only update the virtual goal line if goal has not been reached yet
        # print(f"goal frame: {self.goal_frame}, total frames: {self.num_frames}")
        if frame_num < self.goal_frame:
            # print(f"frame_num: {frame_num}, goal_frame: {self.goal_frame}")
            x = self.human_circ.center[0] - self.goal_dist
            self.virtual_goal.set_xdata([x, x])

        return self.robot_circ, self.human_circ, self.virtual_goal

    def static_plot(self, fig: plt.Figure, ax: plt.Axes):
        """
        Function to plot the trajectories of the agents as static plots
        """
        assert self.num_frames is not None, "Please set observations first"
        assert self.observations is not None, "Please set observations first"
        self.ax = ax
        obs = self.observations[0]
        self.generate_background(self.ax, tuple(obs['robot pos']),
                                 tuple(obs['human pos']))
        self.ax.annotate('0', tuple(obs['robot pos']), ha='center')
        self.ax.annotate('0', tuple(obs['human pos']), ha='center')

        for obs in self.observations:
            self.human_positions.append(obs['human pos'])
            self.robot_positions.append(obs['robot pos'])

        human_pos_array = np.array(self.human_positions)
        robot_pos_array = np.array(self.robot_positions)

        human_rad = self.observations[0]['human rad']
        robot_rad = self.observations[0]['robot rad']
        human_goal_frame = np.nonzero(human_pos_array - human_rad <= 0)[0]
        if len(human_goal_frame) == 0:
            human_goal_frame = self.num_frames - 1
        else:
            human_goal_frame = human_goal_frame[0]
        condition = robot_pos_array + robot_rad >= self.hallway_dimensions['length']
        robot_goal_frame = np.nonzero(condition)[0]
        if len(robot_goal_frame) == 0:
            robot_goal_frame = self.num_frames - 1
        else:
            robot_goal_frame = robot_goal_frame[0]

        self.ax.plot(human_pos_array[:, 0], human_pos_array[:, 1],
                     c=self.human_params['color'], ls='-.', lw=2)
        self.ax.plot(robot_pos_array[:, 0], robot_pos_array[:, 1],
                     c=self.robot_params['color'], ls='-.', lw=2)

        if self.goal_frame < self.num_frames:
            # Add the robot
            self.robot_circ = plt.Circle(robot_pos_array[self.goal_frame, :],
                                         self.robot_params['radius'],
                                         color=self.robot_params['color'])
            ax.add_patch(self.robot_circ)

            # Add the human
            self.human_circ = plt.Circle(human_pos_array[self.goal_frame, :],
                                         self.human_params['radius'],
                                         color=self.human_params['color'])
            ax.add_patch(self.human_circ)

            x = self.human_circ.center[0]
            self.virtual_goal.set_xdata([x, x])

            obs = self.observations[self.goal_frame]
            self.ax.annotate(f'{self.goal_frame}', tuple(obs['robot pos']), ha='center')
            self.ax.annotate(f'{self.goal_frame}', tuple(obs['human pos']), ha='center')

        # Add the robot
        self.robot_circ = plt.Circle(robot_pos_array[-1, :], self.robot_params['radius'],
                                     color=self.robot_params['color'])
        ax.add_patch(self.robot_circ)

        # Add the human
        self.human_circ = plt.Circle(human_pos_array[-1, :], self.human_params['radius'],
                                     color=self.human_params['color'])
        ax.add_patch(self.human_circ)

        obs = self.observations[-1]
        self.ax.annotate(f'{robot_goal_frame}', tuple(obs['robot pos']), ha='center')
        self.ax.annotate(f'{human_goal_frame}', tuple(obs['human pos']), ha='center')

        return fig, ax

    def animate(self, fig: plt.Figure, ax: plt.Axes):
        """
        Function to create an animation from the simulation
        """
        assert self.num_frames is not None, "Please set observations first"
        assert self.observations is not None, "Please set observations first"

        self.ax = ax

        anim = animation.FuncAnimation(fig=fig, func=self.update,
                                       frames=self.num_frames, interval=self.time_step * 1000)

        return anim

    def debug_mode(self, fig: plt.Figure, anim_ax: plt.Axes, velocity_axis: plt.Axes,
                   vc: List[Circle], vo: List[VelocityObstacle],
                   vh_d: List[Point], vr: List[Point], vh_expected: List[Point],
                   vh_new: List[Point], u: List[Point]):
        """
        Running the renderer in debug mode
        Plots the positions of the agents, and the velocity-space diagram of ORCA
        at each frame.
        Use arrow keys to navigate through the frames
        """

        assert self.num_frames is not None, "Please set observations first"
        assert self.observations is not None, "Please set observations first"

        self.ax = anim_ax
        self.debug_vel_ax = velocity_axis
        frame_id = 0

        def onkey(event):
            nonlocal frame_id
            if event.key == 'left':
                frame_id -= 1
                frame_id = max(0, frame_id)
            if event.key == 'right':
                frame_id += 1
                frame_id = min(self.num_frames - 1, frame_id)

            self.robot_circ = None
            self.human_circ = None
            self.vel_arrow_human = None
            self.vel_arrow_robot = None
            self.debug_vel_ax.clear()
            self.ax.clear()
            self.update(frame_id, show_velocities=True)
            vc[frame_id].plot(self.debug_vel_ax)
            vo[frame_id].plot(self.debug_vel_ax)
            self.debug_vel_ax.scatter(vc[frame_id].center[0], vc[frame_id].center[1],
                                      color='red', s=9, label=r'$v_h$')
            self.debug_vel_ax.scatter(vh_d[frame_id][0], vh_d[frame_id][1],
                                      color='tab:purple', label=r'$v_h^d$')
            self.debug_vel_ax.scatter(vr[frame_id][0], vr[frame_id][1],
                                      color='tab:orange', label=r'$v_r$')
            self.debug_vel_ax.scatter(vh_expected[frame_id][0], vh_expected[frame_id][1],
                                      color='black', alpha=0.7,
                                      label=r'$\hat{v}_h^{new}$', marker='*')
            self.debug_vel_ax.scatter(vh_new[frame_id][0], vh_new[frame_id][1],
                                      color='tab:blue', alpha=0.7, label=r'$v_h^{new}$', marker='^')

            # Add the preferred velocity of the human
            self.debug_vel_ax.scatter([-1.], [0.], color='red', marker='*', label=r'$v_h^{pref}$')

            # Add the relative velocity
            relvel = np.array(vc[frame_id].center) - np.array(vr[frame_id])
            self.debug_vel_ax.scatter(relvel[0], relvel[1],
                                      color='green', marker='o', label='relvel')

            # Add the relative velocity with the preferred velocity
            self.debug_vel_ax.scatter([-1.0 - vr[frame_id][0]], [-vr[frame_id][1]],
                                      color='magenta', marker='+', label='relvelpref')

            # Add u
            if u[frame_id] is not None:
                u_ = np.array(vc[frame_id].center) - np.array(vr[frame_id]) + np.array(u[frame_id])
                self.debug_vel_ax.scatter(u_[0], u_[1],
                                          color='blue', marker='s', label=r'$u$')

            self.debug_vel_ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
            # self.ax.grid(True)
            fig.canvas.draw_idle()

        self.update(frame_id, show_velocities=True)
        # self.ax.grid(True)

        vc[frame_id].plot(self.debug_vel_ax)
        vo[frame_id].plot(self.debug_vel_ax)

        self.debug_vel_ax.scatter(vh_d[frame_id][0], vh_d[frame_id][1],
                                  color='tab:purple', label=r'$v_h^d$')
        self.debug_vel_ax.scatter(vr[frame_id][0], vr[frame_id][1],
                                  color='tab:orange', label=r'$v_r$')
        self.debug_vel_ax.scatter(vh_expected[frame_id][0], vh_expected[frame_id][1],
                                  color='black', alpha=0.7, label=r'$\hat{v}_h^{new}$', marker='*')
        self.debug_vel_ax.scatter(vh_new[frame_id][0], vh_new[frame_id][1],
                                  color='tab:blue', alpha=0.9, label=r'$v_h^{new}$', marker='^')
        self.debug_vel_ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

        _ = fig.canvas.mpl_connect('key_press_event', onkey)

        plt.show()

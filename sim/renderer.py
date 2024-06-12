# TODO: Add velocity arrows
# TODO: Check if the framework is compatible with other Social Navigation algorithms

from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from policy.utils.OverlapDetection import Circle, VelocityObstacle, Point

class Renderer:
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

        # Add the robot's goal for the human
        x = self.human_circ.center[0] - self.goal_dist
        ymax = self.goal_height / self.hallway_dimensions['width']
        self.virtual_goal = ax.axvline(x=x, ymax=ymax, lw=1, ls='--', c='gold')

        return ax

    def set_observations(self, observations: List[Dict], goal_frame: int):
        self.goal_frame = goal_frame
        self.num_frames = len(observations)
        self.observations = observations
    
    def set_goal_params(self, goal_dist: float, goal_height: float):
        self.goal_dist = goal_dist
        self.goal_height = goal_height
    
    def update(self, frame_num):
        observation = self.observations[frame_num]
        if self.robot_circ is None or self.human_circ is None:
            self.generate_background(self.ax, tuple(observation['robot pos']),
                                     tuple(observation['human pos']))

        self.robot_circ.set_center(tuple(observation['robot pos']))
        self.human_circ.set_center(tuple(observation['human pos']))

        if self.tails:
            self.ax.scatter(*tuple(observation['robot pos']), alpha=max(0.1, frame_num/self.num_frames), c=self.robot_params['color'])
            self.ax.scatter(*tuple(observation['human pos']), alpha=max(0.1, frame_num/self.num_frames), c=self.human_params['color'])

        # Only update the virtual goal line if goal has not been reached yet
        # print(f"goal frame: {self.goal_frame}, total frames: {self.num_frames}")
        if frame_num < self.goal_frame:
            # print(f"frame_num: {frame_num}, goal_frame: {self.goal_frame}")
            x = self.human_circ.center[0] - self.goal_dist
            self.virtual_goal.set_xdata(x)

        return self.robot_circ, self.human_circ, self.virtual_goal

    def static_plot(self, fig: plt.Figure, ax: plt.Axes):
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

        self.ax.plot(human_pos_array[:, 0], human_pos_array[:, 1],
                     c=self.human_params['color'], ls='-.', lw=2)
        self.ax.plot(robot_pos_array[:, 0], robot_pos_array[:, 1],
                     c=self.robot_params['color'], ls='-.', lw=2)

        if self.goal_frame < self.num_frames:
            # Add the robot
            self.robot_circ = plt.Circle(robot_pos_array[self.goal_frame, :], self.robot_params['radius'],
                                        color=self.robot_params['color'])
            ax.add_patch(self.robot_circ)

            # Add the human
            self.human_circ = plt.Circle(human_pos_array[self.goal_frame, :], self.human_params['radius'],
                                        color=self.human_params['color'])
            ax.add_patch(self.human_circ)

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
        self.ax.annotate(f'{self.num_frames-1}', tuple(obs['robot pos']), ha='center')
        self.ax.annotate(f'{self.num_frames-1}', tuple(obs['human pos']), ha='center')

        return fig, ax

    def animate(self, fig: plt.Figure, ax: plt.Axes):
        assert self.num_frames is not None, "Please set observations first"
        assert self.observations is not None, "Please set observations first"

        self.ax = ax

        anim = animation.FuncAnimation(fig=fig, func=self.update,
                                       frames=self.num_frames, interval=self.time_step * 1000)

        return anim

    def debug_mode(self, fig: plt.Figure, anim_ax: plt.Axes, velocity_axis: plt.Axes,
                   vc: List[Circle], vo: List[VelocityObstacle],
                   vA_d: List[Point], vB: List[Point], vA_new_expected: List[Point],
                   vA_new_actual: List[Point]):

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
            self.debug_vel_ax.clear()
            self.ax.clear()
            self.update(frame_id)
            vc[frame_id].plot(self.debug_vel_ax)
            vo[frame_id].plot(self.debug_vel_ax)
            self.debug_vel_ax.scatter(vc[frame_id].center[0], vc[frame_id].center[1],
                                      color='red', s=9, label=r'$v_h$')

            self.debug_vel_ax.scatter(vA_d[frame_id][0], vA_d[frame_id][1],
                                      color='tab:purple', label=r'$v_h^d$')
            self.debug_vel_ax.scatter(vB[frame_id][0], vB[frame_id][1],
                                      color='tab:orange', label=r'$v_r$')
            self.debug_vel_ax.scatter(vA_new_expected[frame_id][0], vA_new_expected[frame_id][1],
                                      color='black', alpha=0.7, label=r'$\hat{v}_h^{new}$', marker='*')
            self.debug_vel_ax.scatter(vA_new_actual[frame_id][0], vA_new_actual[frame_id][1],
                                      color='tab:blue', alpha=0.7, label=r'$v_h^{new}$', marker='^')
            self.debug_vel_ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
            self.ax.grid(True)
            fig.canvas.draw_idle()

        self.update(frame_id)
        self.ax.grid(True)

        vc[frame_id].plot(self.debug_vel_ax)
        vo[frame_id].plot(self.debug_vel_ax)

        self.debug_vel_ax.scatter(vA_d[frame_id][0], vA_d[frame_id][1],
                                  color='tab:purple', label=r'$v_h^d$')
        self.debug_vel_ax.scatter(vB[frame_id][0], vB[frame_id][1],
                                  color='tab:orange', label=r'$v_r$')
        self.debug_vel_ax.scatter(vA_new_expected[frame_id][0], vA_new_expected[frame_id][1],
                                  color='black', alpha=0.7, label=r'$\hat{v}_h^{new}$', marker='*')
        self.debug_vel_ax.scatter(vA_new_actual[frame_id][0], vA_new_actual[frame_id][1],
                                  color='tab:blue', alpha=0.9, label=r'$v_h^{new}$', marker='^')
        self.debug_vel_ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        
        cid = fig.canvas.mpl_connect('key_press_event', onkey)

        plt.show()

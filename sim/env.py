# TODO: Check if the framework is compatible with other Social Navigation algorithms

import datetime
import uuid
from configparser import RawConfigParser
from typing import Dict, Any
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from sim.agent import Human, Robot
from sim.renderer import Renderer
from policy.utils.overlap_detection import Circle



class HallwayScene(gym.Env):

    metadata = {"render_modes": ["human", "static", "debug"], "render_fps": 5}

    def __init__(self) -> None:
        super().__init__()
        self.human = None
        self.robot = None
        self.config = None
        self.hallway_length = None
        self.hallway_width = None
        self.hallway_dimensions = None
        self.d_virtual_goal = None
        self.y_virtual_goal = None
        self.robot_start_box = None
        self.human_start_box = None

        self.time_step = None
        self.global_time = None
        self.observations = None
        self.time_limit = None

        self.observation_space = None
        self.action_space = None

        left = -100
        right = 100

        self.observation_space = spaces.Dict(
            {
                "robot pos": spaces.Box(left, right,
                                        shape=(2, ), dtype=float),
                "robot vel": spaces.Box(left, right, 
                                        shape=(2,), dtype=float),
                "human pos": spaces.Box(left, right,
                                        shape=(2, ), dtype=float),
                "human vel": spaces.Box(left, right, 
                                        shape=(2,), dtype=float)
            }
        )

        self.action_space = spaces.Dict(
            {
                "robot vel": spaces.Box(left, right, 
                                        shape=(2,), dtype=float),
                "human vel": spaces.Box(left, right, 
                                        shape=(2,), dtype=float)
            }
        )


        self.renderer = None
        self.robot_draw_params = None
        self.human_draw_params = None

        self.frame_count = None
        self.goal_frame = None
        self.goal_frame_set = False

        # For debug mode
        self.debug = None
        self.vo_list = None
        self.vc_list = None
        self.va_d_list = None
        self.vb_list = None
        self.va_exp_list = None
        self.va_act_list = None
        self.u_list = None

        # For saving the plots
        self.render_mode = None
        self.save_anim = None

    def configure(self, config: RawConfigParser, save_anim: bool,
                  render_mode: str = "human"):
        """
        Configures the environment
        """
        self.config = config
        self.hallway_length = config.getfloat('env', 'hallway_length')
        self.hallway_width = config.getfloat('env', 'hallway_width')
        self.hallway_dimensions = {"length": self.hallway_length,
                                   "width": self.hallway_width}
        self.time_step = config.getfloat('env', 'time_step')
        self.time_limit = config.getfloat('env', 'time_limit')
        self.d_virtual_goal = config.getfloat('env', 'd_virtual_goal')
        self.y_virtual_goal = config.getfloat('env', 'y_virtual_goal')

        self.render_mode = render_mode

        if render_mode == 'debug':
            self.debug = True
            self.save_anim = False
        else:
            self.save_anim = save_anim

        if self.debug:
            self.vo_list = []
            self.vc_list = []
            self.va_d_list = []
            self.vb_list = []
            self.va_exp_list = []
            self.va_act_list = []
            self.u_list = []

        left = -4.
        right = self.hallway_length + 4.

        self.observation_space = spaces.Dict(
            {
                "robot pos": spaces.Box(left, right,
                                        shape=(2, ), dtype=float),
                "robot vel": spaces.Box(-self.robot.max_speed, self.robot.max_speed, 
                                        shape=(2,), dtype=float),
                "human pos": spaces.Box(left, right,
                                        shape=(2, ), dtype=float),
                "human vel": spaces.Box(-self.human.max_speed, self.human.max_speed, 
                                        shape=(2,), dtype=float)
            }
        )

        self.action_space = spaces.Dict(
            {
                "robot vel": spaces.Box(-self.robot.max_speed, self.robot.max_speed, 
                                        shape=(2,), dtype=float),
                "human vel": spaces.Box(-self.human.max_speed, self.human.max_speed, 
                                        shape=(2,), dtype=float)
            }
        )

        self.robot_draw_params = {"radius": self.robot.radius, "color": "tab:blue"}
        self.human_draw_params = {"radius": self.human.radius, "color": 'tab:red'}
        self.renderer = Renderer(self.hallway_dimensions,
                                 self.robot_draw_params, self.human_draw_params,
                                 self.time_step)
        self.renderer.set_goal_params(self.d_virtual_goal, self.y_virtual_goal)

        # Set the box for human start positions        
        low = np.array([self.hallway_length * 0.95, self.hallway_width * 0.4 + self.human.radius])
        high = np.array([self.hallway_length * 0.99 - self.human.radius,
                         self.hallway_width * 0.7 - self.human.radius])
        self.human_start_box = spaces.Box(low=low, high=high)

        # Set the box for robot start positions
        low = np.array([self.hallway_length * 0.05 + self.robot.radius,
                        self.hallway_width * 0.6 + self.robot.radius])
        high = np.array([self.hallway_length * 0.1, self.hallway_width * 0.9 - self.robot.radius])
        self.robot_start_box = spaces.Box(low=low, high=high)

        self.render_mode = render_mode

    def set_robot(self, robot: Robot):
        self.robot = robot

    def set_human(self, human: Human):
        self.human = human

    def step(self, action):
        self.robot.step(action, self.time_step)
        center = self.human.get_velocity()      # The velocity before updating it via ORCA
        self.human.step(action, self.time_step)
        self.frame_count += 1
        obs = self.__get_obs()
        self.observations.append(obs)

        if self.debug:
            self.vo_list.append(self.robot.policy.vo)
            self.vc_list.append(Circle(center, self.robot.max_speed))
            self.va_d_list.append(self.robot.vh_desired)
            self.vb_list.append(self.robot.get_velocity())
            self.va_exp_list.append(self.robot.policy.invorca.vA_new)
            self.va_act_list.append(tuple(action["human vel"]))   # The velocity after updating it via ORCA
            self.u_list.append(self.robot.policy.u)

        if (not self.goal_frame_set) and self.robot.human_reached_goal(obs):
            self.goal_frame_set = True
            self.goal_frame = self.frame_count
            # print("Goal frame:", self.goal_frame)

        self.global_time += self.time_step
        reward = 0
        truncated = self.global_time > self.time_limit
        terminated = self.robot.reached_goal() or self.human.reached_goal()
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        # TODO: Assuming that render mode is human and showing a animated video of the frames
        # TODO: We may want to plot the static trajectory
        if self.goal_frame is None:
            self.goal_frame = len(self.observations)

        self.renderer.set_observations(self.observations, self.goal_frame)

        if not self.debug:
            if self.render_mode == 'human':
                fig, ax = plt.subplots(figsize=(9,6))
                ax.set_aspect('equal')
                anim = self.renderer.animate(fig, ax)
                if self.save_anim:
                    writervideo = animation.FFMpegWriter(fps=5)
                    ts = datetime.datetime.now().strftime("%m-%d %H-%M-%S")
                    anim.save(f'media/videos/{ts}-{uuid.uuid4().hex}.mp4', writer=writervideo)
                    plt.close(fig)
                # plt.show()
            if self.render_mode == 'static':
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.set_aspect('equal')
                fig, ax = self.renderer.static_plot(fig, ax)
                if self.save_anim:
                    ts = datetime.datetime.now().strftime("%m-%d %H-%M-%S")
                    plt.savefig(f'media/static-plots/{ts}-{uuid.uuid4().hex}.png')
                    plt.close(fig)
                # plt.show()

        else:
            fig, (ax1, ax2) = plt.subplots(figsize=(9, 9), nrows=2, ncols=1, height_ratios=[0.7, 0.3])
            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            self.renderer.debug_mode(fig, ax2, ax1, self.vc_list, self.vo_list,
                                     self.va_d_list, self.vb_list,
                                     self.va_exp_list, self.va_act_list,
                                     self.u_list)

    def __get_obs(self):
        obs = {
            "robot pos": np.array(self.robot.get_position(), dtype=float),
            "robot vel": np.array(self.robot.get_velocity(), dtype=float),
            "human pos": np.array(self.human.get_position(), dtype=float),
            "human vel": np.array(self.human.get_velocity(),  dtype=float)
        }

        return obs

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)

        # Reset the goal position
        self.human.set_goal(0, self.hallway_width / 2)
        self.robot.set_goal(self.hallway_length, self.hallway_width / 2)

        # Reset the position of human
        human_pos = self.human_start_box.sample()
        self.human.set_position(human_pos[0], human_pos[1])
        # self.human.set_position(self.hallway_length * 0.98, 2.5) # Debugging

        # Reset the position of the robot
        robot_pos = self.robot_start_box.sample()
        self.robot.set_position(robot_pos[0], robot_pos[1])
        # self.robot.set_position(self.hallway_length * 0.02, 2.5)

        # Set the preferred velocity of the human and the robot
        self.human.set_preferred_velocity()
        self.robot.set_preferred_velocity()

        self.global_time = 0
        self.observations = []

        obs = self.__get_obs()
        self.observations.append(obs)
        self.frame_count = 0
        self.goal_frame = None
        self.goal_frame_set = False

        self.renderer = Renderer(self.hallway_dimensions,
                                 self.robot_draw_params, self.human_draw_params,
                                 self.time_step)
        self.renderer.set_goal_params(self.d_virtual_goal, self.y_virtual_goal)

        return obs, {}

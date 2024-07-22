"""
The Hallway Crossing environment, written as a child of gymnasium.Env
"""
# TODO: Check if the framework is compatible with other Social Navigation algorithms

import datetime
import uuid
from typing import Dict, Any
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from sim.agent import Human, Robot
from sim.renderer import Renderer
from sim.collision_checker import check_collision
from policy.utils.overlap_detection import Circle


class HallwayScene(gym.Env):
    """
    The hallway scene environment:
    Two agents start at the ends of a hallway and want to move to the opposite end
    """

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

        left = -100
        right = 100

        self.observation_space = spaces.Dict(
            {
                "robot pos": spaces.Box(left, right,
                                        shape=(2, ), dtype=float),
                "robot vel": spaces.Box(left, right, 
                                        shape=(2,), dtype=float),
                "robot rad": spaces.Box(left, right, 
                                        shape=(1,), dtype=float),
                "human pos": spaces.Box(left, right,
                                        shape=(2, ), dtype=float),
                "human vel": spaces.Box(left, right, 
                                        shape=(2,), dtype=float),
                "human rad": spaces.Box(left, right,
                                        shape=(1,), dtype=float)
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
        self.filename = None

        # Collision checker
        self.collision = False
        self.collision_frames = []

    def configure(self, config: Dict, save_anim: bool,
                  render_mode: str | None = "human"):
        """
        Configures the environment
        """
        self.config = config['env']
        self.hallway_length = self.config['hallway_length']
        self.hallway_width = self.config['hallway_width']
        self.hallway_dimensions = {"length": self.hallway_length,
                                   "width": self.hallway_width}
        self.time_step = self.config['time_step']
        self.time_limit = self.config['time_limit']
        self.d_virtual_goal = self.config['d_virtual_goal']
        self.y_virtual_goal = self.config['y_virtual_goal']

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

        self.robot_draw_params = {"radius": self.robot.radius, "color": "tab:blue"}
        self.human_draw_params = {"radius": self.human.radius, "color": 'tab:red'}
        self.renderer = Renderer(self.hallway_dimensions,
                                 self.robot_draw_params, self.human_draw_params,
                                 self.time_step)
        self.renderer.set_goal_params(self.d_virtual_goal, self.y_virtual_goal)

        # Set the box for human start positions
        radius = self.human.radius
        # low = np.array([self.hallway_length - 5 * radius, self.hallway_width * 0.4])
        # high = np.array([self.hallway_length - radius, self.hallway_width * 0.7])
        low = np.array([self.hallway_length - 5 * radius, self.hallway_width * 0.5])
        high = np.array([self.hallway_length - radius, self.hallway_width * 0.7])
        self.create_agent_start_box('human', low, high)

        # Set the box for robot start positions
        radius = self.robot.radius
        # low = np.array([self.robot.radius, self.hallway_width * 0.7])
        # high = np.array([5 * radius, self.hallway_width - radius])
        low = np.array([self.robot.radius, self.hallway_width * 0.2])
        high = np.array([5 * radius, self.hallway_width * 0.5])
        self.create_agent_start_box('robot', low, high)

        self.render_mode = render_mode

    def set_robot(self, robot: Robot):
        """
        Adds the robot to the environment
        :param robot - the robot
        """
        self.robot = robot

    def create_agent_start_box(self, agent: str, low: np.ndarray, high: np.ndarray):
        """
        Creates a start box for randomly generating the starting position of 
        an agent
        """
        if agent == 'human':
            self.human_start_box = spaces.Box(low=low, high=high)
        elif agent == 'robot':
            self.robot_start_box = spaces.Box(low=low, high=high)


    def set_human(self, human: Human):
        """
        Adds the human to the environment
        :param human - the human
        """
        self.human = human

    def step(self, action):
        """
        Takes one step in the simulation
        :param action - a dict with keys 'human vel' and 'robot vel'
        """
        self.robot.step(action, self.time_step)
        center = self.human.get_velocity()      # The velocity before updating it via ORCA
        self.human.step(action, self.time_step)

        self.frame_count += 1
        obs = self.__get_obs()
        self.observations.append(obs)

        # Check for collision
        self.collision = check_collision(obs)
        if self.collision:
            self.collision_frames.append(self.frame_count)

        if self.debug:
            self.vo_list.append(self.robot.policy.vo)
            self.vc_list.append(Circle(center, self.robot.max_speed))
            self.va_d_list.append(self.robot.vh_desired)
            self.vb_list.append(self.robot.get_velocity())
            self.va_exp_list.append(self.robot.policy.invorca.vh_new)
            # The velocity after updating it via ORCA
            self.va_act_list.append(tuple(action["human vel"]))
            self.u_list.append(self.robot.policy.u)

        if (not self.goal_frame_set) and self.robot.human_reached_goal(obs):
            self.goal_frame_set = True
            self.goal_frame = self.frame_count
            # print("Goal frame:", self.goal_frame)

        self.global_time += self.time_step
        reward = 0
        truncated = self.global_time > self.time_limit
        # terminated = self.robot.reached_goal() or self.human.reached_goal()
        direction = np.sign(self.robot.gx - self.robot.px)
        terminated = self.robot.reached_goal(direction) and self.human.reached_goal()
        info = {}
        return obs, reward, terminated, truncated, info

    def set_output_filename(self, fname: str):
        """
        Sets the filename to be used for saving animations/plots
        """
        self.filename = fname

    def render(self):

        if self.render_mode is None:
            return

        if self.goal_frame is None:
            self.goal_frame = len(self.observations)

        self.renderer.set_observations(self.observations, self.goal_frame)

        if not self.debug:
            if self.render_mode == 'human':
                fig, ax = plt.subplots(figsize=(9,6))
                ax.set_aspect('equal')
                anim = self.renderer.animate(fig, ax)
                if self.save_anim:
                    filename = self.filename
                    if filename is None:
                        ts = datetime.datetime.now().strftime("%m-%d %H-%M-%S")
                        filename = f'{ts}-{uuid.uuid4().hex}'

                    writervideo = animation.FFMpegWriter(fps=5)
                    anim.save(f'media/videos/{filename}.mp4', writer=writervideo)
                    plt.close(fig)
                # plt.show()
            if self.render_mode == 'static':
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.set_aspect('equal')
                fig, ax = self.renderer.static_plot(fig, ax)
                if self.save_anim:
                    filename = self.filename
                    if filename is None:
                        ts = datetime.datetime.now().strftime("%m-%d %H-%M-%S")
                        filename = f'{ts}-{uuid.uuid4().hex}'

                    plt.savefig(f'media/static-plots/{filename}.png')
                    plt.close(fig)

        else:
            fig, (ax1, ax2) = plt.subplots(figsize=(9, 9), nrows=2, ncols=1,
                                           height_ratios=[0.7, 0.3])
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
            "robot rad": np.array([self.robot.get_radius()], dtype=float),
            "human pos": np.array(self.human.get_position(), dtype=float),
            "human vel": np.array(self.human.get_velocity(),  dtype=float),
            "human rad": np.array([self.human.get_radius()], dtype=float)
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
        # self.robot.set_position(self.hallway_length * 0.02, 3.0)

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


class Overtaking(HallwayScene):
    """
    The human and the robot start at the same end of the hallway and have to move 
    towards the other end. The robot wants the human to move downwards. 
    """

    def __get_obs(self):
        obs = {
            "robot pos": np.array(self.robot.get_position(), dtype=float),
            "robot vel": np.array(self.robot.get_velocity(), dtype=float),
            "robot rad": np.array([self.robot.get_radius()], dtype=float),
            "human pos": np.array(self.human.get_position(), dtype=float),
            "human vel": np.array(self.human.get_velocity(),  dtype=float),
            "human rad": np.array([self.human.get_radius()], dtype=float)
        }

        return obs

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)

        # Reset the goal position
        self.human.set_goal(0, self.hallway_width / 2)
        self.robot.set_goal(0, self.hallway_width / 2)

        # Reset the position of human
        human_pos = self.human_start_box.sample()
        self.human.set_position(human_pos[0], human_pos[1])
        # self.human.set_position(self.hallway_length * 0.98, 2.5) # Debugging

        # Reset the position of the robot
        robot_pos = self.robot_start_box.sample()
        self.robot.set_position(robot_pos[0], robot_pos[1])
        # self.robot.set_position(self.hallway_length * 0.02, 3.0)

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

    def configure(self, config: Dict, save_anim: bool, render_mode: str | None = "human"):
        super().configure(config, save_anim, render_mode)

        radius = self.human.radius

        # Case 1: Human above robot, human ahead of robot
        human_low = np.array([0.88 * self.hallway_length - 5 * radius, self.hallway_width * 0.5])
        human_high = np.array([0.88 * self.hallway_length - radius, self.hallway_width * 0.7])
        delta = np.array([0.12 * self.hallway_length, -0.3 * self.hallway_width])

        # Case 2: Human above robot, human behind robot
        # human_low = np.array([self.hallway_length - 5 * radius, self.hallway_width * 0.5])
        # human_high = np.array([self.hallway_length - radius, self.hallway_width * 0.7])
        # delta = np.array([-0.12 * self.hallway_length, -0.3 * self.hallway_width])

        # # Case 3: Human below robot, human ahead of robot
        # human_low = np.array([0.88 * self.hallway_length - 5 * radius, self.hallway_width * 0.4])
        # human_high = np.array([0.88 * self.hallway_length - radius, self.hallway_width * 0.6])
        # delta = np.array([0.12 * self.hallway_length, 0.3 * self.hallway_width])

        # Case 4: Human below robot, human behind robot
        # human_low = np.array([self.hallway_length - 5 * radius, self.hallway_width * 0.4])
        # human_high = np.array([self.hallway_length - radius, self.hallway_width * 0.6])
        # delta = np.array([-0.12 * self.hallway_length, 0.3 * self.hallway_width])

        robot_low = human_low + delta
        robot_high = human_high + delta

        self.create_agent_start_box('human', human_low, human_high)
        self.create_agent_start_box('robot', robot_low, robot_high)

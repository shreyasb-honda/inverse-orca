import matplotlib.pyplot as plt
from sim.renderer import Renderer

def test():
    fig, ax = plt.subplots(layout='tight', figsize=(9, 6))
    ax.set_aspect('equal')

    hallway_dimensions = {"length": 15, "width": 4}
    robot_parameters = {"radius": 0.3, "color": 'tab:blue'}
    human_parameters = {"radius": 0.3, "color": 'tab:red'}
    time_step = 0.25

    renderer = Renderer(hallway_dimensions, robot_parameters, human_parameters, time_step)

    robot_pos = (2.4, 1.4)
    human_pos = (13.2, 2.2)
    # ax = renderer.render_frame(robot_pos, human_pos, ax)

    num_frames = 55
    speed_x = 1.0
    speed_y = 0.08

    observations = []

    for frame in range(num_frames):
        robot_pos_x = robot_pos[0] + frame * speed_x * time_step
        robot_pos_y = robot_pos[1] - frame * speed_y * time_step
        human_pos_x = human_pos[0] - frame * speed_x * time_step
        human_pos_y = human_pos[1] + frame * speed_y * time_step
        observation = {
            "robot pos": (robot_pos_x, robot_pos_y),
            "human pos": (human_pos_x, human_pos_y)
        }
        observations.append(observation)
    
    renderer.set_observations(observations, goal_frame=30)
    renderer.set_goal_params(0.2 * hallway_dimensions['length'], 
                              0.3 * hallway_dimensions['width'])
    renderer.animate(fig, ax)


def test_debug_mode():
    # Tests the debug mode
    pass

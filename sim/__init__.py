"""
Registers the custom hallway scene environment in gymnasium
"""

from gymnasium.envs.registration import register

register(
    id='HallwayScene-v0',
    entry_point='sim.env:HallwayScene',
)

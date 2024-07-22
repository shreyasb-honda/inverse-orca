"""
Registers the custom hallway scene environment in gymnasium
"""

from gymnasium.envs.registration import register

register(
    id='HallwayScene-v0',
    entry_point='sim.env:HallwayScene',
)

register(
    id='OvertakingScene-v0',
    entry_point='sim.env:Overtaking'
)

register(
    id='FixedHumanScene-v0',
    entry_point='sim.env:FixedHuman'
)

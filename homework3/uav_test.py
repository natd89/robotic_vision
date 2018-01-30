#!/usr/bin/env python

import numpy as np

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors

env = Holodeck.make("UrbanCity")

for i in range(10):
    env.reset()

    # This command tells the UAV to not roll or pitch, but to constantly yaw left at 10m altitude.
    command = np.array([0, 0, 1, 10])
    for _ in range(300):
        if _ == 0:
            state, reward, terminal, _ = env.step(command)

        # To access specific sensor data:
        pixels = state[Sensors.PRIMARY_PLAYER_CAMERA]
        velocity = state[Sensors.VELOCITY_SENSOR]
        # For a full list of sensors the UAV has, view the README

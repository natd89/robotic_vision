#!/usr/bin/env python

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
from std_msgs.msg import Float64MultiArray
import pygame
import cv2
import numpy as np


def get_command(command,done):

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            # Figure out if it was an arrow key. If so
                # adjust speed.
            if event.key == pygame.K_r:
                command = command + np.array([0,0,0,.5])
            elif event.key == pygame.K_f:
                command = command + np.array([0,0,0,-.5])
            elif event.key == pygame.K_e:
                command = command + np.array([0,0,.1,0])
            elif event.key == pygame.K_d:
                command = command + np.array([0,0,-.1,0])
            elif event.key == pygame.K_w:
                command = command + np.array([0,.1,0,0])
            elif event.key == pygame.K_s:
                command = command + np.array([0,-.1,0,0])
            elif event.key == pygame.K_q:
                command = command + np.array([.1,0,0,0])
            elif event.key == pygame.K_a:
                command = command + np.array([-.1,0,0,0])
    return command


if __name__=='__main__':

    env = Holodeck.make("UrbanCity")
    
    # Setup
    pygame.init()
    size = [200, 200]
    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("My Game")

    # Loop until the user clicks the close button.
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()
 
    command = np.array([0,0,0,0])

    # Hide the mouse cursor
    pygame.mouse.set_visible(0)

    while not done:
        
        command = get_command(command,done)

        state, reward, terminal, _ = env.step(command)
        
        pixels = state[Sensors.PRIMARY_PLAYER_CAMERA]
        velocity = state[Sensors.VELOCITY_SENSOR]

        gray_image = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image,50,300)
        cv2.imshow('uav_filter',gray_image)
        cv2.imshow('uav_canny',edges)
        cv2.waitKey(1)


    cv2.destroyAllWindows()



















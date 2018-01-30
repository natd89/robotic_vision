#!/usr/bin/env python

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
import pygame
import cv2
import pdb 
import numpy as np


def get_command(command,done):

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            # Figure out if it was an arrow key. If so
                # adjust speed.
            if event.key == pygame.K_e:
                command = command + np.array([0,0,0,.5])
            elif event.key == pygame.K_d:
                command = command + np.array([0,0,0,-.5])
            elif event.key == pygame.K_s:
                command = command + np.array([0,0,.1,0])
            elif event.key == pygame.K_f:
                command = command + np.array([0,0,-.1,0])
            elif event.key == pygame.K_DOWN:
                command = command + np.array([0,.1,0,0])
            elif event.key == pygame.K_UP:
                command = command + np.array([0,-.1,0,0])
            elif event.key == pygame.K_LEFT:
                command = command + np.array([.1,0,0,0])
            elif event.key == pygame.K_RIGHT:
                command = command + np.array([-.1,0,0,0])
            elif event.key == pygame.K_ESCAPE:
                done = True
    return command, done


if __name__=='__main__':

    env = Holodeck.make("UrbanCity")
    
    # Setup
    pygame.init()
    size = [200, 200]
    screen = pygame.display.set_mode(size)
    
    # states for plotting
    t = []
    i = 0 # counter for time
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []

    pygame.display.set_caption("My Game")

    # Loop until the user clicks the close button.
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()
 
    command = np.array([0,0,0,0])

    # Hide the mouse cursor
    pygame.mouse.set_visible(0)

    while not done:
        t.append(i)
        command, done = get_command(command,done)

        state, reward, terminal, _ = env.step(command)
        
        pixels = state[Sensors.PRIMARY_PLAYER_CAMERA]
        velocity = state[Sensors.VELOCITY_SENSOR]
        location = state[Sensors.LOCATION_SENSOR]
        # print(location)
        # pdb.set_trace()
        gray_image = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image,50,300)
        cv2.imshow('uav_filter',gray_image)
        cv2.imshow('uav_canny',edges)
        cv2.waitKey(1)
        x.append(location[0][0])
        y.append(location[1][0])
        z.append(location[2][0])
        vx.append(velocity[0][0])
        vy.append(velocity[1][0])
        vz.append(velocity[2][0])       
        i += 1
    
    
    plt.figure(1)
    plt.plot(t,x,'r',t,y,'b',t,z,'g')
    plt.legend(('x','y','z'))
    plt.figure(2)
    plt.plot(t,vx,'r',t,vy,'b',t,vz,'g')
    plt.legend(('vx','vy','vz'))
    plt.show()

    cv2.destroyAllWindows()















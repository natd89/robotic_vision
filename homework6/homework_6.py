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
from pdb import set_trace as pause
import time

def get_command(command,done):

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            # Figure out if it was an arrow key. If so
                # adjust speed.
            if event.key == pygame.K_e:
                command = command + np.array([0,0,0,1])
            elif event.key == pygame.K_d:
                command = command + np.array([0,0,0,-1])
            elif event.key == pygame.K_s:
                command = command + np.array([0,0,.3,0])
            elif event.key == pygame.K_f:
                command = command + np.array([0,0,-.3,0])
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
    pygame.mouse.set_visible(1)

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 10 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    command, done = get_command(command,done)

    state, reward, terminal, _ = env.step(command)
    
    pixels_old = state[Sensors.PRIMARY_PLAYER_CAMERA]
    
    gray_old = cv2.cvtColor(pixels_old, cv2.COLOR_BGR2GRAY)

    p_old = cv2.goodFeaturesToTrack(gray_old, mask=None, **feature_params)


    while not done:
        t.append(i)
        command, done = get_command(command,done)

        state, reward, terminal, _ = env.step(command)
        
        pixels_new = state[Sensors.PRIMARY_PLAYER_CAMERA]
        velocity = state[Sensors.VELOCITY_SENSOR]
        location = state[Sensors.LOCATION_SENSOR]
        # print(location)
        # pdb.set_trace()
        gray_new = cv2.cvtColor(pixels_new, cv2.COLOR_BGR2GRAY)

        p_new, st, err = cv2.calcOpticalFlowPyrLK(gray_old, gray_new, p_old, None, **lk_params)

        pts1 = p_old
        pts2 = p_new

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # if i%5==0:
        #     print(F,'\n')

        recover = cv2.recoverPose(F, pts1, pts2)
        R = recover[1]
        T = recover[2]

        if i%20==0:
            print('R = ', np.round(R,2), end='\n\n')
            print('T = ', np.round(T,2), end='\n\n')

        p_old = cv2.goodFeaturesToTrack(gray_new, mask=None, **feature_params)
        gray_old = gray_new

        for j in range(len(p_new)):
            cv2.circle(pixels_new, tuple(p_new[j,0,:]),7,(0,255,0),1)

        cv2.imshow('new_frame',pixels_new)
        cv2.waitKey(1)

        # x.append(location[0][0])
        # y.append(location[1][0])
        # z.append(location[2][0])

        i += 1

    
    # plt.figure(1)
    # plt.plot(t,x,'r',t,y,'b',t,z,'g')
    # plt.legend(('x','y','z'))
    # plt.show()

    cv2.destroyAllWindows()















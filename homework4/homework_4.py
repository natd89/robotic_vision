#!/usr/bin/env python

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
import matplotlib.pyplot as plt
from copy import deepcopy 
import pygame
import cv2
from pdb import set_trace as pause
import numpy as np


def get_command(command,done):

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            # Figure out if it was an arrow key. If so
                # adjust speed.
            if event.key == pygame.K_e:
                command = command + np.array([0,0,0,5])
            elif event.key == pygame.K_d:
                command = command + np.array([0,0,0,-5])
            elif event.key == pygame.K_s:
                command = command + np.array([0,0,1,0])
            elif event.key == pygame.K_f:
                command = command + np.array([0,0,-1,0])
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

    
    pygame.display.set_caption("My Game")

    # Loop until the user clicks the close button.
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()
 
    command = np.array([0,0,0,0])

    state, reward, terminal, _ = env.step(command)
    
    pixels_prev = state[Sensors.PRIMARY_PLAYER_CAMERA]
    gray_prev = cv2.cvtColor(pixels_prev, cv2.COLOR_BGRA2GRAY)

    # Hide the mouse cursor
    pygame.mouse.set_visible(1)

    p0 = np.zeros((16*16,1,2))
    k = 0
    for j in range(16):
        for i in range(16):
            p0[k,0,0] = i*32 + 16
            p0[k,0,1] = j*32 + 16
            k+=1
    p0 = np.float32(p0)

    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create a mask image for drawing purposes
    mask = np.zeros_like(pixels_prev)

 
    while not done:

        command, done = get_command(command,done)

        state, reward, terminal, _ = env.step(command)
        
        pixels_cur = state[Sensors.PRIMARY_PLAYER_CAMERA]
        velocity = state[Sensors.VELOCITY_SENSOR]
        location = state[Sensors.LOCATION_SENSOR]

        # convert image to grayscale
        gray_cur = cv2.cvtColor(pixels_cur, cv2.COLOR_BGRA2GRAY)
            
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_cur, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), [0,255,0], 1)
            frame = cv2.circle(pixels_cur,(a,b),2,[0,255,0],-1)
        img = cv2.add(frame,mask)

        cv2.imshow('uav optical flow',img)
        cv2.waitKey(25)

        mask = np.zeros_like(pixels_cur)

        gray_prev = deepcopy(gray_cur)
        
    cv2.destroyAllWindows()















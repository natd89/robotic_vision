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


def get_command(command,done,flag):

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            # Figure out if it was an arrow key. If so
                # adjust speed.
            if event.key == pygame.K_s:
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
            elif event.key == pygame.K_RETURN:
                flag = 3
                print('flag = ', flag) 

    return command, done, flag


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

    n = 16 # number of points across and down 

    p0 = np.zeros((n*n,1,2))
    k = 0
    for j in range(n):
        for i in range(n):
            p0[k,0,0] = i*2*n + n
            p0[k,0,1] = j*2*n + n
            k+=1
    p0 = np.float32(p0)

    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create a mask image for drawing purposes
    mask = np.zeros_like(pixels_prev)
    
    flag = 0
    h_c = 4
    psi_c = .9
    psi = 0
    kp_psi = 5
    kp_theta = 0.01
    k = 0
    x_left = []
    x_right = []
    x_mid_left = []
    x_mid_right = []

    while not done:
        
        if flag==0:
            state, reward, terminal, _ = env.step(command)           
            location = state[Sensors.LOCATION_SENSOR]
            print('setting height to hc...')
            while location[2]<h_c*100:
                command = command + np.array([0,0,0,.1])
                state, reward, terminal, _ = env.step(command)           

            while (psi_c-psi) > 0.001: 
                state, reward, terminal, _ = env.step(command)           
                rotation = state[Sensors.ORIENTATION_SENSOR]
                theta = -np.arcsin(rotation[0][2])
                psi = np.arcsin(rotation[1][0]/np.cos(theta))
                command[2] = kp_psi*(psi_c-psi)
            flag = 2

        elif flag == 2 or flag == 3:
            
            command, done, flag = get_command(command,done,flag)
            state, reward, terminal, _ = env.step(command)
            pixels_cur = state[Sensors.PRIMARY_PLAYER_CAMERA]
            velocity = state[Sensors.VELOCITY_SENSOR]
            location = state[Sensors.LOCATION_SENSOR]
            rotation = state[Sensors.ORIENTATION_SENSOR]
            theta = -np.arcsin(rotation[0][2])
            psi = np.arcsin(rotation[1][0]/np.cos(theta))
            
            if flag==2:
                command[2] = kp_psi*(psi_c-psi)


            if k%2==0:

                # convert image to grayscale
                gray_cur = cv2.cvtColor(pixels_cur, cv2.COLOR_BGRA2GRAY)
                
                p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_cur, p0, None, **lk_params)
            
                for i in range(n*n):
                    if p0[i,0,0]<170 and p0[i,0,1]>256-200/2 and p0[i,0,1]<256+200/2:
                        x_left.append(p1[i,0,0] - p0[i,0,0])
                    elif p0[i,0,0]>340 and p0[i,0,1]>256-200/2 and p0[i,0,1]<256+200/2:
                        x_right.append(p1[i,0,0] - p0[i,0,0])
                    elif p0[i,0,0]>256-128 and p0[i,0,0]<256 and p0[i,0,1]>256-128 and p0[i,0,1]<256+128:
                        x_mid_left.append(p1[i,0,1] - p0[i,0,1])
                    elif p0[i,0,0]>256 and p0[i,0,0]<256+128 and p0[i,0,1]>256-128 and p0[i,0,1]<256+128:
                        x_mid_right.append(p1[i,0,1] - p0[i,0,1])

                sum_mid_x = np.sum(np.abs(x_mid_left)+np.abs(x_mid_right))
                # print(sum_mid_x)
                sum_x = np.sum(np.average(x_left)+np.average(x_right))
                
                if flag==3:
                    command[0] = kp_theta*sum_x 
                    
                    if command[0]>10*np.pi/180.:
                        command[0] = 10*np.pi/180.
                    if command[0]<-10*np.pi/180.:
                        command[0] = -10*np.pi/180.

                    if sum_mid_x > 70:
                        print('watch out!!')
                        print(np.sum(np.abs(x_mid_left)) - np.sum(np.abs(x_mid_right)))
                        if (np.sum(np.abs(x_mid_left)) - np.sum(np.abs(x_mid_right))) > 20:
                            command[2] = kp_psi*((psi_c-.2)-psi)
                            command[1] = .1
                        elif (np.sum(np.abs(x_mid_left)) - np.sum(np.abs(x_mid_right))) < -20:
                            command[2] = kp_psi*((psi_c+2)-psi)
                            command[1] = .1
                        else:
                            command[1] = -.2
                    else:
                        command[2] = kp_psi*(psi_c-psi)
                        command[1] = -0.2


                # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]
            
                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), [0,255,0], 1)
                    # frame = cv2.circle(pixels_cur,(a,b),2,[0,255,0],-1)
                img = cv2.add(pixels_cur,mask)
                
                cv2.imshow('uav optical flow',img)
                cv2.waitKey(25)
                
                mask = np.zeros_like(pixels_cur)
                
                gray_prev = deepcopy(gray_cur)

            x_left = []
            x_right = []
            x_mid_left = []
            x_mid_right = []
            k+=1
    cv2.destroyAllWindows()















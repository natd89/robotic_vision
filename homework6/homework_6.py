#!/usr/bin/env python

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
import matplotlib.pyplot as plt
import pygame
import cv2
import pdb 
import numpy as np
from pdb import set_trace as pause
import pyqtgraph as pg
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

    x_est = []
    y_est = []
    z_est = []

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
    feature_params = dict( maxCorners = 100,
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

    R_cam_uav = np.array([[0, 0, -1],[1, 0, 0],[0, -1, 0]])


    # plotting functions for pyqtgraph
    app = pg.QtGui.QApplication([])
    ekfplotwin = pg.GraphicsWindow(size=(800,400))
    ekfplotwin.setWindowTitle('Position') 
    ekfplotwin.setInteractive(True)
    plt1 = ekfplotwin.addPlot(2,1)
    plt2 = ekfplotwin.addPlot(2,2)
    xycurve = plt1.plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.SolidLine))
    xycurve_est = plt1.plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine))
    ztcurve = plt2.plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.SolidLine))
    ztcurve_est = plt2.plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine))
    

    while not done:
        t.append(i)
        command, done = get_command(command,done)

        state, reward, terminal, _ = env.step(command)
        
        pixels_new = state[Sensors.PRIMARY_PLAYER_CAMERA]
        velocity = state[Sensors.VELOCITY_SENSOR]
        rotation = state[Sensors.ORIENTATION_SENSOR]
        location = state[Sensors.LOCATION_SENSOR]/100. # convert to meters
        # print(location)
        # pdb.set_trace()
        gray_new = cv2.cvtColor(pixels_new, cv2.COLOR_BGR2GRAY)

        p_new, st, err = cv2.calcOpticalFlowPyrLK(gray_old, gray_new, p_old, None, **lk_params)

        pts1 = p_old[st==1]
        pts2 = p_new[st==1]

        # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, param1=0.1)
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=0.1)

        # if i%5==0:
        #     print(F,'\n')

        ret, R, T, mask = cv2.recoverPose(E, pts1, pts2)
       
        T_hat = np.array([[0,-T[2][0],T[1][0]],[T[2][0],0,-T[0][0]],[-T[1][0],T[0][0],0]])

        x2 = np.append(pts2[0,:],1).reshape(3,1)
        x1 = np.append(pts1[0,:],1).reshape(3,1)
        
        # print(np.matmul(x2.T,np.matmul(E,x1)))
        print(np.matmul(x2.T,np.matmul(T_hat,np.matmul(R,x1))))

        T = np.matmul(R_cam_uav, T)

        T = np.matmul(rotation, T)

        # if i%20==0:
        #     print('R = ', np.round(R,2), end='\n\n')
        #     print('T_ori = ', np.round(T,2), end='\n\n')
        #     print('T = ', np.round(T,2), end='\n\n')

        p_old = cv2.goodFeaturesToTrack(gray_new, mask=None, **feature_params)
        gray_old = gray_new

        # for j in range(len(pts2)):
        #     cv2.circle(pixels_new, tuple(pts2[j,:]),7,(0,255,0),1)

        # cv2.imshow('new_frame',pixels_new)
        # cv2.waitKey(1)

        x.append(location[0][0])
        y.append(location[1][0])
        z.append(location[2][0])

        if i==0:
            x_est.append(location[0][0])
            y_est.append(location[1][0])
            z_est.append(location[2][0]/10.)
            

        if not i==0:
            x_est.append(T[0]+x_est[i-1])
            y_est.append(T[1]+y_est[i-1])
            z_est.append(T[2]/10.+z_est[i-1])
        
        xycurve.setData(np.array(x),np.array(y))
        ztcurve.setData(np.array(t),np.array(z))

        xycurve_est.setData(np.array(x_est),np.array(y_est))
        ztcurve_est.setData(np.array(t),np.array(z_est))

        app.processEvents()

        i += 1

        # plotWidget.             
        

    # plt.figure(1)
    # plt.plot(t,x,'r',t,y,'b',t,z,'g')
    # plt.legend(('x','y','z'))
    # plt.show()

    cv2.destroyAllWindows()















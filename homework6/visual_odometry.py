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

    env = Holodeck.make("RedwoodForest")
    
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
    feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 10 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    command, done = get_command(command,done)

    state, reward, terminal, _ = env.step(command)
    
    location = state[Sensors.LOCATION_SENSOR]/100. # convert to meters
    location[0] = -location[0] # flip the x axis to make it a right handed coordinate frame

    pixels_old = state[Sensors.PRIMARY_PLAYER_CAMERA]
    
    gray_old = cv2.cvtColor(pixels_old, cv2.COLOR_BGR2GRAY)

    p_old = cv2.goodFeaturesToTrack(gray_old, mask=None, **feature_params)

    R_cam_uav = np.array([[0, 0, -1],[1, 0, 0],[0, -1, 0]])

    pos_est = location

    # plotting functions for pyqtgraph
    app = pg.QtGui.QApplication([])
    ekfplotwin = pg.GraphicsWindow(size=(1200,400))
    ekfplotwin.setWindowTitle('Position') 
    ekfplotwin.setInteractive(True)
    plt1 = ekfplotwin.addPlot(1,1)
    plt1.showGrid(x=True, y=True)
    plt2 = ekfplotwin.addPlot(1,2)
    plt2.showGrid(x=True, y=True)
    plt3 = ekfplotwin.addPlot(1,3)
    plt3.showGrid(x=True, y=True)
    xtcurve = plt1.plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.SolidLine))
    xtcurve_est = plt1.plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine))
    ytcurve = plt2.plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.SolidLine))
    ytcurve_est = plt2.plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine))
    ztcurve = plt3.plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.SolidLine))
    ztcurve_est = plt3.plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine))
    ekfplotwin.resize(1200,400)

    while not done:
        t.append(i)
        command, done = get_command(command,done)

        state, reward, terminal, _ = env.step(command)
        
        pixels_new = state[Sensors.PRIMARY_PLAYER_CAMERA]
        velocity = state[Sensors.VELOCITY_SENSOR]/100.
        rotation = state[Sensors.ORIENTATION_SENSOR]
        location = state[Sensors.LOCATION_SENSOR]/100. # convert to meters
        location[0] = -location[0] # flip the x axis to make it a right handed coordinate frame

        gray_new = cv2.cvtColor(pixels_new, cv2.COLOR_BGR2GRAY)

        p_new, st, err = cv2.calcOpticalFlowPyrLK(gray_old, gray_new, p_old, None, **lk_params)

        pts1 = p_old[st==1]
        pts2 = p_new[st==1]

        # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, param1=0.1)
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=0.1)

        R1, R2, T = cv2.decomposeEssentialMat(E)
        
        for rot in [R1, R2]:
            if np.trace(rot) > 2.5:
                R = rot 

        # T_hat = np.array([[0,-T[2][0],T[1][0]],[T[2][0],0,-T[0][0]],[-T[1][0],T[0][0],0]])

        # x2 = np.append(pts2[0,:],1).reshape(3,1)
        # x1 = np.append(pts1[0,:],1).reshape(3,1)
        
        # print(np.matmul(x2.T,np.matmul(E,x1)))
        # print(np.matmul(x2.T,np.matmul(T_hat,np.matmul(R,x1))))

        pos_est = pos_est + np.matmul(rotation ,np.matmul( R_cam_uav,T*np.linalg.norm(velocity)*(1/30.)))

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

        x_est.append(pos_est[0][0])
        y_est.append(pos_est[1][0])
        z_est.append(pos_est[2][0])
        

        xtcurve.setData(np.array(t),np.array(x))
        ytcurve.setData(np.array(t),np.array(y))
        ztcurve.setData(np.array(t),np.array(z))

        xtcurve_est.setData(np.array(t),np.array(x_est))
        ytcurve_est.setData(np.array(t),np.array(y_est))
        ztcurve_est.setData(np.array(t),np.array(z_est))

        app.processEvents()

        i += 1

        # plotWidget.             
        

    # plt.figure(1)
    # plt.plot(t,x,'r',t,y,'b',t,z,'g')
    # plt.legend(('x','y','z'))
    # plt.show()

    cv2.destroyAllWindows()















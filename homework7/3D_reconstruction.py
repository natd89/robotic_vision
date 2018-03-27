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

    # env = Holodeck.make("RedwoodForest")
    env = Holodeck.make("UrbanCity")
    
    # Setup
    pygame.init()
    size = [200, 200]
    screen = pygame.display.set_mode(size)
    
    # states for plotting
    time = []
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
    feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.01 ,
                       minDistance = 7,
                       blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    command, done = get_command(command,done)

    state, reward, terminal, _ = env.step(command)

    rotation_old = state[Sensors.ORIENTATION_SENSOR]    
    location_old = state[Sensors.LOCATION_SENSOR]/100. # convert to meters
    location_old[0] = -location_old[0] # flip the x axis to make it a right handed coordinate frame

    pixels_old = state[Sensors.PRIMARY_PLAYER_CAMERA]
    
    R_old = np.concatenate((rotation_old,location_old),axis=1)

    gray_old = cv2.cvtColor(pixels_old, cv2.COLOR_BGR2GRAY)

    p_old = cv2.goodFeaturesToTrack(gray_old, mask=None, **feature_params)

    # R_cam_uav = np.array([[0, 0, -1],[1, 0, 0],[0, -1, 0]])
    R_cam_uav = np.array([[0, 0, -1, 0],[1, 0, 0, 0],[0, -1, 0, 0],[0, 0, 0, 1]])

    # pos_est = location
    pos_est = np.array([[location_old[0][0]],[location_old[1][0]], [location_old[2][0]], [1]])

    # plotting functions for pyqtgraph
    # app = pg.QtGui.QApplication([])
    # ekfplotwin = pg.GraphicsWindow(size=(400,400))
    # ekfplotwin.setWindowTitle('Position') 
    # ekfplotwin.setInteractive(True)
    # plt1 = ekfplotwin.addPlot(1,1)
    # plt1.showGrid(x=True, y=True)
    # plt2 = ekfplotwin.addPlot(1,2)
    # plt2.showGrid(x=True, y=True)
    # plt3 = ekfplotwin.addPlot(1,3)
    # plt3.showGrid(x=True, y=True)
    # xtcurve = plt1.plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.SolidLine))
    # xtcurve_est = plt1.plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine))
    # ytcurve = plt2.plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.SolidLine))
    # ytcurve_est = plt2.plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine))
    # ztcurve = plt3.plot(pen=None, symbol='o')# plt3.plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.SolidLine))
    # ztcurve_est = plt3.plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine))
    # ekfplotwin.resize(1200,400)

    Ck = np.identity(4)

    grid_mask = np.ones((800,800))*255
    
    N = 5 # remembering factor
    M = 255 # forgetting factor

    while not done:

        command, done = get_command(command,done)

        state, reward, terminal, _ = env.step(command)
        
        pixels_new = state[Sensors.PRIMARY_PLAYER_CAMERA]
        velocity_new = state[Sensors.VELOCITY_SENSOR]/100.
        rotation_new = state[Sensors.ORIENTATION_SENSOR]
        location_new = state[Sensors.LOCATION_SENSOR]/100. # convert to meters
        location_new[0] = -location_new[0] # flip the x axis to make it a right handed coordinate frame

        R_new = np.concatenate((rotation_new,location_new),axis=1)

        if i%10==0:

            time.append(i)

            gray_new = cv2.cvtColor(pixels_new, cv2.COLOR_BGR2GRAY)
            p_new, st, err = cv2.calcOpticalFlowPyrLK(gray_old, gray_new, p_old, None, **lk_params)
            
            pold = p_old[st==1]
            pnew = p_new[st==1]
            
            pold = np.squeeze(pold).T
            pnew = np.squeeze(pnew).T

            points_4D = cv2.triangulatePoints(R_old,R_new,pold-256,pnew-256)

            # update grid using 4D points
            for j in range(len(points_4D[0])):
                if (int(points_4D[0,j]/points_4D[3,j])+400)<799 and (int(points_4D[1,j]/points_4D[3,j])+400)<799:
                    if (int(points_4D[0,j]/points_4D[3,j])+400)>0 and (int(points_4D[1,j]/points_4D[3,j])+400)>0:
                        if np.abs(points_4D[2,j]/points_4D[3,j])>17:
                            grid_mask[int(points_4D[0,j]/points_4D[3,j])+int(location_new[0])+400,int(points_4D[1,j]/points_4D[3,j])+int(location_new[1])+400]= 0 
                            # print([int(points_4D[0,j]/points_4D[3,j])+400,int(points_4D[1,j]/points_4D[3,j])+400])
                            if grid_mask[int(points_4D[0,j]/points_4D[3,j])+400,int(points_4D[1,j]/points_4D[3,j])+400]<0:
                                grid_mask[int(points_4D[0,j]/points_4D[3,j])+400,int(points_4D[1,j]/points_4D[3,j])+400]=0

            p_old = p_new # cv2.goodFeaturesToTrack(gray_new, mask=None, **feature_params)
            gray_old = gray_new
            
            if len(pnew.T)<50:
                p_old = cv2.goodFeaturesToTrack(gray_new, mask=None, **feature_params)
                

            for j in range(len(pnew.T)):
                cv2.circle(pixels_new, tuple(pnew.T[j,:]),7,(0,255,0),1)
                
            cv2.imshow('new_frame',pixels_new)
            cv2.imshow('grid',grid_mask)
            cv2.waitKey(1)        
            
            # x.append(location_new[0][0])
            # y.append(location_new[1][0])
            # z.append(location_new[2][0])
            
            # # x_est.append(pos_est[0][0])
            # # y_est.append(pos_est[1][0])
            # # z_est.append(pos_est[2][0])
            
            # xtcurve.setData(np.array(y),np.array(x))
            # ytcurve.setData(np.array(time),np.array(y))
            # # ztcurve.setData(np.array(time),np.array(z))
            # ztcurve.setData(np.array(points_4D[0,:]/points_4D[3,:]),points_4D[1,:]/points_4D[3,:])
            
            # # xtcurve_est.setData(np.array(time),np.array(x_est))
            # # ytcurve_est.setData(np.array(time),np.array(y_est))
            # # ztcurve_est.setData(np.array(time),np.array(z_est))
            
            # app.processEvents()

            
        i += 1

        rotation_old = rotation_new
        location_old = location_new
        R_old = R_new

        # grid_mask = grid_mask + 1
        # grid_mask[grid_mask>255]=255
        
    cv2.destroyAllWindows()















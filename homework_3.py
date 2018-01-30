#!/usr/bin/env python

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
from std_msgs.msg import Float64MultiArray
import cv2
import numpy as np
import rospy

class fly_uav():
    
    def __init__(self):
        self.command = np.array([])
        self.up    = None
        self.down  = None
        self.left  = None
        self.right = None
        rospy.Subscriber('uav_command', Float64MultiArray, self.get_key)

    def get_key(self,msg):
        self.command = self.command + msg.data


if __name__=='__main__':

    rospy.init_node('uav_test',anonymous=True)

    env = Holodeck.make("UrbanCity")

    while not rospy.is_shutdown():

        # Setup
        pygame.init()
 
        # Set the width and height of the screen [width,height]
        size = [700, 500]
        screen = pygame.display.set_mode(size)
        
        pygame.display.set_caption("My Game")
        
        # Loop until the user clicks the close button.
        done = False
 
        while not done:
            # --- Event Processing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    # User pressed down on a key
 
                elif event.type == pygame.KEYDOWN:
                    # Figure out if it was an arrow key. If so
                    # adjust speed.
                    if event.key == pygame.K_r:
                        x_speed = -3
                    elif event.key == pygame.K_v:
                        x_speed = 3
                    elif event.key == pygame.K_e:
                        y_speed = -3
                    elif event.key == pygame.K_c:
                        y_speed = 3
                    elif event.key == pygame.K_w:
                        y_speed = -3
                    elif event.key == pygame.K_x:
                        y_speed = 3
                    elif event.key == pygame.K_q:
                        y_speed = -3
                    elif event.key == pygame.K_z:
                        y_speed = 3
 
                # User let up on a key
                elif event.type == pygame.KEYUP:
                    # If it is an arrow key, reset vector back to zero
                    if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                        x_speed = 0
                    elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                        y_speed = 0
    
        state, reward, terminal, _ = env.step(self.command)
        
        pixels = state[Sensors.PRIMARY_PLAYER_CAMERA]
        velocity = state[Sensors.VELOCITY_SENSOR]
























import cv2
from pdb import set_trace as pause
import numpy as np

# initialize a video capture object
# cap = cv2.VideoCapture('/home/nmd/Downloads/soccer.mp4')
# cap = cv2.VideoCapture(0)

# read in the first frame
ret, frame = cap.read()

# initializ object for mog
mog = cv2.createBackgroundSubtractorMOG2()

# initialize kernel for erroding function
kernel_e = np.ones((15,15),np.uint8)
kernel_d = np.ones((20,20),np.uint8)

# variable for memory
prev = np.zeros_like(frame)

# run loop that takes image[k] and image[k-1] and finds the difference of the two
while(1):
    
    # read in new frame and convert to gray-scale
    ret, frame_new = cap.read()

    # take the difference of the two images and display them
    image = cv2.absdiff(frame_new, frame)

    # use mog2 to do differencing
    # image = mog.apply(frame_new)
    
    # blur the image to smooth out noise
    image = cv2.filter2D(image,-1, np.ones((5,5),np.float32)/25)

    image = cv2.dilate(image,kernel_d/4.,iterations = 1)

    # convert the difference image color to grayscale and threshold
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    temp = image
    ret,image = cv2.threshold(image,30,255,cv2.THRESH_BINARY)

    # errode and dilate the image to clean up noise
    # image = cv2.erode(image,kernel_e/20.,iterations = 1)
    image = cv2.dilate(image,kernel_d,iterations = 1)
    image = cv2.erode(image,kernel_e,iterations = 1)
    # dilation = cv2.dilate(erosion,kernel_d,iterations = 1)
    # dilation = cv2.dilate(erosion,kernel_d,iterations = 1)
    


    # display the difeerence image
    # cv2.imshow('difference_image', difference)
    cv2.imshow('mog_differencer',image)
    cv2.imshow('difference',temp)
    cv2.waitKey(30)
    # pause()
    # set the new frame equal to the old frame
    frame = frame_new

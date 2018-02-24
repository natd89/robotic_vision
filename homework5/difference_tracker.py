import cv2
import numpy as np

# initialize a video capture object
cap = cv2.VideoCapture('mv2_001.avi')
# cap = cv2.VideoCapture(0)

# read in the first frame and convert to gray-scale
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# initializ object for mog
mog = cv2.createBackgroundSubtractorMOG2()

# initialize kernel for erroding function
kernel_e = np.ones((8,8),np.uint8)
kernel_d = np.ones((15,15),np.uint8)

# run loop that takes image[k] and image[k-1] and finds the difference of the two
while(1):
    
    # read in new frame and convert to gray-scale
    ret, frame_new = cap.read()
    frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)

    # take the difference of the two images and display them
    # difference = cv2.absdiff(frame_new, frame)

    # use mog2 to do differencing
    mog_mask = mog.apply(frame_new)
    
    # blur the image to smooth out noise
    mog_mask = cv2.blur(mog_mask, (2,2))

    # errode and dilate the image to clean up noise
    erosion = cv2.erode(mog_mask,kernel_e,iterations = 1)
    dilation = cv2.dilate(erosion,kernel_d,iterations = 1)
    # erosion = cv2.erode(dilation,kernel_e,iterations = 1)
    # dilation = cv2.dilate(erosion,kernel_d,iterations = 1)
    
    ret,thresh1 = cv2.threshold(dilation,50,255,cv2.THRESH_BINARY)

    # display the difeerence image
    # cv2.imshow('difference_image', difference)
    cv2.imshow('mog_differencer',thresh1)
    cv2.imshow('original',frame_new)
    cv2.waitKey(30)

    # set the new frame equal to the old frame
    frame = frame_new

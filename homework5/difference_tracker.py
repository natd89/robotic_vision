import cv2
from pdb import set_trace as pause
import numpy as np

# initialize a video capture object
cap = cv2.VideoCapture('mv2_001.avi')
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


# blob detector params and blob detector object
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Filter by Area.
params.filterByArea = True
params.minArea = 100
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.0
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.00

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)



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
    
    # image = cv2.cvtColor(image, cv2.COLOR_
 
    temp = cv2.bitwise_not(image)

    mask = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)

    # Detect blobs.
    keypoints = detector.detect(cv2.bitwise_not(image))
    
    pause()

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    # display the difeerence image
    # cv2.imshow('difference_image', difference)
    cv2.imshow('blob_detection',im_with_keypoints)
    cv2.imshow('difference',temp)
    cv2.imshow('original',cv2.add(mask,frame_new))
    cv2.waitKey(30)
    # pause()
    # set the new frame equal to the old frame
    frame = frame_new

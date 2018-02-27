import cv2
from pdb import set_trace as pause
import numpy as np

# initialize a video capture object
cap = cv2.VideoCapture('mv2_001.avi')
# cap = cv2.VideoCapture(0)

# read in the first frame
ret, frame = cap.read()

# select an ROI and create roi image
roi = cv2.selectROI(frame)
roi_temp = roi
roi_old = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

# initializ object for mog
# mog = cv2.createBackgroundSubtractorMOG2()

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

# kalman filter initialization for tracker
Ts = 30
kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = 1e+2*np.array([[(Ts**3)/3,0,(Ts**2)/2,0],[0,(Ts**3)/3,0,(Ts**2)/2],[(Ts**2)/2,0,Ts,0],[0,(Ts**2)/2,0,Ts]],np.float32)
kalman.errorCovPost = 0.1*np.eye(4, dtype=np.float32)
kalman.measurementNoiseCov = 1e-5*np.eye(2, dtype=np.float32)
kalman.statePost = np.array([[roi[0] + roi[2]/2], [roi[1] + roi[3]/2],[0.],[0.]], dtype=np.float32)

# run loop that takes image[k] and image[k-1] and finds the difference of the two
while(1):
    
    # read in new frame and convert to gray-scale
    ret, frame_new = cap.read()

    # region of interest from new frame
    roi_new = frame_new[roi_temp[1]:roi_temp[1]+roi_temp[3], roi_temp[0]:roi_temp[0]+roi_temp[2]]

    # take the difference of the two images and display them
    image = cv2.absdiff(frame_new, frame)
    roi_image = cv2.absdiff(roi_new, roi_old)

    # use mog2 to do differencing
    # image = mog.apply(frame_new)
    
    # blur the image to smooth out noise
    image = cv2.filter2D(image,-1, np.ones((5,5),np.float32)/25)
    roi_image = cv2.filter2D(roi_image,-1, np.ones((5,5),np.float32)/25)

    image = cv2.dilate(image,kernel_d/4.,iterations = 1)
    roi_image = cv2.dilate(roi_image,kernel_d/4.,iterations = 1)


    # convert the difference image color to grayscale and threshold
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    temp = image
    roi_image_temp = roi_image

    # threshold the image to create a binary mask
    ret,image = cv2.threshold(image,30,255,cv2.THRESH_BINARY)
    ret,roi_image = cv2.threshold(roi_image,30,255,cv2.THRESH_BINARY)

    # errode and dilate the image to clean up noise
    # image = cv2.erode(image,kernel_e/20.,iterations = 1)
    image = cv2.dilate(image,kernel_d,iterations = 1)
    image = cv2.erode(image,kernel_e,iterations = 1)

    roi_image = cv2.dilate(roi_image,kernel_d,iterations = 1)
    roi_image = cv2.erode(roi_image,kernel_e,iterations = 1)


    # dilation = cv2.dilate(erosion,kernel_d,iterations = 1)
    # dilation = cv2.dilate(erosion,kernel_d,iterations = 1)
    
    # image = cv2.cvtColor(image, cv2.COLOR_
 
    temp = cv2.bitwise_not(image)
    roi_image_temp = cv2.bitwise_not(roi_image)

    mask = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
    roi_mask = cv2.cvtColor(roi_image_temp, cv2.COLOR_GRAY2BGR)

    # Detect blobs.
    keypoints = detector.detect(cv2.bitwise_not(image))
    roi_keypoints = detector.detect(cv2.bitwise_not(roi_image))
    

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    im_roi_with_keypoints = cv2.drawKeypoints(roi_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # predict the next position even if blob is not detected
    predict = kalman.predict()
    print predict
    # pause()
    if roi_keypoints:
        # blob_pos will act as the measurement update
        blob_pos = roi_keypoints[0].pt

        # pause()

         # use kalman correct if blob is found
        update = kalman.correct(np.array([[blob_pos[0]+roi_temp[0]],[blob_pos[1]+roi_temp[1]]],np.float32))
        roi_temp = (int(update[0]-roi[2]/2), int(update[1]-roi[3]/2), roi[2], roi[3])

        top_left = (int(update[0]-int(roi_temp[2]/2)), int(update[1]-int(roi_temp[3]/2)))
        bottom_right = (int(update[0]+int(roi_temp[2]/2)), int(update[1]+int(roi_temp[3]/2)))
        im_with_rectangle = cv2.rectangle(im_roi_with_keypoints, top_left, bottom_right, 255, 2)
        print blob_pos
        cv2.imshow('blob_detection',im_with_rectangle)

    else:
        # pause()   
        top_left = (int(predict[0]-int(roi_temp[2]/2)), int(predict[1]-int(roi_temp[3]/2)))
        bottom_right = (int(predict[0]+int(roi_temp[2]/2)), int(predict[1]+int(roi_temp[3]/2)))
        im_with_rectangle = cv2.rectangle(im_with_keypoints, top_left, bottom_right, 255, 2)

        roi_temp = (int(predict[0]-roi[2]/2), int(predict[1]-roi[3]/2), roi[2], roi[3])
        cv2.imshow('blob_detection',im_with_rectangle)


    # display the difeerence image
    # cv2.imshow('difference_image', difference)
    # cv2.imshow('difference',temp)
    # cv2.imshow('original',cv2.add(mask,frame_new))
    cv2.imshow('roi',roi_image_temp)
    cv2.waitKey(30)
    # pause()
    # set the new frame equal to the old frame
    frame = frame_new
    roi_old = roi_new 

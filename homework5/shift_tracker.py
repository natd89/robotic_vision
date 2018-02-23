import numpy as np
from pdb import set_trace as pause
from scipy.stats import mode
import cv2

cap = cv2.VideoCapture(0)
# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
roi = cv2.selectROI(frame)
track_window = roi
# set up the ROI for tracking
roi = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
most = np.float32(mode(hsv_roi[:,:,0], axis=None))[0][0]
mask = cv2.inRange(hsv_roi, np.array((most-10., 50.,0.)), np.array((most+10.,255.,255.)))
res = cv2.bitwise_and(roi, roi, mask=mask)
cv2.imshow('mask',mask)
cv2.imshow('hsv',res)
cv2.waitKey(30)
# pause()
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
    else:
        break
cv2.destroyAllWindows()
cap.release()

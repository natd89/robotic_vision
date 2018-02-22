#!/usr/bin/env python

import cv2
import numpy as np
from pdb import set_trace as pause


def select_roi(image):
    r = cv2.selectROI(image)
    return r


if __name__=='__main__':

    # cap = cv2.VideoCapture('mv2_001.avi')
    cap = cv2.VideoCapture(0)

    ret, im = cap.read()
    roi = select_roi(im)
    im_crop = im[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    old_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
   
    
    cv2.imshow('ROI', im_crop)

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 1,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0,255,(100,3))

    p_crop = cv2.goodFeaturesToTrack(roi_gray, mask = None, **feature_params)

    p = np.float32(p_crop + [roi[0],roi[1]])
    
    mask = np.zeros_like(im)


    while(1):

        ret, im_new = cap.read()
        new_gray = cv2.cvtColor(im_new, cv2.COLOR_BGR2GRAY)
        p_new, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p, None, **lk_params)
        
        print p_new
        
        pnt1 = (int(p_new[0,0,0])-int(round(roi[2]/2,0)), int(p_new[0,0,1])-int(round(roi[3]/2,0)))
        pnt2 = (int(p_new[0,0,0])+int(round(roi[2]/2,0)), int(p_new[0,0,1])+int(round(roi[3]/2,0)))

        rect = cv2.rectangle(im_new,pnt1, pnt2, [255,0,0], thickness=2)
        # mask_rect = cv2.rectangle(im_new,pnt1, pnt2, [255,0,0], thickness=2)
        # im_rect = cv2.add(mask_rect,im_new)
        cv2.imshow('image',rect)

        mask_rect = np.zeros_like(im)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = new_gray.copy()
        p = p_new        
    















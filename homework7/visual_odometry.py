#!/usr/bin/env python

from pdb import set_trace as pause
from PIL import Image
import numpy as np
import time
import cv2


def import_rotation_images():
    rotation_imgs = []
    for i in range(6):
        rotation_imgs.append(import_img('rotate',i))
    return rotation_imgs


def import_ztranslation_images():
    ztrans_imgs = []
    for i in range(6):
        ztrans_imgs.append(import_img('translatez',i))
    return ztrans_imgs


def import_xtranslation_images():
    xtrans_imgs = []
    for i in range(6):
        xtrans_imgs.append(import_img('translatex',i))
    return xtrans_imgs


def compute_R_and_t(pts1, pts2):
    E, mask = cv2.findEssentialMat(pts1, pts2, focal=256, pp=(256., 256.), method=cv2.RANSAC, prob=0.999, threshold=0.1)
    R1, R2, t = cv2.decomposeEssentialMat(E)
    for rot in [R1, R2]:
        if np.trace(rot) > 2.5:
            R = rot 
    return R,t


def calcOpticalFlow(old, new, p_old):
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p_new, st, err = cv2.calcOpticalFlowPyrLK(old, new, p_old, None, **lk_params)
    return p_new, st, err


def import_img(string,i):
    im = Image.open(string + str(i)+'.JPG')
    return im


if __name__=='__main__':

    img_type = input('x,z, or rot? ')
    if img_type=='x':
        images = import_xtranslation_images()
    elif img_type=='z':
        images = import_ztranslation_images()
    elif img_type=='rot':
        images = import_rotation_images()
    else:
        exit

    im_old = np.array (images[0])
    gray_old = cv2.cvtColor(im_old, cv2.COLOR_BGR2GRAY)
    pause()
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(gray_old, None)
    p_old = np.array([x.pt for x in kp], dtype=np.float32)
    p_old = np.reshape(p_old, (len(p_old),1,2))

    for i in range(len(images)-1):
        
        im_new = np.array(images[i+1], dtype=np.float32)
        gray_new = cv2.cvtColor(im_new, cv2.COLOR_BGR2GRAY)

        p_new, st, err = calcOpticalFlow(gray_old, gray_new, p_old)
        
        pts1 = p_old[st==1]
        pts2 = p_new[st==1]
        
        # find R and t for the essential matrix
        R,t = compute_R_and_t(pts1, pts2)
        t_hat = np.array([[0,-t[2][0],t[1][0]],[t[2][0],0,-t[0][0]],[-t[1][0],t[0][0],0]])
        E = np.matmul(t_hat,R)
      
        x2 = np.append(pts2[0,:],1).reshape(3,1)
        x1 = np.append(pts1[0,:],1).reshape(3,1)
        
        print(np.matmul(x2.T,np.matmul(E,x1)))
        print(np.matmul(x2.T,np.matmul(t_hat,np.matmul(R,x1))))
        
        pos_est = pos_est + np.matmul(rotation ,np.matmul( R_cam_uav,t*np.linalg.norm(velocity)*(1/30.)))
        
        T = np.array([[R[0,0],R[0,1],R[0,2],t[0]*np.linalg.norm(velocity)*(1/30.)],
                      [R[1,0],R[1,1],R[1,2],t[1]*np.linalg.norm(velocity)*(1/30.)],
                      [R[2,0],R[2,1],R[2,2],t[2]*np.linalg.norm(velocity)*(1/30.)],
                      [0,0,0,1]])
        
        Ck = np.matmul(Ck,T)
        
        Ck_R = np.matmul(R_cam_uav,Ck)
        pause()
        
    cv2.destroyAllWindows()















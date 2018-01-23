#!/usr/bin/env python

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(frame,100,100)
    edges_gray = cv2.Canny(gray,100,100)
    blur_color = cv2.blur(frame,(10,10))
    blur_gray = cv2.blur(gray,(10,10))
    # Display the resulting frame
    cv2.imshow('canny_color',edges)
    cv2.imshow('canny_gray',edges_gray)
    cv2.imshow('blur_color',blur_color)
    cv2.imshow('blur_gray',blur_gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

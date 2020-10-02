import numpy as np
import cv2
import sys
from math import sqrt

i = 1
path = "D:\ITU\Final Year Project\Others\BgSub\mog2\mog2"
path1 = "D:\ITU\Final Year Project\Others\BgSub\original\o"
cap = cv2.VideoCapture('originalofvideoOUTPUT.mp4')
cv2.ocl.setUseOpenCL(False)

writer = None

# initialize a background subtractor

fgbg = cv2.createBackgroundSubtractorKNN(100,1000,True)

# fgbg = cv2.createBackgroundSubtractorMOG2()

while (1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    fgmask1=cv2.GaussianBlur(fgmask, (5, 5), 0)
    frame1=cv2.GaussianBlur(frame, (5, 5), 0)
    
    first_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # Apply thresholding on the background and display the resulting mask
    ret, mask = cv2.threshold(first_gray, 25, 255, cv2.THRESH_BINARY)
    
    (contours, hierarchy) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
    
    # looping for contours
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
    
        # get bounding box from countour
        (x, y, w, h) = cv2.boundingRect(c)
        dist = sqrt(((x + w) - x) ** 2 + ((y + h) - y) ** 2)
        # print(dist)
    
        # if (dist > 100 and dist < 350):
            # draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('foreground and background', fgmask)
    # filename = path + str(i) + ".jpg"
    # i = i + 1
    # cv2.imwrite(filename, fgmask)

    cv2.imshow('rgb', frame)

    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("bgsubofvideo1.mp4", fourcc, 30,
                                 (fgmask.shape[1], fgmask.shape[0]), True)
    writer.write(fgmask)

    # filename = path1 + str(i) + ".jpg"
    # i = i + 1
    # cv2.imwrite(filename, frame)

    # cv2.imshow('fgmask1', frame)
    # cv2.imshow('frame1', fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


writer.release()
cap.release()
cv2.destroyAllWindows()
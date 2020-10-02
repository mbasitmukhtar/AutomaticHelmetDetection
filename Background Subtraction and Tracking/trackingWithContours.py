from pyimagesearch.centroidtracker import CentroidTracker
import numpy as np
import cv2
from math import sqrt

video_path = 'highway.mp4'
writer = None
cv2.ocl.setUseOpenCL(False)

# read video file
cap = cv2.VideoCapture("highway.mp4")

# Apply a background subtractor
fgbg = cv2.createBackgroundSubtractorKNN(100,1000,True)

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
rects = []
while (cap.isOpened):

    # if ret is true than no error with cap.isOpened
    ret, frame = cap.read()

    if ret == True:
        # apply background substraction
        fgmask = fgbg.apply(frame)

        fgmask[fgmask==127]=0
        kernel0 = np.ones((1, 1), np.uint8)
        kernel1 = np.ones((3, 3), np.uint8)

        dilation = cv2.dilate(fgmask, kernel0, iterations=3)
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel0, iterations=10)
        erosion = cv2.erode(closing, kernel1)

        cv2.imshow('closing', closing)
        cv2.imshow('dilation', dilation)
        cv2.imshow('erosion', erosion)

        # opening = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel0)
        (contours, hierarchy) = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw and see contours

        # for contour in contours:
        #     cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)

        # looping for contours
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue

            # get bounding box from countour
            (x, y, w, h) = cv2.boundingRect(c)
            box = np.array([x, y, w, h])
            rects.append(box.astype("int"))

            # Add filters
            # dist = sqrt(((x+w)-x) ** 2 + ((y+h)-y) ** 2)
            # # print(dist)
            #
            # if(dist<300 and dist>50):
            #     # draw bounding box
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        objects = ct.update(rects)

        cv2.imshow('foreground and background', fgmask)
        cv2.imshow('rgb', frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output1.avi", fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    # write the output frame to disk
    writer.write(frame)

writer.release()
cap.release()
cv2.destroyAllWindows()
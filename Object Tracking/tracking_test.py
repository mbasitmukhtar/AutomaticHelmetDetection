from __future__ import print_function
from scipy.spatial import distance
import numpy as np
import cv2
import sys
import cv2

def readfortracks():
    boxforvideo1 = []
    boxfromtext1 = []
    i = 1
    file = open("dataa3m3.txt", "r")

    for line in file:

        fields = line.split(",")
        framenumber = fields[0]
        x = fields[1]
        y = fields[2]
        w = fields[3]
        h = fields[4]

        if (int)(framenumber)==i:
            boxfromtext1.append([(int)(x),(int)(y), (int)(w), (int)(h)])
        else:
            boxforvideo1.append(boxfromtext1)
            boxfromtext1 = []
            i = i + 1

    boxforvideo1.append(boxfromtext1)
    file.close()
    return [boxforvideo1]

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


# *****Main Code*****
# *****Main Code*****
# *****Main Code*****
# *****Main Code*****


print ("[INFO] Starting up")

nooffrmaes1 = 10

[videomatrix1] = readfortracks()
print (videomatrix1[1])

print("\nDistances\n")
res = []
for i in range(0,nooffrmaes1-1):
    vm0 = np.array(videomatrix1[i])
    vm1 = np.array(videomatrix1[i+1])
    a = distance.cdist(vm0, vm1, 'euclidean')
    res.append(a)

print("\nMins\n")
mins = []
for i in range(0,nooffrmaes1-1):
    length = int(len(videomatrix1[i]))
    a = []
    for j in range(0,length):
        a.append(np.argmin(res[i][j]))
    mins.append(a)

print("\nCentroids\n")
framecount1 = 0
rects = []
trackerType = "CSRT"
multiTracker = cv2.MultiTracker_create()
cap = cv2.VideoCapture("a3m.mp4")
writer = None


while (cap.isOpened()):
    if framecount1<nooffrmaes1-1:
        ret, frame = cap.read()
        if ret == True:
            i = framecount1
            framecount1 = framecount1 + 1

            print ("Videomatrix1  "+str(i))
            print (videomatrix1[i])
            print ("Videomatrix1  "+str(i+1))
            print (videomatrix1[i+1])

            vm0 = np.array(videomatrix1[i])
            vm1 = np.array(videomatrix1[i+1])
            a = distance.cdist(vm0, vm1, 'euclidean')
            print(a)

            vm0length = int(len(videomatrix1[i]))
            vm1length = int(len(videomatrix1[i+1]))

            for x in range(0,vm0length):
                mindist = min(a[x])
                print("Min dist: " + (str)(mindist))



            # cv2.imshow('MultiTracker', frame)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter('pathdetectionupdated2.mp4', fourcc, 30,
                                         (frame.shape[1], frame.shape[0]), True)
            writer.write(frame)

    else:
        writer.release()
        cap.release()
from pyimagesearch.centroidtracker import CentroidTracker
from scipy.spatial import distance
import numpy as np
import cv2


ct = CentroidTracker()

def readfortracks():
    boxforvideo1 = []
    boxfromtext1 = []
    i = 1
    file = open("yolovideoboxes.txt", "r")

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





# *****Main Code*****
# *****Main Code*****
# *****Main Code*****
# *****Main Code*****



print ("[INFO] Starting up")

nooffrmaes1 = 220

# tracking(nooffrmaes1)

[videomatrix1] = readfortracks()

print ("\nChecking")
print (videomatrix1[0])

# for i in range(0,nooffrmaes1):
#     print ("\n")
#     print (videomatrix1[i])


print("\nDistances\n")
res = []
for i in range(0,nooffrmaes1-1):
    vm0 = np.array(videomatrix1[i])
    vm1 = np.array(videomatrix1[i+1])
    # print("Arrays\n")
    # print (vm0)
    # print (vm1)
    a = distance.cdist(vm0, vm1, 'euclidean')
    res.append(a)
# print(res)

#
print("\nMins\n")
mins = []
for i in range(0,nooffrmaes1-1):
    length = int(len(videomatrix1[i]))
    a = []
    for j in range(0,length):
        a.append(np.argmin(res[i][j]))
    mins.append(a)
print (mins)

print("\nCentroids\n")
framecount1 = 0
centroids1 = []
centroidsrow1 = []
centroids2 = []
centroidsrow2 = []


rects = []

cap = cv2.VideoCapture("videos/a3m.mp4")
writer = None
while (cap.isOpened()):
    if framecount1<nooffrmaes1-1:
        ret, frame = cap.read()
        if ret == True:
            i = framecount1
            framecount1 = framecount1 + 1

            length0 = int(len(videomatrix1[i]))
            for x in range(0, length0):
                cv2.rectangle(frame, (videomatrix1[i][x][0], videomatrix1[i][x][1]),
                          (videomatrix1[i][x][2], videomatrix1[i][x][3]), (0, 255, 0), 2)

            # objects = ct.update(videomatrix1[i])
            # for (objectID, centroid) in objects.items():
            #     text = "ID {}".format(objectID)
            #     cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #     cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            if framecount1>50:
                k = framecount1-30
            else:
                k=0

            while k < framecount1:
                length = int(len(videomatrix1[k]))
                for j in range(0, length):

                    var = mins[k][j]

                    cx1 = int((videomatrix1[k][j][0] + videomatrix1[k][j][2]) / 2)
                    cy1 = int((videomatrix1[k][j][1] + videomatrix1[k][j][3]) / 2)

                    cx2 = int((videomatrix1[k+1][var][0] + videomatrix1[k+1][var][2]) / 2)
                    cy2 = int((videomatrix1[k+1][var][1] + videomatrix1[k+1][var][3]) / 2)

                    if abs(cx2-cx1)<40:
                        cv2.line(frame,(cx1,cy1),(cx2,cy2),(0,255,0),9)

                    centroidsrow1.append([cx1, cy1])
                    centroidsrow2.append([cx2, cy2])
                k=k+1
            # MJPG
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter('output/pathdetectionupdated1.mp4', fourcc, 30,
                                         (frame.shape[1], frame.shape[0]), True)
            writer.write(frame)

            centroids1.append(centroidsrow1)
            centroids2.append(centroidsrow2)
            centroidsrow1 = []
            centroidsrow2 = []

    else:
        writer.release()
        cap.release()

        print("Centroids 1")
        print(centroids1)
        print("Centroids 2")
        print(centroids2)

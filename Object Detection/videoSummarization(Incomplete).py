from pyimagesearch.centroidtracker import CentroidTracker
from scipy.spatial import distance
import numpy as np
import cv2
from PIL import Image

ct = CentroidTracker()

def readfortracks():
    boxforvideo1 = []
    boxfromtext1 = []
    i = 1
    file = open("Singles/ferozpur road_riksha.txt", "r")

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

nooffrmaes1 = 40

[videomatrix1] = readfortracks()

print ("\nChecking")
print (videomatrix1[0])



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
# videobg = cv2.imread("videobg.jpg")
videobg = cv2.imread("bgcropmade.png")
# cv2.imshow("O Frame",videobg)
frame1 = videobg
writer = None
writer1 = None


imgi=1
while (cap.isOpened()):
    if framecount1<nooffrmaes1-1:
        ret, frame = cap.read()
        if ret == True:
            i = framecount1
            framecount1 = framecount1 + 1

            length0 = int(len(videomatrix1[i]))
            # for x in range(0, length0):
            #     cv2.rectangle(frame, (videomatrix1[i][x][0], videomatrix1[i][x][1]),
            #               (videomatrix1[i][x][2], videomatrix1[i][x][3]), (0, 255, 0), 2)

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
                        framecopy = frame
                        cropimg = framecopy[videomatrix1[k][j][1] : videomatrix1[k][j][3], videomatrix1[k][j][0] : videomatrix1[k][j][2]]
                        # cv2.imwrite("imgs\img"+ (str)(imgi) +".jpg", cropimg)
                        # imgi = imgi+1
                        # print (imgi)
                        frame1[int(videomatrix1[k][j][1]) : int(videomatrix1[k][j][3]), int(videomatrix1[k][j][0]) : int(videomatrix1[k][j][2])] = cropimg
                        # cv2.imshow("S Frame",frame1)
                        # cv2.imwrite("imgs\img" + (str)(imgi) + ".jpg", frame1)
                    # imgi = imgi + 1
                    centroidsrow1.append([cx1, cy1])
                    centroidsrow2.append([cx2, cy2])
                k=k+1

            if writer1 is None:
                fourcc1 = cv2.VideoWriter_fourcc(*"mp4v")
                writer1 = cv2.VideoWriter('output/videosummarizationcaronly.mp4', fourcc1, 30,
                                         (frame1.shape[1], frame1.shape[0]), True)
            writer1.write(frame1)

            frame1 = videobg

            centroids1.append(centroidsrow1)
            centroids2.append(centroidsrow2)
            centroidsrow1 = []
            centroidsrow2 = []

    else:
        # writer.release()
        writer1.release()
        cap.release()

        print("Centroids 1")
        print(centroids1)
        print("Centroids 2")
        print(centroids2)

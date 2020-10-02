# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from scipy.spatial import distance
# from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
ap.add_argument("-o", "--output", required=True,
                help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])


# **********Functions********** #

def framejump(framenumber,x1,y1,w1,h1,classid,boxforvideo):
    # framenumber = int(framenumber)
    i=1
    # for f in framenumber:
    #     thisframenumber = f
    #     print (f)
    for a in boxforvideo:
        (x,y,w,h)= a
        print ("now")
        print (a)
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        cap = cv2.VideoCapture("videos/a3m.mp4")
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # while (cap.isOpened()):
        if(cap.isOpened()):
            # cap.set(cv2.CAP_PROP_POS_FRAMES, thisframenumber)
            i = i+1
            # framenumber = framenumber + 1
            ret, frame = cap.read()
            if ret == True:
                # text = "{}".format(str(LABELS[classid]))
                # cv2.putText(frame, text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                #             2)
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.imshow('Tracked Video', frame)
                # return
                # time.sleep(2)
            # else:
            #     break
            # break
            if cv2.waitKey(25) & 0xFF == ord('q'):
                return
                # break
        cap.release()

def framejump1(framenumber,x1,y1,w1,h1,classid,boxforvideo):
    # framenumber = int(framenumber)
    i=0
    # for f in framenumber:
    #     thisframenumber = f
    #     print (f)
    cap = cv2.VideoCapture("videos/a3m.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            for f in framenumber:
                if((int)(f) == i):
                    print ("f"+f)
                    (x,y,w,h)= boxforvideo[(int)(f)]

                    # print ("now")
                    # print (a)
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)


                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.imshow('Tracked Video', frame)
            i = i + 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                return
                # break
    cap.release()



boxfromtext = []
boxforvideo = []



def readdatafromfile():
    file = open("dataa3m.txt", "r")
    x=0
    y=0
    w=0
    h=0
    framenumbera = []
    framenumberb = []
    classID = ''
    for line in file:
        fields = line.split(",")
        framenumber = fields[0]
        x = fields[1]
        y = fields[2]
        w = fields[3]
        h = fields[4]
        classID = fields[5]

        print("framenumber: " + framenumber + ",x:" + x + ",y:" + y + ",w:" + w + ",h:" + h + ",classID:" + classID)
        boxfromtext = np.array([x,y,w,h])
        boxforvideo.append(boxfromtext.astype("int"))
        framenumbera = np.array([framenumber])
        # framenumberb.append(framenumbera.astype("int"))
        framenumberb.append(framenumber)

        # framejump(framenumber,x,y,w,h,classID,boxforvideo)
        # print("Frame Jump Called")
        # classID = int(classID)
        # print(LABELS[classID])

    # print(boxforvideo)
    print(framenumberb)
    framejump1(framenumberb,x,y,w,h,classID,boxforvideo)
    file.close()

# NEW FUNCTION
# noofframes1=2

framematrix = ([])
videomatrix = ([])

# videomatrix = np.zeros((noofframes1,1))
# framematrix = np.zeros((1, 5))

# videomatrix.append(videomatrix,[],axis=0)
# framematrix.append(framematrix,[], axis=0)

# videomatrix = np.array([ [0,0,0,0,0] ])
# framematrix = np.array([ [0,0,0,0,0] ])

def tracking(nooffrmaes):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO configs and weights...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    framematrix = []
    # videomatrix = np.zeros((1,nooffrmaes))


    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    vs = cv2.VideoCapture(args["input"])
    writer = None
    (W, H) = (None, None)

    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()

    myiter1 = 1

    f = open("dataa3mwithWIDTH_HEIGHT.txt","w")
    framecount = 1
    # loop over the frames from the video stream
    while framecount<nooffrmaes+1:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        rects = []
        boxes = []
        boxes1 = []
        confidences = []
        classIDs = []

        # pts = []
        # objects= []
        myiter = 0
        framematrix1 = []

        for output in layerOutputs:
            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    boxes1.append([x, y, int(x+width), int(y+height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

                    # f.write("name= " + str(classID) + ",x= " + str(x) + ",y= " + str(y) + ",w= " + str(x+width) + ",h= " + str(
                    #         y+height) + "\n")

                    # f.write(str(framecount) + "," + str(x) + "," + str(y) + "," + str(x + width) + "," + str(
                    #     y + height) + "," + str(LABELS[classID]) + ",1\n")
                    f.write(str(framecount) + "," + str(x) + "," + str(y) + "," + str(width) + "," + str(height) + "," + str(LABELS[classID]) + ",1\n")
                    # framematrix=np.append(framematrix,framecount,axis=0)
                    # framematrix=np.append(framematrix,x,axis=0)
                    # framematrix=np.append(framematrix,y,axis=0)
                    # framematrix=np.append(framematrix,x+width,axis=0)
                    # framematrix=np.append(framematrix,y+height,axis=0)
                    # framematrix.append(framecount)
                    framematrix.append([x,y,x+width,y+height])
                    # framematrix.append(y)
                    # framematrix.append(x + width)
                    # framematrix.append(y + height)
            # framematrix1.append(framematrix)
            # framematrix = []
        tempmatrix = []
        # tempmatrix = np.array(framematrix)
        videomatrix.append(framematrix)
        framematrix = []
        # videomatrix=np.append(videomatrix,framematrix,axis=0)

        # framematrix = np.zeros((1,5))
        # videomatrix.append([framematrix])
        # print(videomatrix)


        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                                args["threshold"])
        # ensure at least one detection exists
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                # text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                # 	confidences[i])

                # objects = ct.update(boxes1)
                # for (objectID, centroid) in objects.items():
                #     text = "ID {}".format(objectID)
                #     # text = "{}: {}".format(LABELS[classIDs[objectID]],objectID)
                #     cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #                 (0, 255, 0), 2)
                #     cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                #
                #     f.write(str(framecount) + "," + str(objectID) + "," + str(x) + "," + str(y) + "," + str(
                #         x + w) + "," + str(
                #         y + h) + "," + str(LABELS[classID]) + "\n")

                # text = "{}: {}".format(LABELS[classIDs[i]], i)
                # cv2.putText(frame, text, (x, y - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print("[INFO] Done with adding rectangles "+str(myiter1))
        print("[INFO] Starting tracking "+str(myiter1))

        objects = ct.update(boxes1)
        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            # text = "{}: {}".format(LABELS[classIDs[objectID]],objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        #
        #     f.write(str(framecount) + "," + str(objectID) + "," + str(x) + "," + str(y) + "," + str(x + width) + "," + str(
        #         y + height) + "," + str(LABELS[classID]) + "\n")
        #


        print("[INFO] Done with tracking now writing video "+str(myiter1))
        myiter1 = myiter1 + 1
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish {} franes is: {:.4f}".format(nooffrmaes,
                    elap * nooffrmaes))

        writer.write(frame)
        framecount = framecount + 1

    print("[INFO] cleaning up...")
    writer.release()
    vs.release()


def readfortracks():
    boxforvideo1 = []
    boxfromtext1 = []
    i = 1
    file = open("dataa3m.txt", "r")

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

nooffrmaes1 = 250

tracking(nooffrmaes1)

# [videomatrix1] = readfortracks()

# print ("\nChecking")
# print (videomatrix1[0])


# for i in range(0,nooffrmaes1):
#     print ("\n")
#     print (videomatrix1[i])

exit()


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

# 3
#
# linearray = []
# cap = cv2.VideoCapture("videos/a3m.mp4")
# writer = None
# while (cap.isOpened()):
#     if framecount1<nooffrmaes1-1:
#         ret, frame = cap.read()
#         if ret == True:
#             i = framecount1
#             framecount1 = framecount1 + 1
#             # for i in range(0,nooffrmaes1-1):
#             length = int(len(videomatrix1[i]))
#             for j in range(0, length):
#
#                 var = mins[i][j]
#
#                 # print("\nFrame array")
#                 # print(videomatrix1[i][j])
#                 cx1 = int((videomatrix1[i][j][0] + videomatrix1[i][j][2]) / 2)
#                 cy1 = int((videomatrix1[i][j][1] + videomatrix1[i][j][3]) / 2)
#                 # print ("Centroid Frame array")
#                 # print (cx1, cy1)
#
#                 # print("\nDistance array")
#                 # print(videomatrix1[i+1][var])
#                 cx2 = int((videomatrix1[i+1][var][0] + videomatrix1[i+1][var][2]) / 2)
#                 cy2 = int((videomatrix1[i+1][var][1] + videomatrix1[i+1][var][3]) / 2)
#                 # print("Centroid Distance array")
#                 # print(cx2, cy2)
#
#                 # cv2.line(frame,(cx1,cy1),(cx2,cy2),(0,255,0),9)
#
#                 centroidsrow1.append([cx1, cy1])
#                 centroidsrow2.append([cx2, cy2])
#
#             cv2.imshow('Line',frame)
#             if writer is None:
#                 fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#                 writer = cv2.VideoWriter(args["output"], fourcc, 30,
#                                          (frame.shape[1], frame.shape[0]), True)
#             writer.write(frame)
#
#             centroids1.append(centroidsrow1)
#             centroids2.append(centroidsrow2)
#             centroidsrow1 = []
#             centroidsrow2 = []
#
#     else:
#         writer.release()
#         cap.release()
#
#         print("Centroids 1")
#         print(centroids1)
#         print("Centroids 2")
#         print(centroids2)
#
# # 4
#
# clen = len(centroids1)
# cap = cv2.VideoCapture("videos/a3m.mp4")
# writer = None
#
# for i in range(0,clen):
#     clen1 = len(centroids1[i])
#     ret, frame = cap.read()
#     if ret == True:
#         for j in range(0,clen1):
#                 cv2.line(frame,(centroids1[i][j][0],centroids1[i][j][1]),(centroids2[i][j][0],centroids2[i][j][1]),(0,255,0),9)
#
#         if writer is None:
#             fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#             writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
#         writer.write(frame)
#
# writer.release()

# 5

# linearr =  []
# linearr1 = []
# cx1 = int((videomatrix1[0][0][0] + videomatrix1[0][0][2]) / 2)
# cy1 = int((videomatrix1[0][0][1] + videomatrix1[0][0][3]) / 2)
# linearr.append([cx1, cy1])
#
# for i in range(0,nooffrmaes1-1):
#     length = int(len(videomatrix1[i]))
#
#     var = mins[i][0]
#
#     cx2 = int((videomatrix1[i+1][var][0] + videomatrix1[i+1][var][2]) / 2)
#     cy2 = int((videomatrix1[i+1][var][1] + videomatrix1[i+1][var][3]) / 2)
#
#     linearr.append([cx2,cy2])
#
# print (linearr)
#
# cap = cv2.VideoCapture("videos/a3m.mp4")
# writer = None
#
# for k in range(1,nooffrmaes1-1):
#     i=0
#     ret, frame = cap.read()
#     while (i<i+k):
#         cv2.line(frame, (linearr[i][0],linearr[i][1]) , (linearr[i+1][0],linearr[i+1][1]) ,(0,255,0),9)
#         i=i+1
#
#     if writer is None:
#         fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#         writer = cv2.VideoWriter(args["output"], fourcc, 30,
#                                  (frame.shape[1], frame.shape[0]), True)
#     writer.write(frame)
#
# writer.release()

# 6

linearray = []
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

            if framecount1>50:
                k = framecount1-30
            else:
                k=0

            while k < framecount1:
                length = int(len(videomatrix1[k]))
                for j in range(0, length):
                    # for l in range(0,length):

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
                writer = cv2.VideoWriter(args["output"], fourcc, 30,
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


# 1

# for i in range(0,nooffrmaes1-1):
#     length = int(len(videomatrix[i]))
#     for j in range(0, length):
#
#         # var = mins1[i-1][j]
#         var = mins[i][j]
#         print (var)
#
#         print("\nFrame array")
#         print(videomatrix[i][j])
#         cx1 = int((videomatrix[i][j][0] + videomatrix[i][j][2]) / 2)
#         cy1 = int((videomatrix[i][j][1] + videomatrix[i][j][3]) / 2)
#         print ("\nCentroid Frame array")
#         print (cx1, cy1)
#
#         print("\nDistance array\n")
#         print(videomatrix[i + 1][var])
#         cx2 = int((videomatrix[i+1][var][0] + videomatrix[i+1][var][2]) / 2)
#         cy2 = int((videomatrix[i+1][var][1] + videomatrix[i+1][var][3]) / 2)
#         print("\nCentroid Distance array")
#         print(cx2, cy2)
#
#         centroidsrow1.append([cx1,cy1])
#         centroidsrow2.append([cx1, cy1])
#     centroids1.append(centroidsrow1)
#     centroids2.append(centroidsrow2)
# print (centroids1)
# print (centroids2)




# Done Making Tracks

print ("RUNNING COMPLETE.")
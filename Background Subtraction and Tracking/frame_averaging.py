import cv2
import numpy as np

# write path

path = "D:\ITU\Final Year Project\Others\BgExtract\mybg"

# input video path
c = cv2.VideoCapture("D:\ITU\Final Year Project\Others\input.mp4")

writer = None
writer1 = None

i = 1
avg1 = np.float32(f)
avg2 = np.float32(f)

# Read frames in a loop and writes as a video
while (1):
    _, f = c.read()

    cv2.accumulateWeighted(f, avg1, 0.01)
    cv2.accumulateWeighted(f, avg2, 0.05)

    res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)

    cv2.imshow('img', f)
    cv2.imshow('avg1', res1)

    cv2.imshow('avg2', res2)
    k = cv2.waitKey(20)


    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("originalofvideo.avi", fourcc, 30,
                                 (f.shape[1], f.shape[0]), True)

    # write the output frame with avg weight 0.01 to pc
    writer.write(f)

    if writer1 is None:
        # initialize our video writer
        fourcc1 = cv2.VideoWriter_fourcc(*"mp4v")
        writer1 = cv2.VideoWriter("bgofvideo.avi", fourcc1, 30,
                                 (res1.shape[1], res1.shape[0]), True)

    # write the output frame with avg weight 0.05 to pc
    writer1.write(res1)

    if k == 27:
        break

writer.release()
writer1.release()
cv2.destroyAllWindows()
c.release()
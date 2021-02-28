from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

orangeLower = (0, 91, 45)
orangeUpper = (37, 255, 255)
pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
    camera = cv2.VideoCapture(1)
else:
    camera = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)



while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        pts.appendleft(center)
        cv2.putText(frame, ('x='+str(int(x))), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,100), 2, cv2.LINE_AA)
        cv2.putText(frame, ('y='+str(int(y))), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,100), 2, cv2.LINE_AA)
        cv2.putText(frame, ('z='+str(int(radius))), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,100), 2, cv2.LINE_AA)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()


import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
cv.namedWindow("Mavi Nesne Takibi", cv.WINDOW_NORMAL)
cv.namedWindow("Mask", cv.WINDOW_NORMAL)

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

yellow = (0, 255, 255)
red = (0, 0, 255)
green = (0, 255, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, lower_blue, upper_blue)

    kernel =np.ones((7,7), np.uint8)
    mask = cv.GaussianBlur(mask, (7,7), 0)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    min_area = 300  
    good_contours = [c for c in contours if cv.contourArea(c) > min_area]

    centers = [] 
    boxes = []

    for c in good_contours:
     x, y, w, h = cv.boundingRect(c)
     cx, cy = x + w // 2, y + h // 2
     centers.append((cx, cy))
     boxes.append((x, y, w, h))
    for i, (x, y, w, h) in enumerate(boxes):
     cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
     cx, cy = centers[i]
     cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
     for i in range(len(centers)):
         for j in range(i+1, len(centers)):
            cv.line(frame, centers[i], centers[j], (0, 255, 0), 2)
    cv.imshow("mavi nesne takibi", frame)

    if cv.waitKey(1) & 0xFF == 27:
         break

cap.release()
cv.destroyAllWindows()
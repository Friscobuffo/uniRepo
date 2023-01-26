import cv2
import numpy as np
 
# Turn on Laptop's webcam
cap = cv2.VideoCapture(0)
i=0
while True:
    
    ret, frame = cap.read()
 
    # Locate points of the documents
    # or object which you want to transform
    pts1 = np.float32([[0, 100], [300, 100],
                       [0, 400], [300, 400]])
    pts2 = np.float32([[0, 0], [400, 0],
                       [0, 640], [400, 640]])
     
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (400, 640))
    if i==0:
        print(matrix)
    # Wrap the transformed image
    cv2.imshow('frame', frame) # Initial Capture
    cv2.imshow('frame1', result) # Transformed Capture
    i += 1
    if cv2.waitKey(24) == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
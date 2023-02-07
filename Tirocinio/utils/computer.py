import cv2 as cv
import numpy as np
import itertools
from utils.bar import ProgressBar
from math import dist

class Computer():
    def __init__(self):
        self.__sift = cv.SIFT_create()
        self.__matcher = cv.BFMatcher()

    def compute(self, manager):
        print("Starting computing of scenevideo")
        originalColored = manager.getOriginal()
        original = cv.cvtColor(originalColored, cv.COLOR_BGR2GRAY)
        dataframe = manager.getDataframe()
        fps = manager.getFps()
        dataframeIndex = 0
        frameNumber = 0
        videoWidth = manager.getVideoWidth()*manager.getVideoScale()
        videoHeight = manager.getVideoHeight()*manager.getVideoScale()
        processedCoordinates = [] # output
        bar = ProgressBar(manager.getVideoLenght())
        # output video makers initializing
        videoSize = original.shape[1], original.shape[0]
        video1 = cv.VideoWriter(manager.getOutputPath()+'/video1.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, videoSize)
        video2 = cv.VideoWriter(manager.getOutputPath()+'/video2.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, videoSize)
        print("Processing frames\n")
        while manager.hasNextFrame():
            bar.next()
            frame = manager.getFrame()
            # getting dataframe infos
            row = dataframe.iloc[[dataframeIndex]]
            # taking the matching timestamp eyegaze dataframe row 
            timeStamp = frameNumber/fps
            gazeTimeStamp = float(row["timestamps"])
            while gazeTimeStamp < timeStamp:
                dataframe.drop([dataframeIndex], axis=0, inplace=True)
                dataframeIndex += 1
                row = dataframe.iloc[[dataframeIndex]]
                gazeTimeStamp = float(row["timestamps"])
            # x,y are between 0 and 1
            x = row["x"].tolist()[0]
            y = row["y"].tolist()[0]
            if x!='-1' and y!='-1': # if dataframe contains valid input
                matrix = self.__perspectiveTransformMatrix(frame, original)
                # converting x,y to video frame coordinates
                x = int(float(x)*videoWidth)
                y = int(float(y)*videoHeight)
                vec = np.float32([x,y,1])
                res = np.matmul(matrix, vec)
                # x,y now are original image coordinates
                x,y = res[0]/res[2], res[1]/res[2]
                processedCoordinates.append((timeStamp, (x/original.shape[1]), (y/original.shape[0]))) # dividing by original's size to make x,y again between 0 and 1
                # output videos making
                result1 = originalColored.copy()
                cv.circle(img=result1, center=(int(x),int(y)), radius=10, color=(0, 255, 0), thickness=5)
                result2 = cv.warpPerspective(frame, matrix, (original.shape[1], original.shape[0]))
                result2 = cv.cvtColor(result2,cv.COLOR_GRAY2BGR) # video output doesnt work with image in grayscale
                video1.write(result1)
                video2.write(result2)
            else:
                video1.write(originalColored)
                processedCoordinates.append((timeStamp, -1,-1))
            frameNumber += 1
        print("\nAll frames processed\n")
        video1.release()
        video2.release()
        cv.destroyAllWindows()
        print("Videos making completed")
        print("Gazedata conversion completed\n")
        return processedCoordinates

    def __perspectiveTransformMatrix(self, frame, original):
        # find the keypoints and descriptors with SIFT
        points1, desc1 = self.__sift.detectAndCompute(frame,None)
        points2, desc2 = self.__sift.detectAndCompute(original,None)
        matches = self.__matcher.match(desc1,desc2) # match descriptors
        queryPoints, trainPoints = self.__matchesHandler(matches, points1, points2)
        # opencv function wants numpy array of floats
        queryPoints = np.float32(queryPoints)
        trainPoints = np.float32(trainPoints)
        # final results
        matrix = cv.getPerspectiveTransform(queryPoints, trainPoints)
        return matrix
    
    def __matchesHandler(self, matches, points1, points2):
        # sort them in the order of their distance
        matches = sorted(matches, key = lambda x:x.distance)
        # get first 20 matches.
        matches = matches[:20]
        queryPoints = []
        trainPoints = []
        for match in matches:
            p1 = points1[match.queryIdx].pt
            p2 = points2[match.trainIdx].pt
            p1 = round(p1[0]), round(p1[1])
            p2 = round(p2[0]), round(p2[1])
            # 
            if p1 not in queryPoints and p2 not in trainPoints:
                queryPoints.append(p1)
                trainPoints.append(p2)
        # get 4 points with maximised distance between them
        # makes perspective transformation more accurate
        combinations = itertools.combinations(range(0,len(queryPoints)), 4)
        maxDistance = 0
        for c1,c2,c3,c4 in combinations:
            distance = dist(queryPoints[c1],queryPoints[c2]) + dist(queryPoints[c1],queryPoints[c3])
            distance += dist(queryPoints[c1],queryPoints[c4]) + dist(queryPoints[c2],queryPoints[c3])
            distance += dist(queryPoints[c2],queryPoints[c4]) + dist(queryPoints[c3],queryPoints[c4])
            if distance>maxDistance:
                maxDistance = distance
                bestIndexes = c1,c2,c3,c4
        queryPoints = [queryPoints[bestIndexes[0]], queryPoints[bestIndexes[1]], queryPoints[bestIndexes[2]], queryPoints[bestIndexes[3]]]
        trainPoints = [trainPoints[bestIndexes[0]], trainPoints[bestIndexes[1]], trainPoints[bestIndexes[2]], trainPoints[bestIndexes[3]]]
        return queryPoints, trainPoints
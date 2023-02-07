import os
import cv2 as cv
import pandas as pd
from math import sqrt

class Manager():
    def __init__(self):
        self.__path = os.path.dirname(__file__)[:-6]
        self.__dataPath = self.__path + "/data"
        self.__outputPath = self.__path + "/output"

    def autoGetData(self):
        print("Auto-getting all input data")
        files = os.listdir(self.__dataPath)
        check = ((len(files)==3) and ("scenevideo.mp4" in files) and ('gazedata' in files) and ("original.jpg" in files))
        if check:
            self.__original = cv.imread(self.__dataPath+'/original.jpg')
            self.__vidcap = cv.VideoCapture(self.__dataPath+'/scenevideo.mp4')
            self.__dataframe = self.__getDataFrame() # contains eyegaze's coordinates
            # getting video infos
            self.__fps = round(self.__vidcap.get(cv.CAP_PROP_FPS))
            self.__videoWidth  = int(self.__vidcap.get(cv.CAP_PROP_FRAME_WIDTH))
            self.__videoHeight  = int(self.__vidcap.get(cv.CAP_PROP_FRAME_HEIGHT))
            self.__videolength = int(self.__vidcap.get(cv.CAP_PROP_FRAME_COUNT))
            self.__dataResizes()
            print("Success\n")
        else:
            print("Error: there should be 3 files inside data folder:")
            print(" - scenevideo.mp4")
            print(" - gazedata")
            print(" - original.jpg")
            print("Instead, there are", len(files), "files:")
            for file in files: print(" -", file)
            print()
            raise FileNotFoundError
    
    def __dataResizes(self, maxVideoDiagonal=1000, maxOriginalDiagonal=1000):
        # original image and video resizes, makes keypoints retrieving quicker 
        # and makes matches between images more accurate
        videoDiagonal = sqrt(self.__videoWidth**2 + self.__videoHeight**2)
        originalDiagonal = sqrt(self.__original.shape[1]**2 + self.__original.shape[0]**2)
        self.__videoScale = maxVideoDiagonal / max([videoDiagonal, maxVideoDiagonal])
        self.__originalScale = maxOriginalDiagonal / max([originalDiagonal, maxOriginalDiagonal])
        if self.__originalScale < 1:
            self.__original = self.__imageResize(self.__original, self.__originalScale)

    def __getDataFrame(self):
        df = pd.read_csv(self.__dataPath+'/gazedata', names = list(range(0,21)))
        df.drop(list(range(4,21)), axis=1, inplace=True)
        df.drop(0, axis=1, inplace=True)
        dataFrame = pd.DataFrame(columns=['timestamps','x','y'])
        for index, row in df.iterrows():
            timestamp = (row[1][10:])
            if row[2] == 'data:{}}':
                x = '-1'
                y = '-1'
            else:
                x = (row[2][16:])
                y = (row[3][:-1])
            dataFrame.loc[index] = [timestamp,x,y]
        return dataFrame
    
    # checks if video has another frame, and makes setup conversions (resize and grayscale)
    def hasNextFrame(self):
        hasFrame,self.__frame = self.__vidcap.read()
        if hasFrame:
            # converting frame to grayscale
            self.__frame = cv.cvtColor(self.__frame,cv.COLOR_BGR2GRAY)
            # resize of the frame
            if self.__videoScale < 1:
                self.__frame = self.__imageResize(self.__frame, self.__videoScale)
        return hasFrame
    
    def __imageResize(self, image, scaleFactor):
        width = int(image.shape[1] * scaleFactor)
        height = int(image.shape[0] * scaleFactor)
        dim = (width, height)
        image =  cv.resize(image, dim, interpolation = cv.INTER_AREA)
        return image

    # getters
    def getFrame(self):
        return self.__frame
    def getDataframe(self):
        return self.__dataframe
    def getFps(self):
        return self.__fps
    def getVideoWidth(self):
        return self.__videoWidth
    def getVideoHeight(self):
        return self.__videoHeight
    def getOriginal(self):
        return self.__original
    def getVideoScale(self):
        return self.__videoScale
    def getVideoLenght(self):
        return self.__videolength
    def getPath(self):
        return self.__path
    def getOutputPath(self):
        return self.__outputPath
import cv2
import numpy as np

from SIFT_MAIN import dataBase
from SIFT.parts.matching import matching

# MAKE DATABASE
imgDir1 = "D:\\cv\\data\\prac\\bcard.jpg"
imgDir2 = "D:\\cv\\data\\prac\\bcard_background.jpg"

dataBase1 = dataBase(imgDir=imgDir1,
                     kdTree=True) # kdTree
dataBase2 = dataBase(imgDir=imgDir2,
                     kdTree=False)

match = matching()
# COMPARE / MATCHING
for feature in dataBase2 :

    match.BBF(kdTree=dataBase1,
              target=feature)
    ratio = match.nearestDistance/match.secondDistance
    if ratio <= 0.8 : # ratio under 0.8

        print("NEAREST NODE : ", match.nearestNode.val)
        print("TARGET NODE ", feature)

    # initialize class members
    match.nearestDistance = np.inf
    match.secondDistance = np.inf
    match.tryCnt = 0
    match.nearestNode = None
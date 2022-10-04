import cv2
from SIFT_MAIN import dataBase
from SIFT.parts.matching import matching

# MAKE DATABASE
imgDir1 = "D:\\cv\\data\\prac\\bcard.jpg"
imgDir2 = "D:\\cv\\data\\prac\\bcard_background.jpg"

dataBase1 = dataBase(imgDir=imgDir1,
                     kdTree=True) # kdTree
dataBase2 = dataBase(imgDir=imgDir2,
                     kdTree=False)
print(dataBase1.left)
print(dataBase1.right)
# COMPARE / MATCHING
match = matching()
for feature in dataBase2 :
    match.BBF(kdTree=dataBase1,
              target=feature)
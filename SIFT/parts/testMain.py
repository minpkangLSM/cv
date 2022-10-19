import cv2
import numpy as np
from matplotlib import pyplot as plt
from SIFT_MAIN import dataBase
from SIFT_INIT.parts.matching import matching

# MAKE DATABASE
imgDir1 = "D:\\cv\\data\\prac\\cannytest.png"
imgDir2 = "D:\\cv\\data\\prac\\cannytest.png"

# MAKE DATABASE FOR IMG1 (KDTREE)
dataBase1 = dataBase(imgDir=imgDir1,
                     kdTree=True) # kdTree

# MAKE DATABASE FOR IMG2 (DTYPE : numpy, DSHAPE : (N, 132))
dataBase2 = dataBase(imgDir=imgDir2,
                     kdTree=False)

match = matching()
matchPairs = {}
idx = 0

# COMPARE / MATCHING
for feature in dataBase2 :
    match.BBF(kdTree=dataBase1,
              target=feature)
    ratio = match.nearestDistance/match.secondDistance
    if ratio <= 0.8 : # ratio under 0.8
        idx += 1
        pair1 = match.nearestNode.val[-5:]
        pair2 = feature[-5:]
        matchPairs[idx] = (pair1, pair2)

    # initialize class members
    match.nearestDistance = np.inf
    match.secondDistance = np.inf
    match.tryCnt = 0
    match.nearestNode = None

for i in matchPairs.keys() :
    print(matchPairs[i])

# visualize feature points
img1 = cv2.imread(imgDir1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(imgDir2, cv2.IMREAD_GRAYSCALE)
deg = 0
fig, ax = plt.subplots()
# ax.imshow(img1, cmap='gray')
ax.imshow(img2, cmap='gray')

for pair1, pair2 in matchPairs.values():
    x_list = []
    y_list = []

    x1 = pair1[0]
    y1 = pair1[1]
    z1 = pair1[2]
    o1 = pair1[3]
    octave1 = pair1[4]

    x2 = pair2[0]
    y2 = pair2[1]
    z2 = pair2[2]
    o2 = pair2[3]
    octave2 = pair2[4]

    # scale(시그마 말고)에 따른, 사각형의 중심 위치
    newX = x2*(2**octave1) + (2**octave2-1)/2 # center X of Scale
    newY = y2*(2**octave1) + (2**octave2-1)/2 # center Y of Scale
    x_list.append(newX)
    y_list.append(newY)

    ax.scatter(x_list, y_list)
plt.show()
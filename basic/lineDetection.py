import os
import cv2
import numpy as np
import numba as nb
from PIL import Image
from time import process_time
from numba import njit, jit, int16, int32, int64, float32, float64, typed
from matplotlib import pyplot as plt

import filters
from tools.video import *

class thinning :

    @staticmethod
    def SPTA(img):
        """
        생성된 edge를 입력으로 받아, edge를 한 줄로 만들어 준다.
        binary 맵에 대해서만 적용 가능한 알고리즘.
        :param img: Binary image, shape (height, width, channel=1)
        :return:
        """
        # check input image if binary or not.
        if len(img.shape) > 2 and img.shape[2] > 1: raise ValueError("SPTA is working on 2-dim(binary) images.")

        # shrink img 3 dims into 2 dim (채널값 삭제)
        img = np.reshape(img, (img.shape[0], img.shape[1]))

        # pad image
        img = np.pad(img, (1, 1))
        initImg = img
        count = 0
        t1 = process_time()
        while True:
            count += 1
            edgeIndex = np.where(initImg == 1)
            edgeBoolean = np.where(initImg == 1, True, False)

            n0 = edgeBoolean[(edgeIndex[0], edgeIndex[1] + 1)]
            n1 = edgeBoolean[(edgeIndex[0] + 1, edgeIndex[1] + 1)]
            n2 = edgeBoolean[(edgeIndex[0] + 1, edgeIndex[1])]
            n3 = edgeBoolean[(edgeIndex[0] + 1, edgeIndex[1] - 1)]
            n4 = edgeBoolean[(edgeIndex[0], edgeIndex[1] - 1)]
            n5 = edgeBoolean[(edgeIndex[0] - 1, edgeIndex[1] - 1)]
            n6 = edgeBoolean[(edgeIndex[0] - 1, edgeIndex[1])]
            n7 = edgeBoolean[(edgeIndex[0] - 1, edgeIndex[1] + 1)]

            mask0 = np.invert(n0) & (n4 & (n5 | n6 | n2 | n3) & (n6 | np.invert(n7)) & (n2 | np.invert(n1)))
            mask2 = np.invert(n4) & (n0 & (n1 | n2 | n6 | n7) & (n2 | np.invert(n3)) & (n6 | np.invert(n5)))
            mask4 = np.invert(n2) & (n6 & (n7 | n0 | n4 | n5) & (n0 | np.invert(n1)) & (n4 | np.invert(n3)))
            mask6 = np.invert(n6) & (n2 & (n3 | n4 | n0 | n1) & (n4 | np.invert(n5)) & (n0 | np.invert(n7)))
            merge = (mask0 | mask2 | mask4 | mask6)
            if not merge.any():
                break
            noneEdge = (edgeIndex[0][merge], edgeIndex[1][merge])
            initImg[noneEdge] = 0

        # remove pad
        img = initImg[1:img.shape[0] - 1, 1:img.shape[1] - 1]
        t2 = process_time()
        print("SPTA PROCESSING TIME : {0}s".format(t2 - t1))
        return img


    # deprecated version, 이유 : 병렬문으로 처리하는 것(def SPTA)과 동일한 결과를 보이나, 소요시간이 10배 이상 더 걸림
    @staticmethod
    def SPTA_loop(img):
        if len(img.shape) > 2 and img.shape[2] > 1: raise ValueError("SPTA is working on 2-dim(binary) images.")
        raise DeprecationWarning("This version takes quiet long time. Strongly recommend to use SPTA.")
        # reshape img 3 dims into 2 dim
        img = np.reshape(img, (img.shape[0], img.shape[1]))
        # pad image
        img = np.pad(img, (1,1))
        init_img = img
        count = 0
        t1 = process_time()
        while True :
            count+=1
            edge_index = np.where(init_img == 1)
            edge_boolean = np.where(init_img == 1, True, False)
            loop = False

            for j, i in zip(edge_index[0], edge_index[1]):

                n0 = edge_boolean[j  , i+1]
                n1 = edge_boolean[j+1, i+1]
                n2 = edge_boolean[j+1, i  ]
                n3 = edge_boolean[j+1, i-1]
                n4 = edge_boolean[j  , i-1]
                n5 = edge_boolean[j-1, i-1]
                n6 = edge_boolean[j-1, i  ]
                n7 = edge_boolean[j-1, i+1]

                mask0 = np.invert(n0) & (n4 & (n5 | n6 | n2 | n3) & (n6 | np.invert(n7)) & (n2 | np.invert(n1)))
                mask2 = np.invert(n4) & (n0 & (n1 | n2 | n6 | n7) & (n2 | np.invert(n3)) & (n6 | np.invert(n5)))
                mask4 = np.invert(n2) & (n6 & (n7 | n0 | n4 | n5) & (n0 | np.invert(n1)) & (n4 | np.invert(n3)))
                mask6 = np.invert(n6) & (n2 & (n3 | n4 | n0 | n1) & (n4 | np.invert(n5)) & (n0 | np.invert(n7)))
                merge = (mask0 | mask2 | mask4 | mask6)

                if merge :
                    loop = True
                    init_img[j,i]=0

            if not loop : break
        img = init_img[1:img.shape[0] - 1, 1:img.shape[1] - 1]
        t2 = process_time()
        return img

class linedetection:

    @staticmethod
    def edgeSegment(img):
        """
        local operation algorithm to detect edge segment (or line).
        :param img: 세선화가 완료된 binary edge image (예 - 1) canny edge 적용 -> 2) 세선화 작업(SPTA) -> 3) 엣지 토막(edge segment) 검출)
        :return: 각 엣지 토막 번호(순서대로 부여)를 키로 갖으며, value는 엣지를 이루고 있는 픽셀의 ((y 좌표 리스트), (x 좌표 리스트))로 구성
        예) { 0 : (array([1440, 1440, 1440, 1440, 1440, 1440], dtype=int64), array([1053, 1054, 1055, 1056, 1057, 1058], dtype=int64)),
             1 : (array([1440, 1440, 1440], dtype=int64), array([887, 888, 889], dtype=int64)) ... }
        """
        t1 = process_time()
        # find end or bifurcation point
        endBifDict = {}
        img = np.pad(img, (1, 1))
        edgeIndex = np.where(img == 1)

        n0_group = img[(edgeIndex[0], edgeIndex[1] + 1)]
        n1_group = img[(edgeIndex[0] + 1, edgeIndex[1] + 1)]
        n2_group = img[(edgeIndex[0] + 1, edgeIndex[1])]
        n3_group = img[(edgeIndex[0] + 1, edgeIndex[1] - 1)]
        n4_group = img[(edgeIndex[0], edgeIndex[1] - 1)]
        n5_group = img[(edgeIndex[0] - 1, edgeIndex[1] - 1)]
        n6_group = img[(edgeIndex[0] - 1, edgeIndex[1])]
        n7_group = img[(edgeIndex[0] - 1, edgeIndex[1] + 1)]

        idx = 0
        for n0, n1, n2, n3, n4, n5, n6, n7 in zip(n0_group, n1_group, n2_group, n3_group, n4_group, n5_group, n6_group,
                                                  n7_group):
            count = 0
            dirList = []
            if n0 == 1 and n1 == 0:
                count += 1
                dirList.append(0)
            if n1 == 1 and n2 == 0:
                count += 1
                dirList.append(1)
            if n2 == 1 and n3 == 0:
                count += 1
                dirList.append(2)
            if n3 == 1 and n4 == 0:
                count += 1
                dirList.append(3)
            if n4 == 1 and n5 == 0:
                count += 1
                dirList.append(4)
            if n5 == 1 and n6 == 0:
                count += 1
                dirList.append(5)
            if n6 == 1 and n7 == 0:
                count += 1
                dirList.append(6)
            if n7 == 1 and n0 == 0:
                count += 1
                dirList.append(7)
            if count == 1 or count >= 3:
                endBifDict[(edgeIndex[0][idx], edgeIndex[1][idx])] = dirList
            idx += 1

        # start edge segmentation
        segId = 0
        visited = np.zeros_like(img)

        # initialize seg_dict, cy, cx
        segDict = {}
        cy = 0
        cx = 0

        # define front pixel location
        frontLoc = {
            0: [[-1, +1, 7], [0, +1, 0], [+1, +1, 1]],
            1: [[-1, +1, 7], [0, +1, 0], [+1, +1, 1], [+1, 0, 2], [+1, -1, 3]],
            2: [[+1, -1, 3], [+1, 0, 2], [+1, +1, 1]],
            3: [[-1, -1, 5], [0, -1, 4], [+1, -1, 3], [+1, 0, 2], [+1, +1, 1]],
            4: [[-1, -1, 5], [0, -1, 4], [+1, -1, 3]],
            5: [[+1, -1, 3], [0, -1, 4], [-1, -1, 5], [-1, 0, 6], [-1, +1, 7]],
            6: [[-1, -1, 5], [-1, 0, 6], [-1, +1, 7]],
            7: [[-1, -1, 5], [-1, 0, 6], [-1, +1, 7], [0, +1, 0], [+1, +1, 1]]
        }

        endBifList = endBifDict.keys()
        for coord, dirList in endBifDict.items():
            y = coord[0]
            x = coord[1]
            for part_dir in dirList:
                tmpY = []
                tmpX = []
                segTmpList = []
                if part_dir == 0:
                    cy = y
                    cx = x + 1
                elif part_dir == 1:
                    cy = y + 1
                    cx = x + 1
                elif part_dir == 2:
                    cy = y + 1
                    cx = x
                elif part_dir == 3:
                    cy = y + 1
                    cx = x - 1
                elif part_dir == 4:
                    cy = y
                    cx = x - 1
                elif part_dir == 5:
                    cy = y - 1
                    cx = x - 1
                elif part_dir == 6:
                    cy = y - 1
                    cx = x
                elif part_dir == 7:
                    cy = y - 1
                    cx = x + 1
                if visited[cy, cx] == 1: continue

                # increase segment id(segID)
                segId += 1
                tmpY.append(y)
                tmpX.append(x)
                tmpY.append(cy)
                tmpX.append(cx)
                segTmpList.append((y, x))
                segTmpList.append((cy, cx))
                visited[y, x] = 1
                visited[cy, cx] = 1

                # 두 칸짜리 토막
                if (cy, cx) in endBifList: continue
                switch = True
                while switch:
                    # investigate front pixels
                    moveLocs = frontLoc[part_dir]
                    for my, mx, sub_dir in moveLocs:
                        frontCoord = (cy + my, cx + mx)
                        if img[frontCoord[0], frontCoord[1]] == 1:
                            if frontCoord in endBifList:  # 마디 끝
                                tmpY.append(frontCoord[0])
                                tmpX.append(frontCoord[1])
                                segTmpList.append(frontCoord)
                                visited[frontCoord[0], frontCoord[1]] = 1
                                switch = False  # while loop 문 끝
                                break
                            # 만일 frontCoord가 이민 segTmpList 내에 존재하면, 원형에서 돌고 있으므로 벗어나야 한다.
                            if frontCoord in segTmpList:
                                switch = False
                                break
                            tmpY.append(frontCoord[0])
                            tmpX.append(frontCoord[1])
                            segTmpList.append(frontCoord)
                            visited[frontCoord[0], frontCoord[1]] = 1
                            cy = frontCoord[0]
                            cx = frontCoord[1]
                            part_dir = sub_dir
                            break

                segDict[segId] = (np.array(tmpY, dtype=np.int64), np.array(tmpX, dtype=np.int64))
        t2 = process_time()
        print("EDGE SEGMENTATION TIME : {0}s".format(t2 - t1))
        return segDict

    @staticmethod
    @njit(int64[:, :](float32[:], float32[:], float32[:], float32[:], int64))
    def faster(edgeYIndex, edgeXIndex, radList, thetaList, D):

        accArr = np.zeros((D * 2, 181)).astype(np.int64)

        for edgeY, edgeX in zip(edgeYIndex, edgeXIndex):
            pCeil = np.ceil(edgeY * np.cos(radList) + edgeX * np.sin(radList))
            pCeil = pCeil + D
            tmpThetaList = thetaList.astype(np.int64) + 90
            for p, t in zip(pCeil.astype(np.int64), tmpThetaList.astype(np.int64)):
                accArr[p, t] = accArr[p, t] + 1

        return accArr

    @staticmethod
    def hough(edge_img,
              thr,
              unit=1,
              img=None,
              time=None):
        # img shape
        shape = edge_img.shape

        # set range of D, p
        D = int(np.sqrt(shape[0] * shape[0] + shape[1] * shape[1]))
        thetaList = np.array([i for i in range(-90, 90 + 1, unit)])  # degree(theta)
        # pList = np.array([p for p in range(-D, D+1, unit)]) # distance(p)

        # get edge index
        edgeIndex = np.where(edge_img == 1)
        radList = np.deg2rad(thetaList)

        accArr = linedetection.faster(edgeYIndex=edgeIndex[0].astype(np.float32),
                                      edgeXIndex=edgeIndex[1].astype(np.float32),
                                      radList=radList.astype(np.float32),
                                      thetaList=thetaList.astype(np.float32),
                                      D=D)
        fig = plt.pyplot.figure()
        plt.pyplot.imshow(img)
        lineIdx = np.where(accArr >= thr)

        x = np.array([i for i in range(0, shape[1])])
        for p, t in zip(lineIdx[0], lineIdx[1]):
            pList = (-np.sin(np.deg2rad(t - 90)) * x + p - D) // np.cos(np.deg2rad(t - 90))
            plt.pyplot.plot(x, pList, color='orange', alpha=0.1)
        plt.pyplot.xlim(0, shape[1])
        plt.pyplot.ylim(0, shape[0])
        plt.pyplot.gca().invert_yaxis()
        fig.canvas.draw()
        f_arr = np.array(fig.canvas.renderer._renderer)

        return f_arr
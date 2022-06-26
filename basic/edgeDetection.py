import os
import cv2
import numpy as np
import numba as nb
from PIL import Image
from time import process_time
from numba import njit, jit, int16, int32, int64, float32, float64, typed

import filters
from tools.video import *
from utils.utils import *

class edgeDetection :

    @staticmethod
    def edgeDirection(Dy,
                      Dx,
                      Dyx=None,
                      mode=0):
        """
        -27.5 <= deg < 27.5 : 0
        27.5 <= deg < 72.5 : 1
        72.5 <= deg < 117.5 : 2
        117.5 <= deg < 162.5 : 3
        162.5 <= deg < 180 : 4
        -180 <= deg < -162.5 : 4
        -162.5 <= deg < -117.5 : 5
        -117.5 <= deg < -72.5 : 6
        -72.5 <= deg < -27.5 : 7
        90 <= deg < 135
        :param Dy: Dx calculated by sobel y
        :param Dx: Dy calculated by sobel x
        :param mode : 0 - default / 1 - de genzo
        :return:
        """
        if mode not in [0, 1]: raise ValueError("MODE has 2 values : 0 - default, 1 - di genzo mode.")
        # if (mode==1) & (Dyx==None) : raise ValueError("It is necessary for MODE=1(De genzo) to have Dyx value.")
        if mode == 0:
            edgeDirMapRaw = np.rad2deg(np.arctan2(Dy, Dx))
        elif mode == 1:
            edgeDirMapRaw = np.rad2deg(np.arctan2(2 * Dyx, (Dx - Dy)))
        edgeDirMapQuant = np.zeros_like(edgeDirMapRaw)
        edgeDirMapQuant[np.where((edgeDirMapRaw >= -27.5) & (edgeDirMapRaw < 27.5))] = 0
        edgeDirMapQuant[np.where((edgeDirMapRaw >= 27.5) & (edgeDirMapRaw < 72.5))] = 1
        edgeDirMapQuant[np.where((edgeDirMapRaw >= 72.5) & (edgeDirMapRaw < 117.5))] = 2
        edgeDirMapQuant[np.where((edgeDirMapRaw >= 117.5) & (edgeDirMapRaw < 162.5))] = 3
        edgeDirMapQuant[np.where((edgeDirMapRaw >= 162.5) & (edgeDirMapRaw <= 180))] = 4
        edgeDirMapQuant[np.where((edgeDirMapRaw >= -180) & (edgeDirMapRaw < -162.5))] = 4
        edgeDirMapQuant[np.where((edgeDirMapRaw >= -162.5) & (edgeDirMapRaw < -117.5))] = 5
        edgeDirMapQuant[np.where((edgeDirMapRaw >= -117.5) & (edgeDirMapRaw < -72.5))] = 6
        edgeDirMapQuant[np.where((edgeDirMapRaw >= -72.5) & (edgeDirMapRaw < -27.5))] = 7

        return edgeDirMapRaw, edgeDirMapQuant

    @staticmethod
    def NMS(edgeMap,
            edgeDirMapQuant,
            edge_dir=8):
        """
        NMS : None Maximum Suppression
        :return:
        """
        edgeMap = np.pad(edgeMap, (1, 1))

        edge_dict = {
            0: {'x': [+0, -0], 'y': [+1, -1]},
            1: {'x': [+1, -1], 'y': [-1, +1]},
            2: {'x': [+1, -1], 'y': [+0, -0]},
            3: {'x': [+1, -1], 'y': [+1, -1]},
            4: {'x': [+0, -0], 'y': [-1, +1]},
            5: {'x': [+1, -1], 'y': [+1, -1]},
            6: {'x': [+0, -0], 'y': [-1, +1]},
            7: {'x': [+1, -1], 'y': [+1, -1]},
        }

        for dir in range(edge_dir):
            dir_dict = edge_dict[dir]
            y = dir_dict['y']
            x = dir_dict['x']

            coord = np.where(edgeDirMapQuant == dir)
            coord1 = np.copy(coord)
            coord2 = np.copy(coord)

            # 비교대상 추출
            coord1 = (coord1[0] + y[0], coord1[1] + x[0])
            coord2 = (coord2[0] + y[1], coord2[1] + x[1])

            # extract target suppressed
            sup_target = ((edgeMap[coord] <= edgeMap[coord1]) | (edgeMap[coord] <= edgeMap[coord2]))
            coord_list = np.array([coord[0], coord[1]])
            idx_target = coord_list[:, sup_target]
            edgeMap[np.array(idx_target[0, :]), np.array(idx_target[1, :])] = 0

        edgeMap = edgeMap[1:edgeMap.shape[0] - 1, 1:edgeMap.shape[1] - 1]
        return edgeMap

    @staticmethod
    def hys_thr(edgeMap,
                tLow,
                tHigh):

        edge = np.zeros_like(edgeMap)
        visited = np.zeros_like(edgeMap)
        shape = edge.shape

        q = queue()
        y_coord = [-1, +1, -1, +1, -1, +1, 0, 0]
        x_coord = [0, 0, -1, +1, +1, -1, -1, +1]
        q.enqueue([0, 0])

        for yj in range(0, shape[0] - 1):
            for xi in range(0, shape[1] - 1):

                if visited[yj, xi] == 0:
                    visited[yj, xi] = 1
                    if edgeMap[yj, xi] > tHigh:
                        if yj != 0 and xi != 0: q.enqueue([yj, xi])
                        edge[yj, xi] = 1

                while len(q.q_list) != 0:
                    for _ in q.q_list:
                        yj, xi = q.dequeue()
                        for ym, xm in zip(y_coord, x_coord):
                            yNew = yj + ym
                            xNew = xi + xm
                            # 영상의 범위 안에 들어오고
                            if yNew >= 0 and yNew < edge.shape[0] and xNew >= 0 and xNew < edge.shape[1]:
                                if visited[yNew, xNew] == 0:
                                    visited[yNew, xNew] = 1
                                    if edgeMap[yNew, xNew] > tLow:
                                        q.enqueue([yNew, xNew])
                                        edge[yNew, xNew] = 1

        return edge

    @staticmethod
    def canny_edge(img,
                   tLow,
                   tHigh,
                   resize_scale=0,
                   ksize=3,
                   sigmaX=1,
                   sigmaY=1,
                   time=None):

        t1 = process_time()
        # load image
        img = img
        # resize image
        if resize_scale: img = cv2.resize(img, dsize=(0, 0), fx=resize_scale, fy=resize_scale)

        # distinguish GRAY or RGB
        if len(img.shape) == 2:
            channels = 1  # gray scale
        elif len(img.shape) >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
            channels = img.shape[2]  # RGB or over than RGB channels
        # Gaussian Blurring
        imgBlurred = cv2.GaussianBlur(img,
                                      ksize=(ksize, ksize),
                                      sigmaX=sigmaX,
                                      sigmaY=sigmaY)
        if len(imgBlurred.shape) == 2: imgBlurred = np.expand_dims(imgBlurred, -1)
        img_frame = np.zeros_like(imgBlurred)

        for channel in range(channels):

            gray_scale = imgBlurred[:, :, channel]

            # calculate edge-map
            Dy, Dx = filter.sobel(gray_scale)
            edgeMap = np.sqrt(Dx * Dx + Dy * Dy).astype(np.float)
            raw, quant = edgeDetection.edgeDirection(Dy, Dx)

            # None Maximum Suppression
            edgeMagnitude = edgeDetection.NMS(edgeMap=edgeMap,
                                edgeDirMapQuant=quant)

            # hysteresis thresholding
            edge = edgeDetection.hys_thr(edgeMap=edgeMagnitude,
                                         tLow=tLow,
                                         tHigh=tHigh)
            img_frame[:, :, channel] = edge
        t2 = process_time()
        if time : print("CANNY PROCESSING TIME : {0}s".format(t2 - t1))
        return img_frame

    @staticmethod
    def canny_edge_genzo(img,
                         tLow,
                         tHigh,
                         resize_scale=0,
                         ksize=3,
                         sigmaX=3,
                         sigmaY=3):

        t1 = process_time()
        # load image
        img = img
        # reszie the image
        if resize_scale: img = cv2.resize(img, dsize=(0, 0), fx=resize_scale, fy=resize_scale)

        # Gaussian blurring
        imgBlurred = cv2.GaussianBlur(img,
                                      ksize=(ksize, ksize),
                                      sigmaX=sigmaX,
                                      sigmaY=sigmaY)

        # edge intensity, direction
        Dry, Drx = filters.sobel(imgBlurred[:, :, 0])  # Red derivation
        Dgy, Dgx = filters.sobel(imgBlurred[:, :, 1])  # Green derivation
        Dby, Dbx = filters.sobel(imgBlurred[:, :, 2])  # Blue derivation

        gyy = Dry * Dry + Dgy * Dgy + Dby * Dby
        gxx = Drx * Drx + Dgx * Dgx + Dbx + Dbx
        gyx = Dry * Drx + Dgy * Dgx + Dby * Dbx

        raw, quant = edgeDetection.edgeDirection(Dy=gyy,
                                                 Dx=gxx,
                                                 Dyx=gyx,
                                                 mode=1)
        edgeMagnitude = np.sqrt(0.5 * ((gyy + gxx) - (gxx - gyy) * np.cos(2 * raw) + 2 * gyx * np.sin(2 * raw)))

        # None Maximum Suppression
        edgeMap = edgeDetection.NMS(edgeMap=edgeMagnitude,
                                    edgeDirMapQuant=quant)

        # hysteresis thresholding
        edge = edgeDetection.hys_thr(edgeMap=edgeMap,
                                     tLow=tLow,
                                     tHigh=tHigh)
        t2 = process_time()
        return edge

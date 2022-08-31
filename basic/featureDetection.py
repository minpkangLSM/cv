import os
import cv2
import numpy as np
import numba as nb
from PIL import Image
from time import process_time
from matplotlib import pyplot as plt
from numba import njit, jit, int16, int32, int64, float32, float64, typed

import filters
from tools.video import *

class featuredetection :

    @staticmethod
    @njit(int64[:,:](float32[:,:], float32[:,:], float32[:,:], float32, float32))
    def harrisCNMS(gdy2,
                   gdx2,
                   gdyx,
                   k,
                   thr):
        """
        harrisC with None Maximum Suppression for 4 direction
        (8 direction with BROWN2005 추가로 진행할 것 - 220623)
        :param gdy2: gaussian after dy->dy
        :param gdx2: gaussian after dx->dx
        :param gdyx: gaussian after dy->dx
        :param k: default = 0.04
        :param thr: threshold for determinant value of A matrix
        :return:
        """
        shape = gdy2.shape
        CMap = np.zeros_like(gdy2.astype(float32))
        featImg = np.zeros_like(gdy2.astype(int64))

        # calculate CMap
        for v in range(shape[0]):
            for u in range(shape[1]):
                A = np.array([[gdy2[v,u], gdyx[v,u]],
                              [gdyx[v,u], gdx2[v,u]]])
                C = np.linalg.det(A)-k*((A[0,0]+A[1,1])**2)
                CMap[v,u] = C
        # create featImg with NMS for 4 dir.
        for v in range(1,shape[0]-1):
            for u in range(1,shape[1]-1):
                if (CMap[v,u] > thr) and (CMap[v,u]>CMap[v+1,u]) and (CMap[v,u]>CMap[v-1,u]) and (
                        CMap[v,u]>CMap[v,u+1]) and (CMap[v,u]>CMap[v,u-1]) :
                    featImg[v,u]=1

        return featImg

    @staticmethod
    @njit(int64[:,:](float32[:,:], float32[:,:], float32[:,:], float32, float32))
    def harrisC(gdy2,
                gdx2,
                gdyx,
                k,
                thr):
        """
        harrisC
        :param gdy2: gaussian after dy->dy
        :param gdx2: gaussian after dx->dx
        :param gdyx: gaussian after dy->dx
        :param k: default = 0.04
        :param thr: threshold for determinant value of A matrix
        :return:
        """
        shape = gdy2.shape
        featImg = np.zeros_like(gdy2.astype(int64))
        for v in range(shape[0]):
            for u in range(shape[1]):
                A = np.array([[gdy2[v,u], gdyx[v,u]],
                              [gdyx[v,u], gdx2[v,u]]])
                C = np.linalg.det(A)-k*((A[0,0]+A[1,1])**2)
                if (C >= thr) and (C) : featImg[v,u] = 1

        return featImg

    @staticmethod
    @njit(int64[:, :](float32[:, :], float32[:, :], float32[:, :], float32, int16))
    def hessianCNMS(dy2,
                    dx2,
                    dyx,
                    thr,
                    cmode):
        """
        hessian C with None Maximum Suppression for 8 dir
        :param dy2: dyy after gaussian
        :param dx2: dxx after gaussian
        :param dyx: dy -> dx, after gaussian
        :param thr: threshold for determinant value of A matrix
        :param cmode:
               1) cmode = 0 : hessian determinant
               2) cmode = 1 : laplacian determinant
        :return:
        """

        shape = dy2.shape
        CMap = np.zeros_like(dy2.astype(float32))
        featImg = np.zeros_like(dy2.astype(int64))
        # calculate CMap
        for v in range(shape[0]):
            for u in range(shape[1]):
                A = np.array([[dy2[v, u], dyx[v, u]],
                              [dyx[v, u], dx2[v, u]]])
                # hessian determinant
                if cmode == 0:
                    C = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
                # gaussian laplacian(LOG)
                elif cmode == 1:
                    C = A[0, 0] + A[1, 1]

                CMap[v,u] = C

        # create featImg with NMS for 4 dir.
        for v in range(1, shape[0] - 1):
            for u in range(1, shape[1] - 1):
                if (CMap[v, u] > thr) and (CMap[v, u] > CMap[v + 1, u]) and (CMap[v, u] > CMap[v - 1, u]) and (
                        CMap[v, u] > CMap[v, u + 1]) and (CMap[v, u] > CMap[v, u - 1]):
                            featImg[v, u] = 1

        return featImg

    @staticmethod
    @njit(int64[:,:](float32[:,:], float32[:,:], float32[:,:], float32, int16))
    def hessianC(dy2,
                 dx2,
                 dyx,
                 thr,
                 cmode):
        """
        hessian C with None Maximum Suppression for 8 dir
        :param dy2: dyy after gaussian
        :param dx2: dxx after gaussian
        :param dyx: dy -> dx, after gaussian
        :param thr: threshold for determinant value of A matrix
        :param cmode:
               1) cmode = 0 : hessian determinant
               2) cmode = 1 : laplacian determinant
        :return:
        """

        shape = dy2.shape
        featImg = np.zeros_like(dy2.astype(int64))

        # calculate CMap
        for v in range(shape[0]):
            for u in range(shape[1]):
                A = np.array([[dy2[v,u], dyx[v,u]],
                              [dyx[v,u], dx2[v,u]]])
                # hessian determinant
                if cmode==0 : C=A[0,0]*A[1,1]-A[1,0]*A[0,1]
                # gaussian laplacian(LOG)
                elif cmode==1 : C=A[0,0]+A[1,1]

                if C >= thr : featImg[v,u] = 1

        return featImg

    @staticmethod
    def harrisCorner(img,
                     ksize,
                     sigmaX,
                     sigmaY,
                     thr,
                     k=0.01):

        # counting dy, dx (process time : 0.00000s, 512x512)
        dy, dx = filters.sobel(img)

        # calculate component of A matrix (processing time : 0.00000s, 512x512) -> dx^2 : 를 왜 dx x dx로 하지? 그냥 x로 미분 더 해야하는거 아닌가?
        gdy2 = filters.gaussian(img=dy*dy,
                               ksize=ksize,
                               sigmaX=sigmaX,
                               sigmaY=sigmaY).astype(np.float32) # dtype : float32
        gdx2 = filters.gaussian(img=dx*dx,
                               ksize=ksize,
                               sigmaX=sigmaX,
                               sigmaY=sigmaY).astype(np.float32) # dtype : float32
        gdyx = filters.gaussian(img=dy*dx,
                               ksize=ksize,
                               sigmaX=sigmaX,
                               sigmaY=sigmaY).astype(np.float32) # dtype : float32

        # calculate A matrix for each points of the img(process time : 0.32s, 512x512)
        feature = featuredetection. CNMS(gdy2=gdy2,
                                              gdx2=gdx2,
                                              gdyx=gdyx,
                                              k=k,
                                              thr=thr)
        idx = np.where(feature==1)
        img[idx] = 255
        return img

    @staticmethod
    def hessian(img,
                ksize,
                sigmaX,
                sigmaY,
                thr,
                cmode):
        # counting dy, dx (process time : 0.00000s, 512x512)
        gImg = filters.gaussian(img=img,
                               ksize=ksize,
                               sigmaX=sigmaX,
                               sigmaY=sigmaY)

        # calculate hessian matrix factor (process time : 0.015625s, 512x512)
        dy, dx = filters.sobel(gImg)
        dy2, dyx = filters.sobel(dy)
        dx2, dxy = filters.sobel(dx)

        # generate feature map (process time : 0.1875s, 512x512)
        feature = featuredetection.hessianCNMS(dy2=dy2.astype(np.float32),
                                            dx2=dx2.astype(np.float32),
                                            dyx=dyx.astype(np.float32),
                                            thr=thr,
                                            cmode=cmode)
        idx = np.where(feature == 1)
        img[idx] = 255
        return img

class multiscale :

    @staticmethod
    @njit(int64[:,:](float32[:,:], float32[:,:], float32[:,:], float32, float32))
    def featureMap(Gdy2,
                   Gdx2,
                   Gdydx,
                   k,
                   thr):

        shape = Gdy2.shape
        featImg = np.zeros_like(Gdy2.astype(int64))
        for v in range(shape[0]):
            for u in range(shape[1]):
                A = np.array([[Gdy2[v,u], Gdydx[v,u]],
                              [Gdydx[v,u], Gdx2[v,u]]])
                C = np.linalg.det(A)-k*((A[0,0]+A[1,1])**2)
                if C >= thr : featImg[v,u] = 1
        return featImg


    @staticmethod
    def scaleMap(img,
                 sigma,
                 depth,
                 thr,
                 ksi=1.4,
                 scale=0.7):

        # generate scale map
        shape = img.shape
        scaleMap = np.zeros((shape[0], shape[1], depth))
        sigmaMap = {}

        # STEP 1 : find local feature in 2D for each channels
        for idx in range(depth):
            # set sigma
            sigmaN = (ksi**idx) * sigma
            sigmaI = sigmaN
            sigmaD = scale * sigmaN
            # make d(sigma)
            imgGaus = filters.gaussian(img,
                                       sigmaX=sigmaD,
                                       sigmaY=sigmaD)
            dy, dx = filters.sobel(imgGaus)
            Gdy2 = (sigmaD**2)*filters.gaussian(img=dy*dy,
                                                sigmaX=sigmaI,
                                                sigmaY=sigmaI).astype(np.float32)
            Gdx2 = (sigmaD**2)*filters.gaussian(img=dx*dx,
                                                sigmaX=sigmaI,
                                                sigmaY=sigmaI).astype(np.float32)
            Gdydx = (sigmaD**2)*filters.gaussian(img=dy*dx,
                                                 sigmaX=sigmaI,
                                                 sigmaY=sigmaI).astype(np.float32)
            sigmaIFeatMap = multiscale.featureMap(Gdy2=Gdy2,
                                                  Gdx2=Gdx2,
                                                  Gdydx=Gdydx,
                                                  k=0.04,
                                                  thr=thr)

            scaleMap[:,:,idx] = sigmaIFeatMap
            sigmaMap[idx] = sigmaI

        # STEP 2 : select SCALE in 3D axis
        xyIdx = np.where(scaleMap==1)
        for y, x, s in zip(xyIdx[0], xyIdx[1], xyIdx[2]):
            sigma = sigmaMap[s]
            while True:
                subSigma = 0.7*sigma
                subLaplaceList = []
                while subSigma <= 1.7*sigma :
                    subGauImg = filters.gaussian(img=img,
                                                 sigmaX=subSigma,
                                                 sigmaY=subSigma)
                    dy, dx = filters.sobel(subGauImg)
                    # laplacian
                    dyy = dy*dy
                    dxx = dx*dx
                    subLaplace = (subSigma**2)*abs(dyy[y,x]+dxx[y,x])
                    if subLaplace >= thr : subLaplaceList.append(subLaplace)
                    subSigma += 1.1

                if len(subLaplaceList)==0 : break
                else : sigmaNew = max(subLaplaceList)


if __name__ == "__main__" :

    # setting image directory
    imgdir = "D:\\cv\\data\\parts\\cannytest.png"
    img = cv2.imread(imgdir, 0)
    multiscale.scaleMap(img=img,
                        sigma=1.0,
                        depth=6,
                        thr=1000)
    # # video version
    # capture = cv2.VideoCapture(0)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 325)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 325)
    #
    # videoReader.videoReader(function=featuredetection.harrisCorner,
    #                         ksize=3,
    #                         sigmaX=3,
    #                         sigmaY=3,
    #                         thr=1000,
    #                         capture=capture,
    #                         cmode=1)

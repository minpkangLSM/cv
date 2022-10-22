"""
SIFT IMPLEMENTATION
from "Distinctive Image Features from Scale-Invariant Keypoints", Lowe, 2004.
Kangmin Park, 2022.10.20.
"""
import cv2
import numpy as np
import numba as nb
from numba import jit, njit, int16, int64, float64
from basic.filters import *
from feature_extractor import *
"""
Chapter 4.
Accurate Keypoint Localization
"""
def localize(dogSpace,
             extremaLocation,
             interval_num,
             offsetThr=0.5,
             convergenceThr=5,
             eigenRatio=10):
    """
    :param dogSpace
    :param extremaLocation
           the order of data : [y, x, sigma]
    :param offsetThr
    :return:
    """
    print("\nCHAPTER 4 : LOCALIZING EXTREMAS FROM DOG SPACE")
    print("\t LOCALIZING EXTREMAS & REMOVING FEATURES ON EDGES...")
    t1 = process_time()
    extremaLocalizedDict = {}
    # localizing
    for octaveIdx in extremaLocation.keys() :

        octave = dogSpace[octaveIdx]
        extremas = extremaLocation[octaveIdx] # y, x, sigma
        extremasTmp = np.stack((extremas[0], # y
                                extremas[1], # x
                                extremas[2]), axis=1)

        extremasLocalized = localizeSub(octave=octave,
                                        extremas=extremasTmp,
                                        interval_num=interval_num,
                                        offsetThr=offsetThr,
                                        convergenceThr=convergenceThr,
                                        eigenRatio=eigenRatio)
        extremaLocalizedDict[octaveIdx] = extremasLocalized

    t2 = process_time()
    print("\t- FINISHED LOCALIZING & REMOVING. TIME : {0}".format(t2-t1))
    print("\t================= RESULTS =================")
    for octaveIdx in extremaLocation.keys() :
        print("\t  [OCTAVE {0}]".format(octaveIdx))
        print("\t  - THE NUM OF KEYPOINTS : BEFORE : {0} -> AFTER {1}".format(len(extremaLocation[octaveIdx][0]),
                                                                              extremaLocalizedDict[octaveIdx].shape[0]))
    print("\t===========================================")

@jit(float64[:,:](float64[:,:,:], int64[:,:], float64, float64, float64, float64))
def localizeSub(octave,
                extremas,
                interval_num,
                offsetThr,
                convergenceThr,
                eigenRatio):

    extremasLocalized = np.zeros_like(extremas).astype(float64)
    cnt = 0
    for y, x, s in extremas:

        convergence = True
        attempt = 0
        while True:

            if attempt > convergenceThr:  # 수렴 시도 횟수를 넘어간 경우
                convergence = False
                break
            if y < 1 or y >= octave.shape[0] - 1 or x < 1 or x >= octave.shape[1] - 1 or s < 1 or s >= octave.shape[
                2] - 1:  # 조정된 값이 영상의 범위를 넘어간 경우
                convergence = False
                break

            # gradient
            dx = 0.5 * (octave[y, x + 1, s] - octave[y, x - 1, s])
            dy = 0.5 * (octave[y + 1, x, s] - octave[y - 1, x, s])
            ds = 0.5 * (octave[y, x, s + 1] - octave[y, x, s - 1])
            gradient = np.array([[dy], [dx], [ds]])
            # Hessian
            dxx = octave[y, x + 1, s] - 2 * octave[y, x, s] + octave[y, x - 1, s]
            dyy = octave[y + 1, x, s] - 2 * octave[y, x, s] + octave[y - 1, x, s]
            dss = octave[y, x, s + 1] - 2 * octave[y, x, s] + octave[y, x, s + 1]
            dxy = 0.25 * (octave[y + 1, x + 1, s] - octave[y + 1, x - 1, s] - octave[y - 1, x + 1, s] + octave[
                y - 1, x - 1, s])
            dys = 0.25 * (octave[y + 1, x, s + 1] - octave[y - 1, x, s + 1] - octave[y + 1, x, s - 1] + octave[
                y - 1, x, s - 1])
            dsx = 0.25 * (octave[y, x + 1, s + 1] - octave[y, x + 1, s - 1] - octave[y, x - 1, s + 1] + octave[
                y, x - 1, s - 1])
            hessian = np.array([[dyy, dxy, dys],
                                [dxy, dxx, dsx],
                                [dys, dsx, dss]])
            hessianInv = np.linalg.inv(hessian)
            # calculate residual(offset), xhat
            xhat = -np.dot(hessianInv, gradient).flatten()  # residual

            if abs(xhat[0]) < offsetThr and abs(xhat[1]) < offsetThr and abs(xhat[2]) < offsetThr: # 수렴
                break
            y += int(round(xhat[0]))
            x += int(round(xhat[1]))
            s += int(round(xhat[2]))
            attempt += 1

        if not convergence:
            continue  # 수렴이 되지 않은 값이면 버린다.

        # continue
        Dxhat = octave[y,x,s] + 0.5 * np.dot(gradient.T, xhat.reshape(3, -1))

        # # chapter4. [4.1. Eliminating Edge Responses]
        if abs(Dxhat[0][0]) * interval_num >= 0.04 :

            trace = hessian[0,0] + hessian[1,1]
            det = hessian[0,0]*hessian[1,1] - hessian[0,1]*hessian[1,0]

            if det > 0 and trace**2 * eigenRatio < det * (eigenRatio+1)**2 :
                extremasLocalized[cnt, 0] = y
                extremasLocalized[cnt, 1] = x
                extremasLocalized[cnt, 2] = s
                cnt+=1

    return extremasLocalized[:int(cnt), :]

if __name__ == "__main__" :

    pass
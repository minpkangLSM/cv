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
             sigma=1.6,
             offsetThr=0.5,
             convergenceThr=5,
             eigenRatio=10):
    """
    :param dogSpace
    :param extremaLocation
           the order of data : [y, x, sigma]
    :param interval_num
           var. s in def scaleSpace
           몇 번째 레이어를 각 옥타브 base layer에 주어지는 시그마의 2배로 할지 결정하는 파라미터
    :param sigma
           각 옥타브 내 존재하는 layer에 가우시안 스무딩을 할 때 사용되는 기준 시그마 값 (default = 1.6)
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
                                        sigma=sigma,
                                        extremas=extremasTmp,
                                        octaveIdx=octaveIdx,
                                        interval_num=interval_num,
                                        offsetThr=offsetThr,
                                        convergenceThr=convergenceThr,
                                        eigenRatio=eigenRatio)
        extremaLocalizedDict[octaveIdx] = extremasLocalized

    t2 = process_time()
    print("\t - FINISHED LOCALIZING & REMOVING. TIME : {0}".format(t2-t1))
    print("\t================= RESULTS =================")
    for octaveIdx in extremaLocation.keys() :
        print("\t  [OCTAVE {0}]".format(octaveIdx))
        print("\t  - THE NUM OF KEYPOINTS : BEFORE : {0} -> AFTER {1}".format(len(extremaLocation[octaveIdx][0]),
                                                                              extremaLocalizedDict[octaveIdx].shape[0]))
    print("\t===========================================")
    return extremaLocalizedDict

@jit(float64[:,:](float64[:,:,:], int64[:,:], float64, float64, float64, float64, float64, float64))
def localizeSub(octave,
                extremas,
                sigma,
                octaveIdx,
                interval_num,
                offsetThr,
                convergenceThr,
                eigenRatio):

    # extremasLocalized = np.zeros_like(extremas).astype(np.float64)
    extremasLocalizedMod = np.zeros_like(extremas).astype(np.float64)

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

        # Dxhat : 해당 위치의 extremum의 contrast값 계산
        Dxhat = octave[y,x,s] + 0.5 * np.dot(gradient.T, xhat.reshape(3, -1))

        # # chapter4. [4.1. Eliminating Edge Responses]
        if abs(Dxhat[0][0]) * interval_num >= 0.04 :

            trace = hessian[0,0] + hessian[1,1]
            det = hessian[0,0]*hessian[1,1] - hessian[0,1]*hessian[1,0]

            if det > 0 and trace**2 * eigenRatio < det * (eigenRatio+1)**2 :

                # # 옥타브 그대로 저장한 버전
                # extremasLocalized[cnt, 0] = y
                # extremasLocalized[cnt, 1] = x
                # extremasLocalized[cnt, 2] = s

                # 옥타브 생성하기 전 위치로 변환하여 저장한 버전 (모두 옥타브 0으로 위치/범위 변경) - Medium 참고
                extremasLocalizedMod[cnt, 0] = (y + xhat[0]) * (2 ** octaveIdx) # 옥타브에 따라 1/2씩 영상의 크기가 줄어든 점을 고려하여 위치 원복
                extremasLocalizedMod[cnt, 1] = (x + xhat[1]) * (2 ** octaveIdx) # 옥타브에 따라 1/2씩 영상의 크기가 줄어든 점을 고려하여 위치 원복
                # 각 키포인트들의 scale(=sigma) 값을 원영상 기준으로 원복
                # 각 옥타브 내 레이어에 위치한 키포인트들의 k값이 곱해지는 원리를 생각해보면 된다. -> 옥타브가 올라갈 때마다 2^(octaveIdx)만큼 배율이 증가
                extremasLocalizedMod[cnt, 2] = sigma * (2 ** ((s + xhat[2]) / np.float32(interval_num))) * (2 ** (octaveIdx + 1))  # octave_index + 1 because the input image was doubled

                keypoinOctave = octaveIdx + s * (2 ** 8) + int(round((xhat[2] + 0.5) * 255)) * (2 ** 16)
                response = abs(Dxhat[0][0])

                cnt+=1

    return extremasLocalizedMod[:int(cnt), :] #extremasLocalized[:int(cnt), :]

if __name__ == "__main__" :

    pass
"""
3. Detection of Scale-Space Extrema, Lowe(2004)
- SCALE SPACE, Kang min Park, 2022.08.09.
"""
import cv2
import math
import numpy as np
import numba as nb
from numba import jit, njit, uint8, int64
from basic.filters import *
from basic.geometry import *
from time import process_time
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

class feature :

    @staticmethod
    def scaleSpace(img,
                     s,
                     octaveNum,
                     sigma=1.6):
       """
       the number of images in the stack of blurred images for each octave : s+3
       :param img: image type = gray scale
       :param s: a parameter for constant factor k = 2^(1/s)
                 s는 이전 옥타브 내 s번째 이미지를 다음 옥타브의 첫 번째 이미지로 삼겠다는 의미가 된다.
       :param octaveNum: the number of octaves
       :param sigma : gaussian blurr parameter
       :return:
       """
       # set parameters of scale space
       k = 2**(1/s) # gFactor = k, sigma factor
       octaveLayer = img
       scaleSpace = {}
       sigmas = {}

       # create scale space
       for oIdx in range(octaveNum):
           initSigma = sigma
           sigmaList = []
           # create each octave
           for iIdx in range(s+3):

               if iIdx==0 and oIdx!=0:
                   octave = octaveLayer[:,:,np.newaxis]
                   continue

               # set sigma factor
               initSigma = initSigma * (k ** iIdx)
               if iIdx==0 and oIdx==0 : sigmaDiff = np.sqrt(initSigma**2 - 0.5**2)
               else : sigmaDiff = np.sqrt(initSigma**2 - prevSigma**2)

               sigmaList.append(initSigma)

               # make a layer
               octaveLayer = gaussian(octaveLayer,
                                      sigmaX=sigmaDiff,
                                      sigmaY=sigmaDiff)[:,:,np.newaxis]
               # make octave
               if iIdx==0 : octave = octaveLayer
               else : octave = np.concatenate([octave, octaveLayer], axis=-1)

               prevSigma = initSigma

           scaleSpace[oIdx] = octave
           sigmas[oIdx] = sigmaList
           prevSigma = sigma
           octaveLayer = cv2.resize(src=octave[:,:,s],
                                    dsize=None,
                                    fy=0.5,
                                    fx=0.5,
                                    interpolation=cv2.INTER_AREA)
       return scaleSpace, sigmas

    @staticmethod
    def dog(scaleSpace):
        """
        order) feature class scale space -> dog
        create difference of gaussian (DoG) space from scale space of feature.scalespace function.
        :param scaleSpace:
        :return: dog space, dtype = dictionary
        """
        # setting
        dogSpace = {}
        octaves = scaleSpace.keys()

        for idx in octaves :
            octave = scaleSpace[idx]
            floor = octave.shape[2]
            for sub_idx in range(0, floor-1):
                l1 = octave[:,:,sub_idx]
                l2 = octave[:,:,sub_idx+1]
                diff = l1-l2
                diff = diff[:,:,np.newaxis]
                if sub_idx == 0 : dogOctave = diff
                else : dogOctave = np.concatenate([dogOctave, diff], axis=-1)

            dogSpace[idx] = dogOctave

        return dogSpace

    @staticmethod
    @jit (uint8[:,:,:](uint8[:,:,:]))
    def __extremumSub(octave):
        """
        SUB function for locate extremum in DOG space
        :param octave:
        :return:
        """

        boolBox = np.ones_like(octave)
        extrBox = np.zeros_like(octave).astype(uint8)

        for x in range(1, boolBox.shape[1]-1):
            for y in range(1, boolBox.shape[0]-1):
                for z in range(1, boolBox.shape[2]-1):

                    if boolBox[y, x, z]==0 : continue

                    dVal = octave[y, x, z]
                    sample = octave[y-1:y+2, x-1:x+2, z-1:z+2]
                    results = np.sum((dVal>sample).flatten())-1
                    if results==0 or results==26 :
                        boolBox[y-1:y+2, x-1:x+2, z-1:z+2] = 0
                        extrBox[y, x, z] = 1

        return extrBox

    @staticmethod
    def extremum(dogSpace, sigmas):
        """
        order) scale space -> dog -> extremum
        select the candidate points from scale space from feature.scaleSpace function.
        :param scaleSpace:
        :param sigmas:
        :return:
        """
        dogIdx = dogSpace.keys()
        extremum = {}

        for key in dogIdx :

            octave = dogSpace[key]
            extreBox = feature.__extremumSub(octave)
            extIdx = np.where(extreBox==1)
            extremum[key] = extIdx

        return extremum



if __name__ == "__main__":

    imgDir = "D:\\cv\\data\\prac\\cannytest.png"
    img = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200))
    deg = 0
    img_rot = rotation(img=img,
                       theta=deg)
    t1 = process_time()

    ss, sigmas = feature.scaleSpace(img=img_rot,
                                    s=3,
                                    octaveNum=5)
    ds = feature.dog(ss)
    ex = feature.extremum(ds, sigmas)
    t2 = process_time()
    print("Process time : ", t2 - t1)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    for idx in ex.keys():
        if idx!=0 : continue
        x_list = []
        y_list = []

        # deg rotation -> 그냥 로테이션 원상복귀만 시키는 파트 (scale과는 아무 관련 없음)
        # X, Y, Z 순서 주의할 것 -> 영상은 기본적으로 (Height, Width, Channel) = (Y, X, Z) 순서임
        shift = tran(-img.shape[1]/2, -img.shape[0]/2) # Homogeneous matrix는 x, y 순서로 받아야 한다.
        shift_rev = tran(img.shape[1]/2, img.shape[0]/2)
        rot_matrix = rot(deg)
        h_matrix = np.linalg.multi_dot([shift_rev, rot_matrix, shift])

        coord_homo = np.stack([ex[idx][1],
                               ex[idx][0],
                               np.ones_like(ex[idx][0])], axis=0).astype(np.int64)
        source_coord = np.dot(lin.inv(h_matrix), coord_homo) # 결과로 받은 source_coord는 [x, y] 순서임

        # 사격형 그리기 -> scale 관련
        for x, y, z in zip(source_coord[0,:], source_coord[1,:], ex[idx][2]):
            # scale(시그마 말고)에 따른, 사각형의 중심위치
            x = x*(2**idx) + (2**idx-1)/2
            y = y*(2**idx) + (2**idx-1)/2
            x_list.append(x)
            y_list.append(y)

            # 범위에 따른 사각형
            rec = Rectangle((x-math.floor(6*sigmas[idx][z]/2), y-math.floor(6*sigmas[idx][z]/2)),
                            math.floor(6 * sigmas[idx][z]), math.floor(6 * sigmas[idx][z]),
                            linewidth=1,
                            edgecolor='r',
                            facecolor='none')
            ax.add_patch(rec)
            ax.scatter(x_list, y_list)

    plt.show()
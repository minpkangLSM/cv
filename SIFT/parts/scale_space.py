"""
3. Detection of Scale-Space Extrema, Lowe(2004)
- SCALE SPACE, Kang min Park, 2022.08.09.
"""
import cv2
import numpy as np
from basic.filters import *
from time import process_time
from matplotlib import pyplot as plt

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

       # create scale space
       for oIdx in range(octaveNum):
           initSigma = sigma

           # create each octave
           for iIdx in range(s+3):

               if iIdx==0 and oIdx!=0:
                   octave = octaveLayer[:,:,np.newaxis]
                   continue

               # set sigma factor
               initSigma = initSigma * (k ** iIdx)
               if iIdx==0 and oIdx==0 : sigmaDiff = np.sqrt(initSigma**2 - 0.5**2)
               else : sigmaDiff = np.sqrt(initSigma**2 - prevSigma**2)

               # make a layer
               octaveLayer = gaussian(octaveLayer,
                                      sigmaX=sigmaDiff,
                                      sigmaY=sigmaDiff)[:,:,np.newaxis]
               # make octave
               if iIdx==0 : octave = octaveLayer
               else : octave = np.concatenate([octave, octaveLayer], axis=-1)

               prevSigma = initSigma

           scaleSpace[oIdx] = octave
           prevSigma = sigma
           octaveLayer = cv2.resize(src=octave[:,:,s],
                                    dsize=None,
                                    fy=0.5,
                                    fx=0.5,
                                    interpolation=cv2.INTER_AREA)

       return scaleSpace

if __name__ == "__main__":

    imgDir = "D:\\cv\\data\\prac\\cannytest.png"
    img = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE)
    t1 = process_time()
    ss = feature.scaleSpace(img=img,
                            s=2,
                            octaveNum=4)
    t2 = process_time()

    for i in ss.keys() :
        plt.subplot(231)
        plt.imshow(ss[i][:, :, 0], cmap='gray')
        plt.subplot(232)
        plt.imshow(ss[i][:, :, 1], cmap='gray')
        plt.subplot(233)
        plt.imshow(ss[i][:, :, 2], cmap='gray')
        plt.subplot(234)
        plt.imshow(ss[i][:, :, 3], cmap='gray')
        plt.subplot(235)
        plt.imshow(ss[i][:, :, 4], cmap='gray')
        plt.show()
        
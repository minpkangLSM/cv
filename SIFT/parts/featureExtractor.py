"""
3. Detection of Scale-Space Extrema, Lowe(2004)
- SCALE SPACE, Kang min Park, 2022.08.09.
"""
import cv2
import math
import copy
import numpy as np
from numba import jit, njit, uint8, int16, int64, float64
from basic.filters import *
from basic.geometry import *
from time import process_time
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

class extract_feature :

    @staticmethod
    def scaleSpace(img,
                   s,
                   octaveNum,
                   sigma=1.6):
       """
       STEP1) build scale space
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
        STEP2) calculate difference of gaussian for scale space
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
    def extractExtremum(dogSpace):
        """
        STEP3) extract extremum from dog
        order) scale space -> dog -> extremum
        select the candidate points from scale space from feature.scaleSpace function.
        :param scaleSpace:
        :param sigmas:
        :return:
        """
        dogIdx = dogSpace.keys()
        extremum = {}

        # STEP 1) extract sample points(of extremum)
        """
        dog : scale space에 대하여 difference를 수행한 scale space
        extremum : dog scale space 상에서, 극점을 갖는 위치를 갖는 변수
        """
        for idx in dogIdx:
            octave = dogSpace[idx]
            extreBox = extract_feature.__extremumSub(octave)
            extIdx = np.where(extreBox == 1)
            extremum[idx] = extIdx

        return extremum

    @staticmethod
    @jit(uint8[:, :, :](uint8[:, :, :]))
    def __extremumSub(octave):
        """
        SUB function of def. extractExtremum for locate extremum in DOG space
        :param octave:
        :return:
        """
        boolBox = np.ones_like(octave)
        extrBox = np.zeros_like(octave).astype(uint8)
        for x in range(1, boolBox.shape[1] - 1):
            for y in range(1, boolBox.shape[0] - 1):
                for z in range(1, boolBox.shape[2] - 1):
                    if boolBox[y, x, z] == 0: continue
                    dVal = octave[y, x, z]
                    sample = octave[y - 1:y + 2, x - 1:x + 2, z - 1:z + 2]
                    results = np.sum((dVal > sample).flatten()) - 1
                    if results == 0 or results == 26:
                        boolBox[y - 1:y + 2, x - 1:x + 2, z - 1:z + 2] = 0
                        extrBox[y, x, z] = 1
        return extrBox

    @staticmethod
    def localization(dogSpace,
                     extremum,
                     offsetThr,
                     contrastThr):
        """
        STEP4) extract feature points with contrast
        Contrast가 낮은 키포인트 제거하기
        1) calculate Gradient / Heissan for extremum points
        2) interploate the location of extremum which have value under 0.5
        3) discard the location of extremum which have D(hat x) under contrast threshold
        """
        dogIdx = dogSpace.keys()
        localizedExtremum = {}

        for idx in dogIdx:
            # dog scale space 상 octave 순서대로 진행
            """1. derivate Y"""
            octaveY = dogSpace[idx]
            dy = sobelHeightAxis(octaveY)  # shape (Y, X, Z)
            dy2 = sobelHeightAxis(dy)  # shape (Y, X, Z)
            dyx = sobelWidthAxis(dy)  # shape (Y, X, Z)
            """2. derivate X"""
            octaveX = np.transpose(octaveY, (1, 0, 2))
            dx = sobelHeightAxis(octaveX)  # shape (X, Y, Z)
            dx2 = sobelHeightAxis(dx)  # shape (X, Y, Z)
            dxT = np.transpose(dx, (2, 0, 1))  # shape (Z, X, Y)
            dxz = sobelHeightAxis(dxT)  # shape (Z, X, Y)
            # (Y, X, Z) 순서로 모양 맞추기
            dx = np.transpose(dx, (1, 0, 2))
            dx2 = np.transpose(dx2, (1, 0, 2))  # shape (Y, X, Z)
            dxz = np.transpose(dxz, (2, 1, 0))  # shape (Y, X, Z)
            """3. derivate Z"""
            octaveZ = np.transpose(octaveY, (2, 0, 1))
            dz = sobelHeightAxis(octaveZ)  # shape (Z, Y, X)
            dz2 = sobelHeightAxis(dz)  # shape (Z, Y, X)
            dzy = sobelWidthAxis(dz)  # shape (Z, Y, X)
            # (Y, X, Z) 순서로 모양 맞추기
            dz = np.transpose(dz, (1, 2, 0))
            dz2 = np.transpose(dz2, (1, 2, 0))  # shape (Y, X, Z)
            dzy = np.transpose(dzy, (1, 2, 0))  # shape (Y, X, Z)

            # gradient \frac{dD}{dX}
            samplePoints = extremum[idx]
            dDdx = dx[samplePoints][:, np.newaxis]
            dDdy = dy[samplePoints][:, np.newaxis]
            dDdz = dz[samplePoints][:, np.newaxis]
            gradient = np.concatenate([dDdx, dDdy, dDdz], axis=1)[:, :, np.newaxis]
            # hessian \frac{d^2D}{dX^2}
            dDdx2 = dx2[samplePoints][np.newaxis, :]
            dDdxz = dxz[samplePoints][np.newaxis, :]
            dDdyx = dyx[samplePoints][np.newaxis, :]
            dDdy2 = dy2[samplePoints][np.newaxis, :]
            dDdzy = dzy[samplePoints][np.newaxis, :]
            dDdz2 = dz2[samplePoints][np.newaxis, :]
            hessianTmp = np.concatenate([dDdx2, dDdyx, dDdxz, dDdyx, dDdy2, dDdzy, dDdxz, dDdzy, dDdz2], axis=0)[:, :,
                         np.newaxis]
            hessian = hessianTmp.reshape((-1, 3, 3))
            """interpolate extremum location"""
            XhatVal = np.matmul(hessian / 255, gradient / 255)  # normalize value in range 0 ~ 1
            interpolatedTmp = copy.deepcopy(extremum[idx])
            for no, val in enumerate(XhatVal):
                if any(abs(val)) >= offsetThr:
                    val = val.flatten()
                    interpolatedTmp[0][no] = interpolatedTmp[0][no] - val[0]
                    interpolatedTmp[1][no] = interpolatedTmp[1][no] - val[1]
                    interpolatedTmp[2][no] = interpolatedTmp[2][no] - val[2]
            """remove low contrast value"""
            iTmp = np.concatenate((interpolatedTmp[0][:, np.newaxis],
                                   interpolatedTmp[1][:, np.newaxis],
                                   interpolatedTmp[2][:, np.newaxis]), axis=-1)  # (N, 3)
            iTmp = iTmp[:, :, np.newaxis]  # (N, 3, 1)
            gradientRshp = gradient.transpose((0, 2, 1))
            contrastVal = dogSpace[idx][extremum[idx]] / 255 + 0.5 * np.matmul(gradientRshp / 255, iTmp).flatten()
            contrastValIdx = np.where((contrastVal >= contrastThr) | (contrastVal <= -contrastThr))
            # discard value under contrastThr
            localizedExtremum[idx] = (extremum[idx][0][contrastValIdx],
                                      extremum[idx][1][contrastValIdx],
                                      extremum[idx][2][contrastValIdx])
        return localizedExtremum

    @staticmethod
    def edgeRemover(dogSpace,
                    extremum,
                    sigmaX,
                    sigmaY,
                    r):
        """
        STEP5) Harris corner last step for extract features
        해당 단계는 피쳐(feature, keypoint)에 대한 localization이 끝난 후,
        edge에 존재하는 피쳐포인트를 제거하기 위한 단계로 Harris corner를 사용하여 edge 상에 존재하는 피쳐를 제거한다.
        :param dogSpace:
        :param extremum:
        :param r:
        :return:
        """
        # remove features on the edge using Harris Corner
        extremumOctaveIdx = extremum.keys() # dog space의 모든 옥타브를 조사할 필요는 없고, feature가 존재하는 옥타브에 대해서만 진행하면 됨
        ratio = (r+1)*(r+1)/r
        features = {}
        # calculate derivation for each octave
        for idx in extremumOctaveIdx :
            dogNorm = dogSpace[idx]/255.
            # calculate derivation for earch layer
            dy = sobelHeightAxis(dogNorm,
                                 ddepth=cv2.CV_64F)
            dx = sobelWidthAxis(dogNorm,
                                ddepth=cv2.CV_64F)
            gdy2 = gaussian(img=dy*dy,
                            sigmaX=sigmaX,
                            sigmaY=sigmaY)
            gdx2 = gaussian(img=dx*dx,
                            sigmaX=sigmaX,
                            sigmaY=sigmaY)
            gdyx = gaussian(img=dy*dx,
                            sigmaX=sigmaX,
                            sigmaY=sigmaY)
            # harris corner
            ey = extremum[idx][0]
            ex = extremum[idx][1]
            ez = extremum[idx][2]
            isCorner = extract_feature.__harrisCorner(locY=ey,
                                              locX=ex,
                                              locZ=ez,
                                              gdy2=gdy2,
                                              gdx2=gdx2,
                                              gdyx=gdyx,
                                              ratio=ratio)
            cornerY = ey[isCorner.astype(bool)]
            cornerX = ex[isCorner.astype(bool)]
            cornerZ = ez[isCorner.astype(bool)]
            features[idx] = (cornerY,
                             cornerX,
                             cornerZ)

        return features

    @staticmethod
    @jit((int64[:])(int64[:], int64[:], int64[:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64))
    def __harrisCorner(locY,
                       locX,
                       locZ,
                       gdy2,
                       gdx2,
                       gdyx,
                       ratio):
        """
        sub function of def edgeRemover
        :param locY:
        :param locX:
        :param locZ:
        :param gdy2:
        :param gdx2:
        :param gdyx:
        :param ratio:
        :return:
        """
        isCorner = np.zeros_like(locY.astype(int64))
        idx = 0
        for y, x, z in zip(locY, locX, locZ):
            A = np.array([[gdy2[y,x,z], gdyx[y,x,z]],
                          [gdyx[y,x,z], gdx2[y,x,z]]])
            trace = A[0,0] + A[1,1]
            det = A[0,0]*A[1,1] - A[1,0]*A[0,1]
            if det < 0 : continue
            elif trace*trace/(det+1e-4) <= ratio : isCorner[idx]=1
            idx+=1
        return isCorner

if __name__ == "__main__":

    imgDir = "D:\\cv\\data\\prac\\cannytest.png"
    img = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE) # Height(Y), Width(X)
    img = cv2.resize(img, (150, 200)) # cv2.resize input shape order ( X(Width), Y(Height))
    # img shape order Y, X (height, width)
    img = img
    deg = 0
    img_rot = rotation(img=img,
                       theta=deg)
    t1 = process_time()

    ## CHAPTER 3 - make scale space, get extremum(sample space)
    ss, sigmas = extract_feature.scaleSpace(img=img_rot,
                                            s=3,
                                            octaveNum=5)
    # DoG space
    DoG = extract_feature.dog(ss)
    # Get extremum
    naiveEx = extract_feature.extractExtremum(dogSpace=DoG)
    # accurate keypoint localization
    localizedEx = extract_feature.localization(dogSpace=DoG,
                                               extremum=naiveEx,
                                               offsetThr=0.5,
                                               contrastThr=0.03)
    features = extract_feature.edgeRemover(dogSpace=DoG,
                                           extremum=localizedEx,
                                           sigmaX=1.5,
                                           sigmaY=1.5,
                                           r=10)
    t2 = process_time()
    print("Process time of Chapter 3 : ", t2 - t1)
    # visualize feature points
    # fig, ax = plt.subplots()
    # ax.imshow(img, cmap='gray')
    # for idx in features.keys():
    #     # if idx!=0 : continue
    #     x_list = []
    #     y_list = []
    #
    #     # deg rotation -> 그냥 로테이션 원상복귀만 시키는 파트 (scale과는 아무 관련 없음)
    #     # X, Y, Z 순서 주의할 것 -> 영상은 기본적으로 (Height, Width, Channel) = (Y, X, Z) 순서임
    #     shift = tran(-img.shape[1]/2, -img.shape[0]/2) # Homogeneous matrix는 x, y 순서로 받아야 한다.
    #     shift_rev = tran(img.shape[1]/2, img.shape[0]/2)
    #     rot_matrix = rot(deg)
    #     h_matrix = np.linalg.multi_dot([shift_rev, rot_matrix, shift])
    #
    #     coord_homo = np.stack([features[idx][1],
    #                            features[idx][0],
    #                            np.ones_like(features[idx][0])], axis=0).astype(np.int64)
    #     source_coord = np.dot(lin.inv(h_matrix), coord_homo) # 결과로 받은 source_coord는 [x, y] 순서임
    #     # 사격형 그리기 -> scale 관련
    #     for x, y, z in zip(source_coord[0,:], source_coord[1,:], features[idx][2]):
    #
    #         # scale(시그마 말고)에 따른, 사각형의 중심 위치
    #         x = x*(2**idx) + (2**idx-1)/2 # center X of Scale
    #         y = y*(2**idx) + (2**idx-1)/2 # center Y of Scale
    #         x_list.append(x)
    #         y_list.append(y)
    #
    #         # # 범위에 따른 사각형
    #         # rec = Rectangle((x-math.floor(6*sigmas[idx][z]/2), y-math.floor(6*sigmas[idx][z]/2)),
    #         #                 math.floor(6*sigmas[idx][z]), math.floor(6*sigmas[idx][z]),
    #         #                 linewidth=1,
    #         #                 edgecolor='r',
    #         #                 facecolor='none')
    #         # ax.add_patch(rec)
    #     ax.scatter(x_list, y_list)
    #
    # plt.show()

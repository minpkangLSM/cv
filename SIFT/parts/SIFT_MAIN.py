import cv2
from featureExtractor import *
from descriptor import *
from matching import *

def dataBase(imgDir,
             imgNorm = 255.,
             kdTree=True) :

    t1 = process_time()

    """STEP 1 : Loading an image"""
    img = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE)  # shape order Y(height), X(width)
    img = cv2.resize(img, dsize=(300, 500))  # cv2.resize input shape order X(width), Y(height)

    # STEP 1-2 : normalize image into 0 ~ 1
    img = img / imgNorm # normalize pixel value in the range [0, 1] - Lowe, 2004, chpater 3.2 (p.96)

    """STEP 2 : Extracting key point from the image"""
    # STEP 2-1 : build scale space from the image
    scaleSpace, sigmas = extract_feature.scaleSpace(img=img,
                                                    s=3,
                                                    octaveNum=5)
    # STEP 2-2 : create Difference of Gaussian (DOG) space from the scale space
    DoG = extract_feature.dog(scaleSpace=scaleSpace)

    # STEP 2-3 : get naive extremums from the DoG space
    naiveExtremum = extract_feature.extractExtremum(dogSpace=DoG)

    # STEP 2-4 : interpolate the extremum and remove what has low constrast
    localizedExtremum = extract_feature.localization(dogSpace=DoG,
                                                     extremum=naiveExtremum,
                                                     offsetThr=0.5,
                                                     contrastThr=0.03) # Thr로 인해 localized 결과가 없어져버릴 수도 있다.
    # print("localizedEx 0 : ", localizedExtremum[0])
    # print("localizedEx 1 : ", localizedExtremum[1])
    # print("localizedEx 2 : ", localizedExtremum[2])
    # print("localizedEx 3 : ", localizedExtremum[3])
    # print("localizedEx 4 : ", localizedExtremum[4])

    # STEP 2-5 : remove features on the edge
    features = extract_feature.edgeRemover(dogSpace=DoG,
                                           extremum=localizedExtremum,
                                           sigmaY=1.5,
                                           sigmaX=1.5,
                                           r=10)

    """STEP 3 : making descriptor"""
    oriFeatures = orientation.assign(dogSpace=DoG,
                                     sigmas=sigmas,
                                     features=features)

    featureVect = orientation.featureVector(oriFeatures=oriFeatures,
                                            dogSpace=DoG)
    t2 = process_time()
    print("t1 ~ t2 PROCESS TIME : ", t2-t1)
    if not kdTree : return featureVect
    else:
        """STEP 4 : matching feature point"""
        # STEP 4-1 : make kd tree for matching
        root = KdTree.makeTree(featureVect)
        return root
    #
    # # visualize feature points
    # img = img
    # deg = 0
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



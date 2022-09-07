import cv2
import numpy as np

def sobel(img):
    f_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    f_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])
    imgDx = cv2.filter2D(img, ddepth=cv2.CV_16S, kernel=f_x)
    imgDy = cv2.filter2D(img, ddepth=cv2.CV_16S, kernel=f_y)

    return imgDy, imgDx

def sobelFloat(img):
    f_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    f_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])
    imgDx = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=f_x)
    imgDy = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=f_y)
    return imgDy, imgDx


def sobelHeightAxis(img,
                    ddepth=cv2.CV_16S):
    """
    해당 필터는 높이 방향으로만 처리되는 필터
    :param img:
    :return:
    """
    fHeight = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    imgDHeight = cv2.filter2D(img, ddepth=ddepth, kernel=fHeight)
    return imgDHeight

def sobelWidthAxis(img,
                   ddepth=cv2.CV_16S):
    """
    해당 필터는 너비 방향으로만 처리되는 필터
    :param img:
    :return:
    """
    fWidth = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    imgWidth = cv2.filter2D(img, ddepth=ddepth, kernel=fWidth)
    return imgWidth

def gaussian(img,
             sigmaX,
             sigmaY):
    xKsize = round(6*sigmaX)
    if xKsize%2 == 0 : xKsize += 1
    yKsize = round(6*sigmaY)
    if yKsize%2 == 0 : yKsize += 1

    imgBlurred = cv2.GaussianBlur(img,
                                  ksize=(yKsize, xKsize),
                                  sigmaX=sigmaX,
                                  sigmaY=sigmaY)
    return imgBlurred

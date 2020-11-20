#!/usr/bin/python

import cv2 as cv
import sys
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import rotate

# load grayscale image
img = cv.cvtColor(cv.imread(str(sys.argv[1])), cv.COLOR_BGR2GRAY)

# resample image by a scale of 2
img = cv.resize(img, (img.shape[1] * 2, img.shape[0] * 2), cv.INTER_CUBIC)
copy = img


# blur the image by convoluting the image with box filters
def _blur(im, n, m):
    kernel = np.ones((n, m))
    kernel /= m
    im[:, :] = convolve2d(im[:, :], kernel, "same")
    return im.astype('uint8')


# crop the image because of the black zone generated by rotation
def _crop(im, r):
    dx = int((r.shape[1] - im.shape[1]) / 2)
    dy = int((r.shape[0] - im.shape[0]) / 2)
    return r[dy:dy + im.shape[0], dx:dx + im.shape[1]].astype('uint8')


# return the localized radon transform
def _rep(im, angle):
    r = rotate(_blur(rotate(im, angle), 1, 3), -angle).astype('uint8')
    if angle != 0 and angle != 90:
        return _crop(im, r)
    else:
        return r


# return derived response map
def _fc(im):
    frep_0 = _rep(im, 0)
    cv.imwrite('src/frep_0.png', frep_0)
    frep_45 = _rep(im, 45)
    cv.imwrite('src/frep_45.png', frep_45)
    frep_90 = _rep(im, 90)
    cv.imwrite('src/frep_90.png', frep_90)
    frep_135 = _rep(im, 135)
    cv.imwrite('src/frep_135.png', frep_135)
    mask = np.zeros(im.shape)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            mask[i, j] = pow(
                max(frep_0[i, j], frep_45[i, j], frep_90[i, j], frep_135[i, j]) - min(frep_0[i, j], frep_45[i, j],
                                                                                      frep_90[i, j],
                                                                                      frep_135[i, j]), 2)
    cv.imwrite('src/frep_c.png', mask.astype('uint8'))
    return mask.astype('uint8')


# return local max img
def _lc_max(im, ksize):
    output = np.zeros(im.shape)
    print("processing.", end="")
    for i in range(ksize // 2, im.shape[0] - ksize // 2):
        if i % 100 == 0: print(".", end="")
        for j in range(ksize // 2, im.shape[1] - ksize // 2):
            output[i, j] = np.max(im[i - ksize // 2:i + ksize // 2, j - ksize // 2:j + ksize // 2])
            #if im[i, j] != max:
            #    output[i, j] = 0  # non maximal suppression
            #else:
            #    output[i, j] = max
    print("finished!")
    cv.imwrite('src/frep_max.png', output.astype('uint8'))
    return output.astype('uint8')


def _tresh(im):
    img_thres = im
    img_thres[im < 200] = 0

    return img_thres.astype('uint8')


def _mask(im):
    original = cv.imread(str(sys.argv[1]))
    original = cv.resize(original, (original.shape[1] * 2, original.shape[0] * 2), cv.INTER_CUBIC)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i, j] != 0:
                original[i, j, :] = 0
                original[i, j, 2] = im[i, j]
    cv.imwrite('src/detected.png',original.astype('uint8'))
    return original.astype('uint8')


img = _fc(img)
# blur resulting response map with a kxk box filter to account for discretization errors
img = _blur(img, 3, 3)
# locate corner by searching local maxima
img = _lc_max(img, 3)
# use tresholding and nonmaximal suppression to filter out weak corners
img = _tresh(img)
# apply the detection on the original image
img = _mask(img)

# display
cv.namedWindow('Paper')
cv.imshow('Paper', img)
while 1:
    k = cv.waitKey(20) & 0xFF
    if k == ord('q'):
        break
cv.destroyAllWindows()

#!/usr/bin/python

import cv2 as cv
import sys
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import rotate

# load grayscale image
img = cv.cvtColor(cv.imread(str(sys.argv[1])), cv.COLOR_BGR2GRAY)

# resample image by a scale of 2
img = cv.resize(img, (img.shape[1], img.shape[0]), cv.INTER_CUBIC)


def _blur(im, n, m):
    kernel = np.ones((n, m))
    kernel /= m
    im[:, :] = convolve2d(im[:, :], kernel, "same")
    return im


def _crop(im, r):
    dx = int((r.shape[1] - im.shape[1]) / 2)
    dy = int((r.shape[0] - im.shape[0]) / 2)
    return r[dy:dy + im.shape[0], dx:dx + im.shape[1]]


def _rep(im, angle):
    if angle != 0 and angle != 90:
        r = rotate(_blur(rotate(im, angle), 1, 3), -angle, reshape=False)
        return _crop(im, r)
    else:
        return rotate(_blur(rotate(im, angle), 1, 3), -angle)


def _fc(im):
    frep_0 = _rep(im, 0)
    frep_45 = _rep(im, 45)
    frep_90 = _rep(im, 90)
    frep_135 = _rep(im, 135)
    print(frep_45.shape)
    print(frep_90.shape)
    mask = np.zeros(im.shape)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            mask[i, j] = pow(
                max(frep_0[i, j], frep_45[i, j], frep_90[i, j], frep_135[i, j]) - min(frep_0[i, j], frep_45[i, j],
                                                                                      frep_90[i, j],
                                                                                      frep_135[i, j]), 2)
    return mask


print(img.shape)
img = _fc(img)
# display
cv.namedWindow('Paper')
cv.imshow('Paper', img)
while 1:
    k = cv.waitKey(20) & 0xFF
    if k == ord('q'):
        break
cv.destroyAllWindows()

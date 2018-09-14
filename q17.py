#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:32:19 2018

@author: hc
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread('./data/left01.jpg',0)
imgR = cv.imread('./data/right01.jpg',0)
#stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
stereo = cv.StereoSGBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
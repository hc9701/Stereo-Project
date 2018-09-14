# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
# my calibrate function written in q6.py
import q6

if __name__=='__main__':
    import sys, getopt,os

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug='])
    args = dict(args)
    args.setdefault('--debug', './output/question6')
    if not img_mask:
        img_mask_l = './data/left??.jpg'
        img_mask_r = './data/right??.jpg'
    else:
        img_mask_l = img_mask[0]
        img_mask_r = img_mask[1]
        
    debug_dir = args.get('--debug')
    if debug_dir and not os.path.isdir(debug_dir):
        os.makedirs(debug_dir)

    print('\nParameter of the left camera:\n')
    rms_l, mtx_l, dist_coefs_l, rvecs_l, tvecs_l = q6.calibrate(img_mask_l, debug_dir)
    print('rms:', rms_l)
    print('distortion coefficients:\n', dist_coefs_l)
    print('camera matrix:\n', mtx_l)
    print('rvecs:\n', rvecs_l)
    print('tvecs:\n', tvecs_l)
    
    print('\nParameter of the left camera:\n')
    rms_r, mtx_r, dist_coefs_r, rvecs_r, tvecs_r = q6.calibrate(img_mask_r, debug_dir)
    print('rms:', rms_r)
    print('distortion coefficients:\n', dist_coefs_r)
    print('camera matrix:\n', mtx_r)
    print('rvecs:\n', rvecs_r)
    print('tvecs:\n', tvecs_r)
    
    print('\nThe transformation between the two camera is:\n')    
    R_l = np.mat(cv.Rodrigues(rvecs_l[0])[0])
    R_r = np.mat(cv.Rodrigues(rvecs_r[0])[0])
    t_l = np.mat(tvecs_l[0])
    t_r = np.mat(tvecs_r[0])
    R = R_r*R_l.T
    T = t_r-R*t_l
    print('rotation matrix:\n',R)
    print('translate vecter:\n',T)
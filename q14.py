#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 12:33:52 2018

@author: hc
"""

import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt

pattern_size = 9, 6  # (h,w)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)



def calibrate(img_mask):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(img_mask)
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)
        else:
            print('chessboard is not found in ', fname, '!')
    ret, mtx, dist_coefs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[:2], None, None)
    mtx  = np.mat(mtx)
    return mtx,rvecs,tvecs,dist_coefs

def undistort(img,camera_matrix,dist_coefs,img_name,output_path='./output/question14'):
    import os
    outfile = os.path.join(output_path, img_name + '_undistorted.jpg')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
    dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
    # crop and save the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
#    cv.imshow(img_name,img)
#    cv.waitKey(500)
    print('Undistorted image written to: %s' % outfile)
    cv.imwrite(outfile, dst)
#    cv.imshow(img_name+'(undistort)',dst)
#    cv.waitKey(500)
    return dst,newcameramtx

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    c = img1.shape[1]
    img1 = img1.copy()
    img2 = img2.copy()
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1[0]),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2[0]),5,color,-1)
    return img1,img2
 
def findPoints(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        return corners2
    else:
        print('chessboard is not found')

def rectify(Po1,Po2):
    from numpy.linalg import svd,norm
    # focal length
    au = norm(np.cross(Po1[0,:3],Po1[2,:3]))
    av = norm(np.cross(Po1[1,:3],Po1[2,:3]))
    
    # optical centres
    c1 = -Po1[:,:3].I*Po1[:,-1]
    c2 = -Po2[:,:3].I*Po2[:,-1]
    
    # retinal planes 
    fl = Po1[-1,:3]
    fr = Po2[-1,:3]

    nn = np.mat(np.cross(fl,fr)).T
    
    # solve the four systems
    A = np.concatenate([c1,c2,nn],axis=1)
    A = np.concatenate([A,np.mat([1,1,0])],axis=0).T
    U,S,VT = svd(A); 
    r = 1/(norm(VT[-1,:3]))
    a3 = r * VT[-1,:].T
    
    A = np.concatenate([c1,c2,a3[:3]],axis=1)
    A = np.concatenate([A,np.mat([1,1,0])],axis=0).T
    U,S,VT = np.linalg.svd(A)
    r = norm(av)/(norm(VT[-1,:3]))
    a2 = r * VT[-1,:].T
    
    A = np.concatenate([c1,a2[:3],a3[:3]],axis=1)
    A = np.concatenate([A,np.mat([1,0,0])],axis=0).T
    U,S,VT = svd(A)
    r = norm(au)/(norm(VT[-1,:3]))
    a1 = r * VT[-1,:].T
    
    A = np.concatenate([c2,a2[:3],a3[:3]],axis=1)
    A = np.concatenate([A,np.mat([1,0,0])],axis=0).T
    U,S,VT = svd(A)
    r = norm(au)/(norm(VT[-1,:3]))
    b1 = r * VT[-1,:].T

    # adjustment
    H = np.mat(np.eye(3))

    # rectifying  projection matrices
    Pn1 = H * np.concatenate([a1,a2,a3],axis=1).T
    Pn2 = H * np.concatenate([b1,a2,a3],axis=1).T

    # rectifying image transformation
    T1 = Pn1[:3,:3]* Po1[:3,:3].I
    T2 = Pn2[:3,:3]* Po2[:3,:3].I
    return T1,T2,Pn1,Pn2

def cal_new_points(pts1,pts2,T1,T2):
    pts_n_1 = pts1.copy()
    pts_n_2 = pts2.copy()
    for i in range(len(pts1)):
        pts_n_1[i] = (T1[:2,:2]*np.mat(pts1[i]).T+T1[:2,-1]).T
    for i in range(len(pts2)):
        pts_n_2[i] = (T2[:2,:2]*np.mat(pts2[i]).T+T2[:2,-1]).T
        
    return pts_n_1,pts_n_2
    
if __name__=='__main__':
    img_l = cv.imread('./data/left01.jpg')
    img_r = cv.imread('./data/right01.jpg')
    mtx_l,rvecs_l,Ts_l,dist_coefs_l = calibrate('./data/left*.jpg')
    mtx_r,rvecs_r,Ts_r,dist_coefs_r = calibrate('./data/right*.jpg')
    img_l,mtx_l = undistort(img_l,mtx_l,dist_coefs_l,'left01')
    img_r,mtx_r = undistort(img_r,mtx_r,dist_coefs_r,'right01')

    cv.waitKey(5000)
    cv.destroyAllWindows()
    
    pts_l = findPoints(img_l)
    pts_r = findPoints(img_r)
    F, mask = cv.findFundamentalMat(pts_l,pts_r,cv.FM_LMEDS)
    
    # We select only inlier points
    pts_l = pts_l[mask.ravel()==1]
    pts_r = pts_r[mask.ravel()==1]
    
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts_r.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img_l,img_r,lines1,pts_l,pts_r)
    cv.imwrite('./output/question14/left01_epilines.jpg', img5)
    cv.imwrite('./output/question14/right01_points.jpg', img6)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img6)
    plt.pause(2)
    plt.show()
    plt.figure()
    
    # the intrinsic parameters of each camera in the first photo
    R_l = np.mat(cv.Rodrigues(rvecs_l[0])[0])
    T_l = Ts_l[0]
    R_r = np.mat(cv.Rodrigues(rvecs_r[0])[0])
    T_r = Ts_r[0]

    # the perspective projection matrix
    Po1 = mtx_l*np.concatenate([R_l,T_l],axis=1)
    Po2 = mtx_r*np.concatenate([R_r,T_r],axis=1)
    
    T1,T2,Pn1,Pn2 = rectify(Po1,Po2)
    
    pts_n_l,pts_n_r = cal_new_points(pts_l,pts_r,T1,T2)
    
    F, mask = cv.findFundamentalMat(pts_n_l,pts_n_r,cv.FM_LMEDS)
    
    # We select only inlier points
    pts_n_l = pts_n_l[mask.ravel()==1]
    pts_n_r = pts_n_r[mask.ravel()==1]
    
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines2 = cv.computeCorrespondEpilines(pts_n_r.reshape(-1,1,2), 2,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img_l,img_r,lines2,pts_n_l,pts_n_r)
    cv.imwrite('./output/question14/left01_rectify.jpg', img3)
    cv.imwrite('./output/question14/right01_rectify.jpg', img4)
    
    
    plt.subplot(121),plt.imshow(img3)
    plt.subplot(122),plt.imshow(img4)
    plt.show()
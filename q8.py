#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:30:06 2018

@author: hc
"""

import cv2 as cv
import numpy as np
import glob


np.set_printoptions(suppress=True)
size = 9, 6  # (h,w)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def cal_rt(homography, intrinsic_mat):
    n = homography.shape[0]
    r = np.empty((n, 3, 3))
    t = np.empty((n, 1, 3))

    for i in range(n):
        lam = np.linalg.norm(intrinsic_mat.I * homography[i, :, :], axis=0)[0]
        r[i, :, :2] = lam * homography[i, :, :2]
        r[i, :, -1] = r[i, 1, 0] * r[i, 2, 1] - r[i, 2, 0] * r[i, 1, 1], \
                      r[i, 2, 0] * r[i, 0, 1] - r[i, 0, 0] * r[i, 2, 1], \
                      r[i, 0, 0] * r[i, 1, 1] - r[i, 1, 0] * r[i, 0, 1]
        t[i, :, :] = lam * homography[i, :, -1].T
    return r, t


def cal_intrinsics(b):
    v0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] ** 2)
    lam = b[5] - (b[3] ** 2 + v0 * (b[1] * b[3] - b[0] * b[4])) / b[0]
    alpha = (lam / b[0]) ** 0.5
    beta = (lam * b[0] / (b[0] * b[2] - b[1] ** 2)) ** 0.5
    gamma = -b[1] * alpha ** 2 * beta / lam
    u0 = gamma * v0 / beta - b[3] * alpha ** 2 / lam
    return np.mat([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1],
    ]), lam


def cal_v(h):
    def _cal_v(h, i, j):
        i, j = i - 1, j - 1
        return np.array([
            h[i, 0] * h[j, 0],
            h[i, 0] * h[j, 1] + h[i, 1] * h[j, 0],
            h[i, 1] * h[j, 1],
            h[i, 2] * h[j, 0] + h[i, 0] * h[j, 2],
            h[i, 2] * h[j, 1] + h[i, 1] * h[j, 2],
            h[i, 2] * h[j, 2],
        ])

    return np.array([_cal_v(h, 1, 2), _cal_v(h, 1, 1) - _cal_v(h, 2, 2)])


def cal_homography(intrinsic_mat, h):
    homography = np.array([intrinsic_mat.I * h1 / np.linalg.norm(intrinsic_mat.I * h1, axis=0)[0] for h1 in h])
    return homography


def func1(x, A):
    if A.shape==(1,4):
        A_new = np.mat([
                [A[0,0],0,A[0,1]],
                [0,A[0,2],A[0,3]],
                [0,0,1]
        ])
    else:
        A_new = A
    M, lam, extrinsics = x[0], x[1], x[2]
    H = [1 / lam * A_new * extrinsic_mat for extrinsic_mat in extrinsics]
    ans = np.array([M[i, :] * H[i].T[:, :2] / (M[i, :] * H[i].T[:, -1]) for i in range(len(H))])
    return ans


def preprocess(images, pattern_size=size):
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    #        zeros = np.zeros((1,3))

    # 4 3d points in real world space in each images
    M = np.mat(objp)
    M[:, -1] = 1
    #        choices=[0,1-pattern_size[0],pattern_size[0]-1,-1]
    #        M1=M[choices1].T
    #        n = len(choices1)
    n = np.prod(pattern_size)

    for fname in images:
        im = cv.imread(fname)
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
        # If found, add object points, image points (after refining them)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            ones = np.ones((n, 1))
            imgp = np.array(np.concatenate((corners2[:, 0], ones), 1))
            imgpoints.append(imgp)

    return np.array(objpoints), np.array(imgpoints)


def cal_closed_form_solution(objpoints, imgpoints,
                             choices=(0, 1 - size[0], size[0] - 1, -1)):
    #     objpoints in each images of coordinate
    #        0 0 1
    #        1 5 1
    #        8 0 1
    #        8 5 1

    choices = np.array(choices)

    N = objpoints.shape[0]
    # N is the number of points used in each images
    # the default value is 13
    n = len(choices)
    # n is the number of images used
    # the default value is 4

    M = objpoints[:, choices, :]
    m = imgpoints[:, choices, :]

    v = []
    A = np.empty((2 * n, 9))
    h = []
    for i in range(N):
        for j in range(n):
            A[2 * j, :3] = M[i, j, :]
            A[2 * j, -3:] = -m[i, j, 0] * M[i, j, :]
            A[2 * j + 1, 3:6] = M[i, j, :]
            A[2 * j + 1, -3:] = -m[i, j, 1] * M[i, j, :]

        u, s, vt = np.linalg.svd(A)
        V1 = vt.T

        h1 = V1[:, -1]
        h1 = h1.reshape(3, 3)
        h.append(h1)
        v.extend(cal_v(h1.T))
    
    mat = np.mat(v)
    u, s, vt = np.linalg.svd(mat)
    V = vt.T
    b = V[:, -1].getA().T[0]
    intrinsic_mat, lam = cal_intrinsics(b)

    homography = cal_homography(intrinsic_mat, h)
    return intrinsic_mat, lam, homography


def LM(x, y, param, func, epsilon1=1e-3, epsilon2=1e-3, kmax=1000, tau=1e5):
    def f(param):
        return (np.linalg.norm(y - func(x, param), axis=-1) ** 2).reshape(-1, 1)

    def F(param):
        return 0.5 * np.linalg.norm(f(param)) ** 2

    def deriv(func, param, step=1e-5):
        derivative = np.mat(np.zeros((np.prod(x[0].shape[:-1]), param.shape[1])))
        for i in range(param.shape[1]):
            param_new = param.copy()
            param_new[:, i] = param_new[:, i] - step
            derivative[:, i] = (func(param) - func(param_new)) / step
        return derivative

    k, v = 0, 2
    J = np.mat(deriv(f, param))
    g = J.T * f(param)
    A = J.T * J

    found = np.linalg.norm(g, ord=np.inf) <= epsilon1
    u = tau * np.max(A)
    
    while not found and k < kmax:
        k += 1
        # solve (A+u*I)hlm=-g
        h_lm = -(A + u*np.eye(A.shape[0])).I * g
        if np.linalg.norm(h_lm) < epsilon2 * (epsilon2 + np.linalg.norm(h_lm)):
            found = True
        else:
            param_new = param + h_lm.T
            p = -(F(param) - F(param_new)) \
                / ((h_lm.T * J.T) * (f(param) + 0.5 * J * h_lm))
            p = p[0,0]
            if p > 0:
                print('iteration', k, 'mse-differ', abs(F(param) - F(param_new)),'mse',F(param_new))
                param = param_new
                J = np.mat(deriv(f, param))
                A = J.T * J
                g = J.T * f(param)
                found = np.linalg.norm(g, ord=np.inf) <= epsilon1
                u = u * max(1 / 3, 1 - (2 * p - 1) ** 3)
                v = 2
            else:
                u = u * v
                v = 2 * v
                
    return param

def cal_distortion_coefficients(intrinsic_mat,imgpoints,objpoints,lam,homography,dist_coefs,func):
    u0,v0 = intrinsic_mat[:2,-1]
    x = M, lam, homography_mat,dist_coefs
    m_hat = func(x,intrinsic_mat)
    r = np.linalg.norm(M[:,:,:2],axis = -1)
    
    d = imgpoints[:,:,:2]-m_hat
    d=d.reshape(-1,1)
    n,m = M.shape[:2]
    D = np.mat(np.empty((2*m*n,2)))
    for i in range(n):
        for j in range(m):
            index = 2*(i*m+j)
            diff = m_hat[i,j,:2]-intrinsic_mat[:2,-1].T
            D[index:index+2,:]= \
            np.concatenate([diff.T*r[i,j]**2,diff.T*r[i,j]**4],1)
    return ((D.T*D).I*D.T*d).T

def func2(x,A):
    if A.shape==(1,4):
        A_new = np.mat([
                [A[0,0],0,A[0,1]],
                [0,A[0,2],A[0,3]],
                [0,0,1]
        ])
    else:
        A_new = A
    M, lam, extrinsics,k = x[0], x[1], x[2],x[3]
    H = [1 / lam * A_new * extrinsic_mat for extrinsic_mat in extrinsics]
    ans = np.array([M[i, :] * H[i].T[:, :2] / (M[i, :] * H[i].T[:, -1]) for i in range(len(H))])
    r = np.linalg.norm(M[:,:,:2],axis=-1)
    diff = ans-[A_new[0,-1],A_new[1,-1]]
    m,n=M.shape[:2]
    
    R = np.array([r**2,r**4])
    for i in range(m):
        for j in range(n):
            diff1 = np.mat(diff[i,j,:])
            R1 = np.mat(R[:,i,j])
            ans[i,j,:] = ans[i,j,:]+k*R1.T*diff1
    return ans
  

if __name__ == '__main__':
    import os
    images = glob.glob('./data/left*.jpg')
    objpoints, imgpoints = preprocess(images)
    intrinsic_mat, lam, homography_mat = cal_closed_form_solution(objpoints, imgpoints)
    r, t = cal_rt(homography_mat,intrinsic_mat)
    M, m = objpoints, imgpoints

    intrinsics = intrinsic_mat.reshape(-1)[0,[0,2,4,5]]
#    intrinsics = intrinsic_mat.reshape(-1)
    x = M, lam, homography_mat
    
    A1= LM(x, m[:, :, :2], intrinsics, func1)

    A1_new = np.mat([
            [A1[0,0],0,A1[0,1]],
            [0,A1[0,2],A1[0,3]],
            [0,0,1]
    ])
    img1 = func1(x,intrinsics)
    print(intrinsic_mat)
    print(A1_new)

    dist_coefs = np.mat([[0,0]])
    dist_coefs = cal_distortion_coefficients(A1_new,imgpoints,objpoints,lam,homography_mat,dist_coefs,func2)

    x = M, lam, homography_mat,dist_coefs
    intrinsics = A1_new.reshape(-1)[0,[0,2,4,5]]
    img2 = func2(x,intrinsics)
    A2= LM(x, m[:, :, :2], intrinsics, func2)
    A2_new = np.mat([
            [A2[0,0],0,A2[0,1]],
            [0,A2[0,2],A2[0,3]],
            [0,0,1]
    ])
    print(A2_new)
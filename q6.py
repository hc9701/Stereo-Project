import numpy as np
import cv2 as cv
import glob


'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration

usage:
    python q6.py [--debug <output path>]

default values:
    --debug:    ./output/question6
'''

pattern_size = 9, 6  # (h,w)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)


# Arrays to store object points and image points from all the images.



def calibrate(img_path, output_path):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    import os
    images = glob.glob(img_path)
    for fname in images:
        img = cv.imread(fname)
        img_name_ext = fname.split('/')[-1]
        img_name = img_name_ext[:img_name_ext.find('.')]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)
            img_chess_name = os.path.join(img_path, img_name + '_chess.jpg')
            cv.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv.imwrite(img_chess_name, img)
        else:
            print('chessboard is not found in ', fname, '!')
    ret, mtx, dist_coefs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[:2], None, None)
    for rvec in rvecs:
        # Change rvecs to rotation matrix
        print('rotation matrix:\n', cv.Rodrigues(rvec)[0])
    return ret, mtx, dist_coefs, rvecs, tvecs


if __name__ == '__main__':
    import sys, getopt,os

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug='])
    args = dict(args)
    args.setdefault('--debug', './output/question6')
    if not img_mask:
        img_mask = './data/left??.jpg'
    else:
        img_mask = img_mask[0]

    debug_dir = args.get('--debug')
    if debug_dir and not os.path.isdir(debug_dir):
        os.makedirs(debug_dir)
    rms, mtx, dist_coefs, rot_mat, tvecs = calibrate(img_mask, debug_dir)
    print('rms:', rms)
    print('distortion coefficients:\n', dist_coefs)
    print('camera matrix:\n', mtx)
    print('rotation matrix:\n', rot_mat)
    print('tvecs:\n', tvecs)

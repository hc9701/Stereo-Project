import cv2 as cv
import numpy as np
import glob

'''
write undistorted images using camera matrix in question 6

usage:
    python calibrate.py [--debug <output path>]

default values:
    --data:     ./data7
'''


def load_data(data_path):
    try:
        with open(data_path) as f:
            camera_matrix = []
            dist_coefs = []
            cnt = 0
            for line in f.readlines():
                cnt += 1
                arrays = line.strip().split('\t')
                if cnt >= 4:
                    dist_coefs.append(arrays)
                else:
                    camera_matrix.append(arrays)
            camera_matrix = np.mat(camera_matrix)
    except:
        camera_matrix = [
            [532.82709963, 0, 342.48678133],
            [0., 532.94587936, 233.85595303],
            [0., 0, 1.],
        ]
        dist_coefs = [
            [-2.80881018e-01, 2.51724593e-02, 1.21657369e-03, - 1.35550674e-04,
             1.63447359e-01]
        ]

    # no data available, just use the default value calculated in question 6
    if not camera_matrix or np.mat(camera_matrix).shape != (3, 3):
        camera_matrix =[
                [532.82709963, 0, 342.48678133],
                [0., 532.94587936, 233.85595303],
                [0., 0, 1.],
            ]

    if not dist_coefs:
        dist_coefs = [
            [-2.80881018e-01, 2.51724593e-02, 1.21657369e-03, - 1.35550674e-04,
             1.63447359e-01]
        ]

    return np.mat(camera_matrix), np.mat(dist_coefs)


def undistort(img_path, output_path, camera_matrix,dist_coefs):
    images = glob.glob(img_path)
    for fname in images:
        img = cv.imread(fname)
        img_name_ext = fname.split('/')[-1]
        img_name = img_name_ext[:img_name_ext.find('_')]

        outfile = os.path.join(output_path, img_name + '_undistorted.jpg')

        img = cv.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
        print(newcameramtx)
        dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        print('Undistorted image written to: %s' % outfile)
        cv.imwrite(outfile, dst)



if __name__ == '__main__':
    import sys, os, getopt

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug='])
    args = dict(args)
    args.setdefault('--debug', './output/question7')

    if not img_mask:
        img_mask = './output/question6/left??_chess.jpg'
    else:
        img_mask = img_mask[0]

    debug_dir = args.get('--debug')
    if debug_dir and not os.path.isdir(debug_dir):
        os.makedirs(debug_dir)
    camera_matrix,dist_coefs = load_data(img_mask)
    print(camera_matrix,dist_coefs)
    undistort(img_mask,debug_dir,camera_matrix,dist_coefs)


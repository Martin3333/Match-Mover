# -*- coding: utf-8 -*-

import numpy as np
import pickle
import cv2
import glob
import os
from sources.object3d import Object3D as obj3D
#import sources.object3d
from matplotlib import pyplot as plt


def translate3DPoints(points,x,y,z):
    newPoints = []
    for p in points:
        newPoints.append((p[0]+x, p[1]+y, p[2]+z))
    return newPoints

def scale3DPoints(points, scaleX, scaleY, scaleZ):
    newPoints = []
    for p in points:
        newPoints.append((p[0]*scaleY, p[1]*scaleX, p[2]*scaleZ))
    return newPoints


def calibrate(maxIndex, path='videoframes/'):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # checkerboard Dimensions
    cbrow = 5
    cbcol = 7

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.



    for i in range(0,maxIndex):
        # Read Image
        img = cv2.imread(path + str(i) + '.jpg' , 0)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img, (cbrow,cbcol),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)

    img = cv2.imread(path + '0.jpg', 0)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1],None,None)
    print(ret)
    return mtx



def render(index1, index2, K_mtx, path='videoframes/', outPath='out/', drawMatches=False):
    # Load images
    img1 = cv2.imread(path + str(index1)+'.jpg' ,1)
    img2 = cv2.imread(path + str(index2)+'.jpg' ,1)

    # Detect Keypoints
    detector = cv2.xfeatures2d.SURF_create()
    kp1,des1 = detector.detectAndCompute(img1, None)
    kp2,des2 = detector.detectAndCompute(img2, None)

    # Initialize matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    # Match
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)


    #Filter good matches
    pts1 = []
    pts2 = []

    if drawMatches: matchesMask = [[0,0] for i in range(len(matches))]

    # # ratio test as per Lowe's paper
    # for i,(m,n) in enumerate(matches):
    #     #if m.distance < 0.7*n.distance:
    #         if drawMatches: matchesMask[i]=[1,0]
    #         pts2.append(kp2[m.trainIdx].pt)
    #         pts1.append(kp1[m.queryIdx].pt)

    pixDiff = 100

    for k1 in kp1:
        for k2 in kp2:
            if abs(k1.pt[0]-k2.pt[0])<pixDiff and abs(k1.pt[1]-k2.pt[1])<pixDiff:
                pts1.append(k1.pt)
                pts2.append(k2.pt)

    if drawMatches:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        plt.imshow(img3,),plt.show()



    ##  manual point correspondences images 0,8
    pts1 = [(477.0, 523.0),
            (1132.0, 447.0),
            (615.0, 119.0),
            (1705.0, 365.0),
            (87.0, 436.0),
            (458.0, 286.0),
            (1217.0, 120.0),
            (1047.0, 146.0),
            (823.0, 184.0)]


    pts2 = [(469.0, 344.0),
            (1062.0, 461.0),
            (730.0, 26.0),
            (1580.0, 541.0),
            (158.0, 170.0),
            (485.0, 130.0),
            (1564.0, 219.0),
            (1135.0, 156.0),
            (941.0, 140.0)]
            


    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # Compute F
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)
    # pts1 = pts1[mask.ravel()==1]
    # pts2 = pts2[mask.ravel()==1]
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # #print(pts1.shape())
    # pts1, pts2 = cv2.correctMatches(F, np.reshape(pts1, (1,len(pts1), 2)), np.reshape(pts2, (1,len(pts2), 2)))

    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # Compute E
    E = np.dot(np.dot(np.transpose(K_mtx), F), K_mtx)
    # Get R and T
    _, R,T, mask= cv2.recoverPose(E, pts1, pts2, K_mtx)

    # R1, R2, T = cv2.decomposeEssentialMat(E)

    # R=R1
    # print(T)
    # imgpoints = []
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # ret, corners1 = cv2.findChessboardCorners(img1, (5,7),None)

    #     # If found, add object points, image points (after refining them)
    #     if ret == True:
    #         cv2.cornerSubPix(img1,corners1,(11,11),(-1,-1),criteria)
    #         imgpoints.append(corners1)

    # ret, corners2 = cv2.findChessboardCorners(img2, (5,7),None)

    #     # If found, add object points, image points (after refining them)
    #     if ret == True:
    #         cv2.cornerSubPix(img2,corners2,(11,11),(-1,-1),criteria)
    #         imgpoints.append(corners2)

    # cv2.triangulatePoints()


    #XYZ_Axis = [(10000.0, 0.0, 0.0), (0.0, 10000.0, 0.0), (0.0, 0.0, -1000.0), (0.0, 0.0, 0.0)]

    T[2] -= 2000.0
    #project into images, first one with 0 vectors for R and T
    img1 = obj3D.render(img1, (0,0,0), (0.0, 0.0, -2000.0), K_mtx)
    img2 = obj3D.render(img2, R, T, K_mtx)


    # axis1,_ = cv2.projectPoints(np.array(XYZ_Axis), (0,0,0), (0,0,-2000.0), K_mtx, (0,0,0,0))
    # axis2,_ = cv2.projectPoints(np.array(XYZ_Axis), R, T, K_mtx, (0,0,0,0))



    # axis1 = [(np.int32(p[0][0]), np.int32(p[0][1])) for p in axis1]
    # axis2 = [(np.int32(p[0][0]), np.int32(p[0][1])) for p in axis2]



    # cv2.line(img1, axis1[3], axis1[0], (255,0,0), 10)
    # cv2.line(img1, axis1[3], axis1[1], (0,255,0), 10)
    # cv2.line(img1, axis1[3], axis1[2], (0,0,255), 10)
    # cv2.line(img2, axis2[3], axis2[0], (255,0,0), 10)
    # cv2.line(img2, axis2[3], axis2[1], (0,255,0), 10)
    # cv2.line(img2, axis2[3], axis2[2], (0,0,255), 10)


    #write images
    cv2.imwrite(outPath + '1.jpg', img1)
    cv2.imwrite(outPath + '2.jpg', img2)

    cv2.imshow('left', img1)
    cv2.imshow('right', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#Params
maxIndex = 32
index1 = 0
index2 = 8


if not os.path.isfile('K.pickle'):
    K_mtx = calibrate(maxIndex)
    with open('K.pickle', 'wb') as handle:
        pickle.dump(K_mtx, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('K.pickle', 'rb') as handle:
    K_mtx = pickle.load(handle)

render(index1, index2, K_mtx, drawMatches=False)







## Just left that block in case we need it for KP-Match checking
    
    # # COMMENT OUT TO PLOT POINT CORRESPONDENCES
    # matchesMask = [[0,0] for i in range(len(matches))]
    # # ratio test as per Lowe's paper
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < 0.5*n.distance:
    #         matchesMask[i]=[1,0]
    # draw_params = dict(matchColor = (0,255,0),
    #                    singlePointColor = (255,0,0),
    #                    matchesMask = matchesMask,
    #                    flags = 0)
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    # plt.imshow(img3,),plt.show()







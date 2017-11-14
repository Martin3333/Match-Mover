# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt


def translate3DPoints(points,x,y,z):
    newPoints = []
    for p in points:
        newPoints.append((p[0]+x, p[1]+y, p[2]+z))
    return newPoints

def scale3DPoints(points, scale):
    newPoints = []
    for p in points:
        newPoints.append((p[0]*scale, p[1]*scale, p[2]*scale))
    return newPoints


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




img = cv2.imread('chessboard.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (cbrow,cbcol),None)

# If found, add object points, image points (after refining them)

if ret == True:
    objpoints.append(objp)

    cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


# #Not sure if we need this...
# img = cv2.imread('scene1.jpg')
# h,  w = img.shape[:2]

# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))




img1 = cv2.imread('scene1.jpg',0)  #queryimage # left image
img2 = cv2.imread('scene2.jpg',0) #trainimage # right image


detector = cv2.xfeatures2d.SIFT_create()

kp1,des1 = detector.detectAndCompute(img1, None)
kp2,des2 = detector.detectAndCompute(img2, None)




FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)



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

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]


#TODO: compare different computations for E
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)
E = np.dot(np.dot(np.transpose(mtx), F), mtx)
#E, mask = cv2.findEssentialMat(pts1, pts2, mtx, cv2.RANSAC, 0.999, 1.0);
_, R,T, mask= cv2.recoverPose(E, pts1, pts2, mtx)





# print(mtx)
# print(E)
# print(R,T)

#Simple Cube in 3D
points3d = []
points3d.append((1.0, 1.0, 1.0))
points3d.append((1.0, -1.0, 1.0))
points3d.append((-1.0, 1.0, 1.0))
points3d.append((-1.0, -1.0, 1.0))
points3d.append((1.0, 1.0, -1.0))
points3d.append((-1.0,1.0, -1.0))
points3d.append((1.0, -1.0, -1.0))
points3d.append((-1.0 , -1.0, -1.0))

#move it around
points3d = scale3DPoints(points3d, 3.0)
points3d = translate3DPoints(points3d, 2.0, 2.0, 20.0)

#project into images, first one with 0 vectors for R and T
points1,_ = cv2.projectPoints(np.array(points3d), (0,0,0), (0,0,0), mtx, (0,0,0,0))
points2,_ = cv2.projectPoints(np.array(points3d), R, T, mtx, (0,0,0,0))


#connect all points with lines
color = (255,255,255)
points1 = [(np.int32(p[0][0]), np.int32(p[0][1])) for p in points1]
points2 = [(np.int32(p[0][0]), np.int32(p[0][1])) for p in points2]

for p1 in points1:
    
    for p2 in points1:
        cv2.line(img1,p1, p2, (255,255,255), 5)

for p1 in points2:
    for p2 in points2:
        cv2.line(img2,p1, p2, (255,255,255), 5)

#write images
cv2.imwrite('1.jpg', img1)
cv2.imwrite('2.jpg', img2)



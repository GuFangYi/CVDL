import numpy as np
import cv2
import glob
import os
	#cmd: pyuic5 -x test.ui -o test.py
	#test.py should include
	#from function import *
   	#trigger event: self.pushButton.clicked.connect(findCorner)

 # Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
framesize = (500,400)
#cameraMatrix = []
#dist = []
def findCorner():

    nx = 11
    ny = 8

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Make a list of calibration images
    images = glob.glob('image/Q1_Image/*.bmp')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        img = cv2.resize(img, framesize, fx=0.75, fy=0.75)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners2, ret)
            cv2.imshow('input image- '+fname[15:], img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

def camera_calibrate():

    nx = 11
    ny = 8

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)


    # Make a list of calibration images
    images = glob.glob('image/Q1_Image/*.bmp')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)


def findIntrinsic():
	#global cameraMatrix
	#global dist
	if not objpoints:
		camera_calibrate()
	#if len(cameraMatrix) == 0:
	#	ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, framesize, None, None)
	ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, framesize, None, None)
	print("Intrinsic:\n",cameraMatrix)

def findExtrinsic(QLineEdit):
	#global cameraMatrix
	#global dist

	nx = 11
	ny = 8
	run = True
	ImageNumber_str = QLineEdit.text()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	if not ImageNumber_str:
		print("Enter image number")
		run = False
	else:
		if int(ImageNumber_str)>15 or int(ImageNumber_str)<1:
			print("Invalid input")
			run = False
		else:
			#print(ImageNumber_str)
			run = True

		
	if run:
		if not objpoints:
			camera_calibrate()
		#if len(cameraMatrix) == 0:
		#	ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, framesize, None, None)
		ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, framesize, None, None)

		objp = np.zeros((nx*ny,3), np.float32)
		objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
		axis = np.float32([[3,0,0], [0,3,0],[0,0,-3]]).reshape(-1,3)
		axisBoxes = np.float32([[0,0,0],[0,3,0],[3,3,0],[3,0,0],
			[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])

		img = cv2.imread('image/Q1_Image/'+ImageNumber_str+'.bmp')
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

		# If found, add object points, image points
		if ret == True:
			corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
			rev, rvecs, tvecs = cv2.solvePnP(objp, corners2, cameraMatrix, dist)
			#print("rotational: ", rvecs)
			#print("translational: ", tvecs)
			rmat,_ = cv2.Rodrigues(rvecs)
			#print("rmat: ", rmat)
			extrinsic = cv2.hconcat([rmat, tvecs])
			print("Extrinsic:\n", extrinsic)



def findDistortion():
	#global cameraMatrix
	#global dist

	if not objpoints:
		camera_calibrate()
	#if len(cameraMatrix) == 0:
	#	ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, framesize, None, None)
	ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, framesize, None, None)	
	print("Distortion:\n",dist)

def findUndistortion():
	#global cameraMatrix
	#global dist

	images = glob.glob('image/Q1_Image/*.bmp')

	if not objpoints:
		camera_calibrate()
	#if len(cameraMatrix) == 0:
	#	ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, framesize, None, None)
	ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, framesize, None, None)
		
	for fname in images:
		img = cv2.imread(fname)
		h, w = img.shape[:2]

		newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix,dist,(w,h),1,(w,h))
		#Undistort
		dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
		#crop the image
		x, y, w, h = roi
		dst = dst[y:y+h, x:x+w]

		#Undistort with Remapping
		mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
		dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

		#crop the image
		x, y, w, h = roi
		dst = dst[y:y+h, x:x+w]
		#path = 'image/Q1_Image/Undistort'
		#if not os.path.isdir(path):
		#	os.mkdir(path)
		#cv2.imwrite(os.path.join(path,fname[15:].replace("bmp","png")), dst)

		img = cv2.resize(img, (500,400))
		dst = cv2.resize(dst, (500,400))

		Hori = np.concatenate((img,dst),axis = 1)
		cv2.imshow('Distorted/Undistorted- '+fname[15:], Hori)
		cv2.waitKey(500)

	cv2.destroyAllWindows()

	# Reprojection Error
	#mean_error = 0

	#for i in range(len(objpoints)):
	#	imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
	#	error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
	#	mean_error += error
	#print("\n total error:{}".format(mean_error/len(objpoints)))



import numpy as np
import cv2
import glob
import os

objpoints = []
imgpoints = []
framesize = (500,400)
cameraMatrix = []
dist =[]

def camera_calibrate():

    nx = 11
    ny = 8

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)


    # Make a list of calibration images
    images = glob.glob('image/Q2_Image/*.bmp')

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

def drawWords(img, corners, imgpts_a, imgpts_b, imgpts_c, imgpts_d, imgpts_e, imgpts_f):
	

	imgpts_a = np.int32(imgpts_a).reshape(-1,2)
	imgpts_b = np.int32(imgpts_b).reshape(-1,2)
	imgpts_c = np.int32(imgpts_c).reshape(-1,2)
	imgpts_d = np.int32(imgpts_d).reshape(-1,2)
	imgpts_e = np.int32(imgpts_e).reshape(-1,2)
	imgpts_f = np.int32(imgpts_f).reshape(-1,2)

	#draw ground floor in green 
	#img = cv2.drawContours(img, [imgpts[:4]], -1, (0,255,0), -3)

	
	#draw pillars in green color
	x=255
	for i,j in zip(range(0,len(imgpts_a),2),range(1, len(imgpts_a),2)):
		img = cv2.line(img, tuple(imgpts_a[i]), tuple(imgpts_a[j]),(0,255,0),10)

	for i,j in zip(range(0,len(imgpts_b),2),range(1, len(imgpts_b),2)):
		img = cv2.line(img, tuple(imgpts_b[i]), tuple(imgpts_b[j]),(0,255,0),10)
	for i,j in zip(range(0,len(imgpts_c),2),range(1, len(imgpts_c),2)):
		img = cv2.line(img, tuple(imgpts_c[i]), tuple(imgpts_c[j]),(0,255,0),10)
	for i,j in zip(range(0,len(imgpts_d),2),range(1, len(imgpts_d),2)):
		img = cv2.line(img, tuple(imgpts_d[i]), tuple(imgpts_d[j]),(0,255,0),10)
	for i,j in zip(range(0,len(imgpts_e),2),range(1, len(imgpts_e),2)):
		img = cv2.line(img, tuple(imgpts_e[i]), tuple(imgpts_e[j]),(0,255,0),10)
	for i,j in zip(range(0,len(imgpts_f),2),range(1, len(imgpts_f),2)):
		img = cv2.line(img, tuple(imgpts_f[i]), tuple(imgpts_f[j]),(0,255,0),10)

	#draw top layer in red color
	#img = cv2.drawContours(img, [imgpts[:4]], -1, (0,255,0), 3)

	return img

def transform_vertical(imgpts):
	#print('imgpts:', imgpts)
	for i, cell in zip(range(imgpts.shape[0]),imgpts):
		#print('cell', cell)
		#print('i', i)
		'''
		if cell[1]==1: 
			imgpts[i] = imgpts[i] + (0,0,1)
		else:
			if cell[1]==0: 
				imgpts[i] = imgpts[i] + (0,0,2)
		'''
		#print(cell[0], cell[1], cell[2])
		imgpts[i] = (cell[0],0,-cell[1])
		
	#print('transformed', imgpts)
	return imgpts

def augment_word(QLineEdit, onboard):

	global cameraMatrix
	global dist
	nx = 11
	ny = 8
	Word_str = QLineEdit.text()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	if not Word_str or len(Word_str)!=6:
		print("Enter 6 letters")
	else:
		if not objpoints:
			camera_calibrate()
		if len(cameraMatrix) == 0:
			ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, framesize, None, None)

		# get position of each characters
		fs = cv2.FileStorage('image/Q2_Image/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
		a, b, c, d, e, f = Word_str
		ch_a = fs.getNode(a).mat()
		ch_b = fs.getNode(b).mat()	
		ch_c = fs.getNode(c).mat()
		ch_d = fs.getNode(d).mat()	
		ch_e = fs.getNode(e).mat()	
		ch_f = fs.getNode(f).mat()	
		ch_a = np.float32(ch_a).reshape(-1,3)
		ch_b = np.float32(ch_b).reshape(-1,3)
		ch_c = np.float32(ch_c).reshape(-1,3)
		ch_d = np.float32(ch_d).reshape(-1,3)
		ch_e = np.float32(ch_e).reshape(-1,3)
		ch_f = np.float32(ch_f).reshape(-1,3)

		if onboard == False:
			ch_a = transform_vertical(ch_a)
			ch_b = transform_vertical(ch_b)
			ch_c = transform_vertical(ch_c)
			ch_d = transform_vertical(ch_d)
			ch_e = transform_vertical(ch_e)
			ch_f = transform_vertical(ch_f)
			

		ch_a = np.float32(ch_a).reshape(-1,3) + (7,5,0)
		ch_b = np.float32(ch_b).reshape(-1,3) + (4,5,0)
		ch_c = np.float32(ch_c).reshape(-1,3) + (1,5,0)
		ch_d = np.float32(ch_d).reshape(-1,3) + (7,2,0)
		ch_e = np.float32(ch_e).reshape(-1,3) + (4,2,0)
		ch_f = np.float32(ch_f).reshape(-1,3) + (1,2,0)

		for image in glob.glob('image/Q2_Image/*.bmp'):
			img = cv2.imread(image)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
			# If found, add object points, image points
			if ret == True:
				objp = np.zeros((nx*ny,3), np.float32)
				objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

				corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
				rev, rvecs, tvecs = cv2.solvePnP(objp, corners2, cameraMatrix, dist)

				#Project 3D points to image plane
				imgpts_a, jac = cv2.projectPoints(ch_a, rvecs, tvecs, cameraMatrix, dist)
				imgpts_b, jac = cv2.projectPoints(ch_b, rvecs, tvecs, cameraMatrix, dist)
				imgpts_c, jac = cv2.projectPoints(ch_c, rvecs, tvecs, cameraMatrix, dist)
				imgpts_d, jac = cv2.projectPoints(ch_d, rvecs, tvecs, cameraMatrix, dist)
				imgpts_e, jac = cv2.projectPoints(ch_e, rvecs, tvecs, cameraMatrix, dist)
				imgpts_f, jac = cv2.projectPoints(ch_f, rvecs, tvecs, cameraMatrix, dist)

				img = drawWords(img, corners2, imgpts_a, imgpts_b, imgpts_c, imgpts_d, imgpts_e, imgpts_f)
				img = cv2.resize(img, (1000,800), fx=100, fy= 100)
				cv2.imshow(image[15:]+'-'+Word_str,img)

				#path = 'image/Q2_Image/board'
				#if not onboard:	
				#	path = 'image/Q2_Image/vertical'
				#if not os.path.isdir(path):
				#	os.mkdir(path)
				#cv2.imwrite(os.path.join(path,Word_str+'-'+image[15:].replace("bmp","png")), img)
				cv2.waitKey(500)

		cv2.waitKey(0)
		cv2.destroyAllWindows()


				
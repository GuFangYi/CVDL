import numpy as np
import cv2
from matplotlib import pyplot as plt

framesize = (699,476)#1904/2, 2796/2
imgL = cv2.imread('image/Q3_Image/imL.png')
imgR = cv2.imread('image/Q3_Image/imR.png')
disparity = []

def draw_circle(event,x,y,flags,param):
    global imgL
    global imgR

    imL = cv2.resize(imgL, framesize)
    imR = cv2.resize(imgR, framesize)

    if event == cv2.EVENT_LBUTTONDBLCLK:

    	#print('(Y,X)= ', y, x)
    	#print('disparity: ',disparity[y][x])

    	# the image size is shrunk to 1/4
    	disp = disparity[y][x]/4
    	#print('disp: ',disp)
    	new_x = x - int(disp)
    	#print('new_x: ', new_x)
    	cv2.circle(imR,(new_x,y),5,(127,8,255),-1) #(255,255,0) turquoise
    	Hori = np.concatenate((imL,imR),axis = 1)
    	cv2.imshow('imgL/imgR', Hori)

    	
    	

def check_disparity(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print('(Y,X)= ', y, x)
        print('disparity: ',disparity[y][x])

def disparityMap():

	global imgL
	global imgR
	global disparity

	imL = cv2.resize(imgL, framesize)
	imR = cv2.resize(imgR, framesize)
	

	imL_gray = cv2.cvtColor(imL,cv2.COLOR_BGR2GRAY)
	imR_gray = cv2.cvtColor(imR,cv2.COLOR_BGR2GRAY)
	
	#height, width = imgL.shape[:2]
	#print(width)
	#print(height)

	stereo = cv2.StereoBM_create(numDisparities=64, blockSize=11)
	disparity = stereo.compute(imL_gray,imR_gray)

	#plt.imshow(disparity,'Accent')
	#plt.imshow(disparity,'gray')
	#plt.show()
	disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
   dtype=cv2.CV_8U)
	cv2.imshow('disparity', disparity)
	cv2.setMouseCallback('disparity',check_disparity)

	Hori = np.concatenate((imL,imR),axis = 1)
	cv2.imshow('imgL/imgR', Hori)
	#cv2.imshow('imgR', imR)
	#cv2.imshow('imgL', imL)
	#cv2.setMouseCallback('imgR',draw_circle)
	#cv2.setMouseCallback('imgL',draw_circle)
	cv2.setMouseCallback('imgL/imgR',draw_circle)


	cv2.waitKey(0)
	cv2.destroyAllWindows()


import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img = cv2.imread('image/Q4_Image/Shark1.jpg')
img2 = cv2.imread('image/Q4_Image/Shark2.jpg')
MIN_MATCH_COUNT = 10
UBIT = "grace20211029"
def detect_keypoints(img, sort=True):
	
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#sift
	sift = cv2.SIFT_create()
	keypoints_1, descriptors_1 = sift.detectAndCompute(img,None)

	if not sort:
		return keypoints_1, descriptors_1
	else:
		# sort the keypoints based on the size 
		keypoints_1 = sorted(keypoints_1, key = lambda x:x.size, reverse = True)[:200]
		
		keypoints, descriptors = sift.compute(img, keypoints_1)
		return keypoints, descriptors

def find_keypoints():
	global img, img2
	

	keypoints_1, descriptors_1 = detect_keypoints(img, True)
	keypoints_2, descriptors_2 = detect_keypoints(img2, True)
	# sort the keypoints based on the size 
	#keypoints_1 = sorted(keypoints_1, key = lambda x:x.size, reverse = True)[:200]
	#keypoints_2 = sorted(keypoints_2, key = lambda x:x.size, reverse = True)[:200]

	# change the radius on the image
	for key in keypoints_1:
		key.size = 5
	for key in keypoints_2:
		key.size = 5

	img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img_g2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

	#繪製關鍵點
	img_key = cv2.drawKeypoints(img_g, keypoints_1, None,flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	img2_key = cv2.drawKeypoints(img_g2, keypoints_2, None, flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	#img = cv2.drawKeypoints(img, keypoints=keypoints, outImage=img, color= (51, 163, 236), flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	#Hori = np.concatenate((img,img2),axis = 1)

	cv2.imshow('sift_keypoints_1',img_key)
	cv2.imshow('sift_keypoints_2',img2_key)
	cv2.imwrite('image/Q4_Image/FeatureShark2.jpg', img2_key)
	cv2.imwrite('image/Q4_Image/FeatureShark1.jpg',img_key)
	cv2.waitKey()
	cv2.destroyAllWindows()

def match_keypoints(keypoints_1, keypoints_2, descriptors_1, descriptors_2):
	#feature matching
	#bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
	bf = cv2.BFMatcher()
	#matches = bf.match(descriptors_1,descriptors_2)
	#goodMatches = sorted(matches, key = lambda x:x.distance)
	matches = bf.knnMatch(descriptors_1,descriptors_2, k=2)
	#print(matches)
	
	goodMatches = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			goodMatches.append(m)
	'''
	for m in matches:
		# ensure the distance is within a certain ratio of each
		# other (i.e. Lowe's ratio test)
		ratio = 0.75
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			goodMatches.append((m[0].trainIdx, m[0].queryIdx))
	'''

	return goodMatches

def draw_matchedKeypoints():

	global img, img2

	keypoints_1, descriptors_1 = detect_keypoints(img, True)
	keypoints_2, descriptors_2 = detect_keypoints(img2,True)

	matches = match_keypoints(keypoints_1, keypoints_2, descriptors_1, descriptors_2)
	
	img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img_g2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

	img3 = cv2.drawMatches(img_g, keypoints_1, img_g2, keypoints_2, matches[:50], img2, flags=2)
	#img3 = cv2.drawMatchesKnn(img,keypoints_1,img2,keypoints_2,matches,None,flags=2)
	img_out = cv2.normalize(img3,img3, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
   dtype=cv2.CV_8U)
	cv2.imshow('matched points', img_out)
	#plt.imshow(img3),plt.show()
def find_homographyMatrix(goodMatch, kp1, kp2):
	#Getting source and destination points
	srce_pts = np.float32([ kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
	dest_pts = np.float32([ kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
		
	#Finding Homography Matrix and mask
	homographyMat, mask = cv2.findHomography(srce_pts, dest_pts, cv2.RANSAC, 5.0)
	return homographyMat, mask
'''
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(descriptors_1,descriptors_2, k=2)
	
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)

	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		homographyMatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

		return homographyMatrix, mask
	return None


		matchesMask = mask.ravel().tolist()

		h,w = img.shape[:2]
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None

	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
	img3 = cv2.drawMatches(img,kp1,img2,kp2,good,None,**draw_params)
	#plt.imshow(img3, 'gray'),plt.show()
	img_out = cv2.normalize(img3,img3, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
   dtype=cv2.CV_8U)
	cv2.imshow('match point', img_out)
	'''

def warp_image():
	global img, img2
	kp1, descriptors_1 = detect_keypoints(img, False)
	kp2, descriptors_2 = detect_keypoints(img2, False)
	goodMatch = match_keypoints(kp1, kp2, descriptors_1, descriptors_2)
	
	H, mask = find_homographyMatrix(goodMatch, kp1, kp2)
	#Converting the mask to a list
	matchesMask = mask.ravel().tolist()
	
	h, w = img.shape[:2]
	pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1, 1, 2)

	matchIndex = []
	for i in range(len(matchesMask)):
		if (matchesMask[i]):
			matchIndex.append(i)

	matchArray = []
	for i in matchIndex:
		matchArray.append(goodMatch[i])

	#Finding 10 random matches using inliers
	np.random.seed(sum([ord(c) for c in UBIT]))
	randomMatch = np.random.choice(matchArray, 10, replace=False)
	#Defining draw parameters
	draw_params = dict(matchColor=(0, 0, 255),
	                   singlePointColor=None,
	                   flags=2)

	#Drawing the match image for 10 random points
	matchImage = cv2.drawMatches(img, kp1, img2, kp2, matchArray, None, **draw_params)

	#cv2.imshow('task1_matches', matchImage)

	h1, w1 = img2.shape[:2]
	h2, w2 = img.shape[:2]
	pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
	pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
	pts2_ = cv2.perspectiveTransform(pts2, H)
	pts = np.concatenate((pts1, pts2_), axis=0)

	#Finding the minimum and maximum coordinates
	[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
	t = [-xmin, -ymin]

	#Translating
	Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
	img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img_g2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	#Warping the first image on the second image using Homography Matrix
	result = cv2.warpPerspective(img_g, Ht.dot(H), (xmax-xmin, ymax-ymin))
	result[t[1]:h1+t[1], t[0]:w1+t[0]] = img_g2

	cv2.imshow('Warped image', result)

	'''
	imageA = cv2.imread('image/Q4_Image/Shark1.jpg')
	imageB = cv2.imread('image/Q4_Image/Shark2.jpg')
	imageA = imutils.resize(imageA, width=400)
	imageB = imutils.resize(imageB, width=400)

	stitcher = Stitcher()
	(result, vis) = stitcher.stitch([imageA, imageB], showMatches = True)
	cv2.imshow('Keypoints Matches', vis)
	cv2.imshow('result', result)
	cv2.waitKey(0)
	'''

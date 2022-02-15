import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
from scipy.stats import stats
import matplotlib.image as mpimg
# from numpy import linalg as LA

error_b = []
error_g = []
error_r = []
error = []
def sum_array(array):
	return sum(map(sum,array))

def Reconstruction_Error():
	if not error:
			Image_Reconstruction_PCA()
	# plt.figure(figsize = (50, 50)) 
	# plt.plot(error_b+error_g+error_r)
	print(error)
	# plt.show()


def Image_Reconstruction_PCA():
# https://towardsdatascience.com/dimensionality-reduction-of-a-color-photo-splitting-into-rgb-channels-using-pca-algorithm-in-python-ba01580a1118
	# fig = plt.figure(figsize = (50, 50)) 
	fig,axes = plt.subplots(4,15,figsize = (50, 50))
	# plt.subplots(4,15,figsize = (50, 50))
	
	for i in range(1,31):
		img = cv2.cvtColor(cv2.imread('./Q4_Image/'+str(i)+'.jpg'), cv2.COLOR_BGR2RGB)
		# print(img.shape)
		#split into 3 channels
		blue, green ,red = cv2.split(img)
		
		# fig = plt.figure(figsize = (15, 7.2)) 
		# fig.add_subplot(131)
		# plt.title("Blue Channel")
		# plt.imshow(blue)
		# fig.add_subplot(132)
		# plt.title("Green Channel")
		# plt.imshow(green)
		# fig.add_subplot(133)
		# plt.title("Red Channel")
		# plt.imshow(red)
		# plt.show()

		#verify data of a channel
		# blue_temp_df = pd.DataFrame(data=blue)
		#print(blue_temp_df)

		#noramlize
		df_blue = blue/255
		df_green = green/255
		df_red = red/255

		#reduce data from 400 dimensions to 50 dimensions 
		pca_b = PCA(n_components=50)
		pca_b.fit(df_blue)
		trans_pca_b = pca_b.transform(df_blue)

		pca_g = PCA(n_components=50)
		pca_g.fit(df_green)
		trans_pca_g = pca_g.transform(df_green)

		pca_r = PCA(n_components=50)
		pca_r.fit(df_red)
		trans_pca_r = pca_r.transform(df_red)
		# print(trans_pca_b.shape)

		#check sum of the most dominated 50 eigenvalues
		# print(f"Blue Channel : {sum(pca_b.explained_variance_ratio_)}")
		# print(f"Green Channel: {sum(pca_g.explained_variance_ratio_)}")
		# print(f"Red Channel  : {sum(pca_r.explained_variance_ratio_)}")
		# fig = plt.figure(figsize = (15, 7.2)) 
		# fig.add_subplot(131)
		# plt.title("Blue Channel")
		# plt.ylabel('Variation explained')
		# plt.xlabel('Eigen Value')
		# plt.bar(list(range(1,51)),pca_b.explained_variance_ratio_)
		# fig.add_subplot(132)
		# plt.title("Green Channel")
		# plt.ylabel('Variation explained')
		# plt.xlabel('Eigen Value')
		# plt.bar(list(range(1,51)),pca_g.explained_variance_ratio_)
		# fig.add_subplot(133)
		# plt.title("Red Channel")
		# plt.ylabel('Variation explained')
		# plt.xlabel('Eigen Value')
		# plt.bar(list(range(1,51)),pca_r.explained_variance_ratio_)
		# plt.show()

		#reconstruct images
		b_arr = pca_b.inverse_transform(trans_pca_b)
		g_arr = pca_g.inverse_transform(trans_pca_g)
		r_arr = pca_r.inverse_transform(trans_pca_r)
		# print(b_arr.shape, g_arr.shape, r_arr.shape)
		img_reduced= cv2.merge([b_arr, g_arr, r_arr])
		# print(img_reduced.shape)	

		#gray value
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_reduced = img_reduced.astype(np.float32)
		gray_reduced = cv2.cvtColor(gray_reduced, cv2.COLOR_BGR2GRAY)
		error.append(sum_array(gray-gray_reduced))


		#error
		# error_b.append(((df_blue-b_arr) **2).mean())
		# error_g.append(((df_green-g_arr) **2).mean())
		# error_r.append(((df_red-r_arr) **2).mean())


		if i<16:
			# fig.add_subplot(4,15,i)
			plt.subplot(4,15,i)
			plt.imshow(img)
			# plt.axis('off') #the label will also be off
			plt.xticks([])
			plt.yticks([])

			# fig.add_subplot(4,15,i+15)
			plt.subplot(4,15,i+15)
			plt.imshow(img_reduced)
			# plt.axis('off')
			plt.xticks([])
			plt.yticks([])

		else:
			# fig.add_subplot(4,15,i+15)
			plt.subplot(4,15,i+15)
			plt.imshow(img)
			# plt.axis('off')
			plt.xticks([])
			plt.yticks([])
			# fig.add_subplot(4,15,i+30)
			plt.subplot(4,15,i+30)
			plt.imshow(img_reduced)
			# plt.axis('off')
			plt.xticks([])
			plt.yticks([])

	# rows = ['origin','reconstruct','origin','reconstruct']
	# for ax, row in zip(axes[:,0], rows):
	# 	ax.set_ylabel(row, rotation=90, size = 'large')

	axes[0,0].set_ylabel('origin', size = '20')
	axes[1,0].set_ylabel('reconstruct', size = '20')
	axes[2,0].set_ylabel('origin', size = '20')
	axes[3,0].set_ylabel('reconstruct', size = '20')

	fig.tight_layout()
	plt.show()

	


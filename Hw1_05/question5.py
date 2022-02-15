from __future__ import print_function
import tensorflow as tf
import keras
import platform
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
import os
from keras.callbacks import EarlyStopping
from keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt 
from keras.utils import np_utils

import pytz
from datetime import datetime
import json
import keras.backend as backend
import numpy
import cv2

#Load data from cifar10

#model_filename = "vgg16_cifar10_10-29_09_56.h"
#hist_filename = "vgg16_cifar10_history_10-29_09_56.json"
model_filename = 'vgg16_cifar10_10-30_03_16.h5'
hist_filename = 'vgg16_cifar10_history_10-30_03_16.json'

batch_size=32
x_train = []
y_train = []
x_test = []
y_test = []

label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
	
datagen = []
num_classes=10
epochs=40

def load_images():
	global x_train, y_train, x_test, y_test
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	#return x_train, y_train, x_test, y_test

def train_images_display():
	global x_train, y_train, x_test, y_test, label_dict
	load_images()
	fig=plt.gcf()
	fig.set_size_inches(12,14)
	for i in range(0,9):
		plt.subplot(3,3,i+1)
		plt.imshow(x_test[i],cmap='binary')
		plt.title(label_dict[y_test[i][0]])
		plt.xticks([])
		plt.yticks([])

	plt.show()
	cv2.waitKey(0)
	
def VGG16(x_train):
	global num_classes
		#create model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same',data_format=None,
                 input_shape=x_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='valid'))
	model.add(Dropout(0.25))

	model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='valid'))
	model.add(Dropout(0.25))

	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='valid'))
	model.add(Dropout(0.25))

	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='valid'))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	return model

def create_model():
	global batch_size
	global x_train, y_train, x_test, y_test
	global datagen
	global num_classes
	global epochs
	data_augmentation=True

	#trained data and tested data
	if len(x_train) == 0:
		(x_train, y_train), (x_test, y_test)= cifar10.load_data()
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	#convert class vectors to binary class matrices
	y_train=keras.utils.np_utils.to_categorical(y_train,num_classes)
	y_test=keras.utils.np_utils.to_categorical(y_test,num_classes)

	model = VGG16(x_train)

	# initiate RMSprop optimizer
	#opt=optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	if not data_augmentation:
		print('Not using data augmentation.')
		model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
	else:
		print('Using real-time data augmentation.')
		# This will do preprocessing and realtime data augmentation:
		datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

		# Compute quantities required for feature-wise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(x_train)

	return model, datagen


def train_model():
	global batch_size
	global x_train, y_train, x_test, y_test
	global datagen
	global num_classes
	global epochs

	model, datagen = create_model()
	# Fit the model on the batches generated by datagen.flow().
	
	train_history = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))
	#score trained model
	scores=model.evaluate(x_test,y_test,verbose=1)
	print('test loss:', scores[0])
	print('test accuracy:', scores[1])

	save_model_and_history(model, train_history)
	
def save_model_and_history(model, train_history):
	timezone = pytz.timezone("Asia/Taipei")
	os_time = datetime.now()
	local_time = timezone.localize(os_time)
	time_stamp = local_time.strftime("%m-%d_%H_%M")

	# Save the model
	model_filename = 'vgg16_cifar10_'+time_stamp+'.h5'
	model.save(model_filename)
	print('Successfully save \'' + model_filename + '\'')

	# Save Training history as json file
	merge_hist = {**train_history.history, **train_history.params}
	j = json.dumps(merge_hist)
	hist_filename = 'vgg16_cifar10_history_'+time_stamp+'.json'
	with open(hist_filename, 'w') as file:
		file.write(j)
		print('Successfully write \'' + hist_filename + '\'')

def reload_model():
	global model_filename
	global hist_filename

	model = tf.keras.models.load_model(model_filename)
	
	with open(hist_filename) as file:
		hist = json.load(file)

	model.compile(loss='sparse_categorical_crossentropy',
	optimizer=optimizers.RMSprop(lr=1e-4),
	metrics=['acc'])


	return model, hist

def show_hyperparameters():
	new_model, hist = reload_model()

	print('batch size:', batch_size)
	print('learning rate:', backend.eval(new_model.optimizer.lr))
	print('optimizer:', new_model.optimizer.get_config()['name'])

def show_architecture():
	global x_train, y_train, x_test, y_test
	if len(x_train) == 0:
		(x_train, y_train), (x_test, y_test)= cifar10.load_data()

	model = VGG16(x_train)
	model.summary()

def show_training_loss_and_accuracy():
	global x_train, y_train, x_test, y_test
	if len(x_train) == 0:
		(x_train, y_train), (x_test, y_test)= cifar10.load_data()
	demo_model, hist = reload_model()
	#pred_res = demo_model.predict(numpy.array([x_test[0]]))
	#for i in pred_res:
	#	print(i)

	scores=demo_model.evaluate(x_test,y_test,verbose=1)
	print('test loss:', scores[0])
	print('test accuracy:', scores[1])

	fig = plt.figure('Accuracy')
	plt.clf()
	plt.subplot(2, 1, 1)
	plt.plot(hist['accuracy'])
	plt.plot(hist['val_accuracy'])
	plt.legend(['Training', 'Testing'])
	plt.xlabel('Epoch')
	plt.ylabel('accuracy')
	plt.title('Accuracy')
	plt.tight_layout()
	plt.subplot(2, 1, 2)
	plt.plot(hist['loss'])
	plt.plot(hist['val_loss'])
	plt.xlabel('Epoch')
	plt.ylabel('loss')
	plt.title('Loss')
	plt.legend(['Training', 'Testing'])
	plt.tight_layout()
	plt.show()

def predict_image(QLineEdit):
	global x_train, y_train, x_test, y_test, label_dict
	if len(x_train) == 0:
		(x_train, y_train), (x_test, y_test)= cifar10.load_data()
	nb = QLineEdit.text()
	if not nb:
		print('Enter number')
	else:
		test_ind = int(nb)
		demo_model, hist = reload_model()
		plt.figure(figsize=(10, 7))
		plt.subplot(2, 1, 1)
		plt.imshow(x_test[test_ind])
		plt.axis('off')
		print('Test Image Index:', test_ind)
		plt.title('Ans: ' + label_dict[y_test[test_ind][0]])
		# plot predict result bar chart
		y_predict = demo_model.predict(numpy.array([x_test[test_ind]]))
		max_predict = numpy.argmax(y_predict[0])
		title = 'Predict Result: '+ label_dict[max_predict]
		plt.title(title)

		label=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
		plt.subplot(2, 1, 2)
		#
		plt.bar(label, y_predict[0])
		plt.show()



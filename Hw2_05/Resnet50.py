import numpy as np
import pandas as pd
import os

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform

import shutil
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import time
import datetime
import pytz
import json
import torch
import cv2
import matplotlib.pyplot as plt

dir = "./cats_dogs_dataset/"

### build model START ###
def identity_block(X, f, filters, stage, block): 
    conv_name_base = 'res' + str(stage) + block + '_branch' 
    bn_name_base = 'bn' + str(stage) + block + '_branch' 
    F1, F2, F3 = filters 
    X_shortcut = X 
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X) 
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X) 
    X = Activation('relu')(X) 
   
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X) 
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X) 
    X = Activation('relu')(X) 
    
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X) 
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X) 
    
    X = Add()([X, X_shortcut]) 
    X = Activation('relu')(X) 
    return X 

def convolutional_block(X, f, filters, stage, block, s=2):
    
    conv_name_base = 'res' + str(stage) + block + '_branch' 
    bn_name_base = 'bn' + str(stage) + block + '_branch' 
    F1, F2, F3 = filters 
    
    X_shortcut = X 
    
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X) 
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X) 
    X = Activation('relu')(X) 
    
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X) 
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X) 
    X = Activation('relu')(X) 
    
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X) 
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X) 
    
    X_shortcut = Conv2D(F3, (1,1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut) 
    X_shortcut = BatchNormalization(axis = 3, name=bn_name_base + '1')(X_shortcut) 
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X) 
    return X

def ResNet50(input_shape=(64, 64, 3), classes=6):

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2,2), name="avg_pool")(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


def show_model():
    model = ResNet50(input_shape = (256, 256, 3), classes = 2)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

### build model END ###

### train and save model START ###
def split_train_validation():
    ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
    batch_size = 30
    height, width = (256,256)   
    imgdatagen = ImageDataGenerator(
        rescale = 1/255., 
        validation_split = 0.2,
    )
    train_dataset = imgdatagen.flow_from_directory(
        dir+'train',
        target_size = (height, width), 
        classes = ('dogs','cats'),
        batch_size = batch_size,
        subset = 'training'
    )

    val_dataset = imgdatagen.flow_from_directory(
        dir+'train',
        target_size = (height, width), 
        classes = ('dogs','cats'),
        batch_size = batch_size,
        subset = 'validation'
    )

    return train_dataset, val_dataset

def train_model():
    ### tensorboard
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = ResNet50(input_shape = (256, 256, 3), classes = 2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model，device_ids=[0,1,2])
    #     model.to(device)


    train_dataset, val_dataset = split_train_validation()
    history = model.fit(
        train_dataset,
        epochs=20,
        callbacks=[tensorboard_callback],
        validation_data = val_dataset,
        workers=10
    )
    save_model(model)
    evaluate_model(model,val_dataset)
    torch.save(model.state_dict(), dir)

def save_model(model):

    timezone = pytz.timezone("Asia/Taipei")
    os_time = datetime.now()
    local_time = timezone.localize(os_time)
    time_stamp = local_time.strftime("%m-%d_%H_%M")

    model_filename = 'kera_resnet50_aug_'+time_stamp+'.h5'
    model.save(model_filename)
    print('Successfully save \'' + model_filename + '\'')
    merge_hist = {**history_augm.history, **history_augm.params}
    j = json.dumps(merge_hist)
    hist_filename = 'kera_resnet50_history_aug_'+time_stamp+'.json'
    with open(hist_filename, 'w') as file:
      file.write(j)
      print('Successfully write \'' + hist_filename + '\'')




### train and save model END ###

### Dataset Pre-Processing START ###
def improve_dataset():
    trash_dir = dir+'trash/'
    if not os.path.exists(trash_dir):
        bad_dog_ids = [5604, 6413, 8736, 8898, 9188, 9517, 10161, 
                   10190, 10237, 10401, 10797, 11186]
        bad_cat_ids = [2939, 3216, 4688, 4833, 5418, 6215, 7377, 
                   8456, 8470, 11565, 12272]

        cleanup(bad_cat_ids, 'cats')
        cleanup(bad_dog_ids, 'dogs')

def cleanup(ids, dirname): 
    trash_dir = dir+'trash/'
    pattern = re.compile(r'.*\.(\d+)\..*')
    if not os.path.exists(trash_dir+dirname):
        os.makedirs(trash_dir+dirname)
        fnames = os.listdir(dir+'train/'+dirname)
        for fname in fnames:
            m = pattern.match(fname)
            if m: 
                # extract the id
                the_id = int(m.group(1))
                if the_id in ids:
                    # this id is in the list of ids to be trashed
                    print('moving to {}: {}'.format(trash_dir+dirname, fname))
                    shutil.move(os.path.join(dir+'train/'+dirname,fname),trash_dir+dirname)

def preprocess_dataset():
    source_dir = dir+"train"
    target_dir = dir+"train/cats"
    target_dir2 = dir+"train/dogs"
    file_names = os.listdir(source_dir)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        for filename in file_names:
            if filename.startswith("cat"):
                shutil.move(os.path.join(source_dir,filename),target_dir)
    if not os.path.exists(target_dir2):
        os.mkdir(target_dir2)
        for filename in file_names:
            if filename.startswith("dog"):
                shutil.move(os.path.join(source_dir,filename),target_dir2)

    improve_dataset()
### Dataset Pre-Processing END ###

### Prediction ###
def load(filename='kera_resnet50_12-17.h5'
        , hist_filename='kera_resnet50_12-17.json'):
    model_name = dir+filename
    hist = dir+hist_filename
    model = tf.keras.models.load_model(model_name)
    with open(hist) as file:
        history = json.load(file)   
    return model, history
def show_tensorboard():
    img_plot=cv2.imread(dir+'tensorboard.png')
    cv2.imshow('tensorboard',img_plot)

def test(QLineEdit):
    model, history = load()
    nb = QLineEdit.text()
    if not QLineEdit.text():
        print('input a value')
    elif int(nb)<1 or int(nb)>12500:
        print('input a number between 1~12500')
    else:
        cls_list = ['cats', 'dogs'] 
        img = image.load_img(dir+'/test/'+nb+'.jpg', target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        pred = model.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        print(nb+'.jpg')
        for i in top_inds:
            print('{}    {:.3f}  {}'.format(i,pred[i], cls_list[i]))

        #plot
        img_plot=cv2.imread(dir+'/test/'+nb+'.jpg')
        plt.figure(nb+'.jpg')
        plt.imshow(img_plot)
        if int(top_inds[0]) is 0:
            plt.title(cls_list[0])  
        else:      
            plt.title(cls_list[1])
        plt.show()

#https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
#https://www.tensorflow.org/tutorials/keras/save_and_load
def acc_comparison():
    # train_dataset, val_dataset = split_train_validation()
    # model, history = load()
    # _, acc = model.evaluate(train_dataset)
    # print('> %.3f' % (acc*100.0))
    # aug_model, aug_history=load('kera_resnet50_aug.h5','kera_resnet50_aug.json')
    # _, aug_acc = aug_model.evaluate(train_dataset)
    # print('> %.3f' % (aug_acc*100.0))
    
    
    # x = ['Before random erasing', 'After random erasing']
    # y = [98.04,96.64]
    # fig, ax = plt.subplots()    
    # ax.grid()
    # width = 0.4 # the width of the bars 
    # ind = np.arange(len(x))  # the x locations for the groups
    # ax.bar(ind, y, width, color="blue")
    # ax.set_xticks(ind)
    # ax.set_xticklabels(x, minor=False)
    # for index, data in enumerate(y):
    #     ax.text(x=index, y=data+1, s=f"{data}", color='blue', fontweight='bold')
    # plt.title('random erasing augmentation comparison')
    # plt.xlabel('x')
    # plt.ylabel('Accuracy')      
    # plt.show()
    img = cv2.imread(dir+'comparison_s.png')
    cv2.imshow('comparison', img)


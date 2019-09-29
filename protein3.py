# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 08:32:48 2018

@author: sn06
"""

import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage import transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

os.chdir('E:\\Kaggle\\HumanProtein\\Data')

THRESHOLD = 0.05

def f1_macro(y_true,y_pred):
    return f1_score(y_true,y_pred,average='macro')

def f1(y_true, y_pred):
    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)

training_path = os.getcwd() + '/train/'
adm = Adam(lr = 0.0001)


#read labels
train_labels = pd.read_csv('train.csv')

for i in range(28):
    train_labels['%s' % i] = 0
    
def split_targets(row):
    labels = row['Target'].split(' ')
    for i in labels:
        row['%s' % i] = 1
    return row    

train_labels = train_labels.apply(split_targets,axis=1)
train_labels['img_path'] = training_path + train_labels['Id']

#view images
def img(index):
    img = imread(training_path + os.listdir(training_path)[index])
    print('%s' % os.listdir(training_path)[index])
    plt.imshow(img)
    return img

def rescale(image):
    image = transform.rescale(image,1/4)
    return image

def create_model():
    model = Sequential()
    model.add(Conv2D(16,kernel_size=(3,3),strides=(1,1),activation='relu',input_shape=(128,128,4)))
    model.add(Conv2D(16,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.22))
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.22))
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.22))
    model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.22))
    model.add(Flatten())
    model.add(Dense(28,activation='relu'))
    model.add(Dense(28,activation='sigmoid'))
    model.compile(optimizer=adm,loss=f1_loss,metrics=['accuracy','top_k_categorical_accuracy',f1])
    return model

def read_image(img_path):
    X = []
    X.append(rescale(imread(img_path+'_%s.png' % 'red')))
    X.append(rescale(imread(img_path+'_%s.png' % 'green')))
    X.append(rescale(imread(img_path+'_%s.png' % 'blue')))
    X.append(rescale(imread(img_path+'_%s.png' % 'yellow')))
    X = np.dstack(X)
    return X

def get_data(df,path):
    X = []
    y = []
    for i in df['Id']:
        X.append(read_image(path + i))
        y.append(train_labels[train_labels['Id']==i].iloc[:,2:30].values)
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(-1,28)
    return X,y

if 'model' in globals():
   del(model)
model = create_model()
model.load_weights('protein3_weights.h5')

samples = 40
sample_size = 3200
training_f1 = []
training_valf1 = []

for i in range(samples):
    sample_index = np.random.choice(range(len(train_labels)),replace=False,size=sample_size)
    sample = train_labels.iloc[sample_index,:]
    
    X,y = get_data(sample,training_path)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
    del(X,y)

    history = model.fit(X_train,y_train,epochs=2,validation_data=(X_test,y_test))
    training_f1.append(history.history['f1'])
    training_valf1.append(history.history['val_f1'])
#    
#    plt.plot(history.history['f1'])
#    plt.plot(history.history['val_f1'])
#    
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
    model.save_weights('protein3_weights.h5')
    
def predict(index_start,index_end):
    test_path = os.getcwd() + '/test/'
    image_list = pd.read_csv('sample_submission.csv')
    image_index = range(index_start,index_end)
    df = image_list.iloc[image_index,:]
    df['Predicted'] = ''
    X,y = get_data(df,test_path)
    y_pred = model.predict(X)
    for j in range(len(y_pred)):
        y_pred_temp = y_pred[j]
        for k in range(len(y_pred_temp)):
            if y_pred_temp[k] >= 0.05:
                if df.iloc[j,1] == '':
                    df.iloc[j,1] = str(k)
                else:
                    df.iloc[j,1] = df.iloc[j,1] + ' ' + str(k)
    return df
    
    
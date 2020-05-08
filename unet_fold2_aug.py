import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import gc
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, confusion_matrix, recall_score, precision_score
import random
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
import time
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint  
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
from sklearn.metrics import f1_score
import tensorflow as tf
import re


def AIF_mae(y_true, y_pred) :

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    y_true = y_true.reshape(1, -1)[0]

    y_pred = y_pred.reshape(1, -1)[0]

    y_pred = np.where(y_pred<0.1, 0.099999, y_pred)
    over_threshold = y_true >= 0.1

    return np.mean(np.abs(y_true[over_threshold] - y_pred[over_threshold]))

def fscore(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    y_true = y_true.reshape(1, -1)[0]

    y_pred = y_pred.reshape(1, -1)[0]

    remove_NAs = y_true >= 0

    y_true = np.where(y_true[remove_NAs] >= 0.1, 1, 0)

    y_pred = np.where(y_pred[remove_NAs] >= 0.1, 1, 0)

    return(f1_score(y_true, y_pred))


def maeOverFscore(y_true, y_pred):
    return AIF_mae(y_true, y_pred) / (fscore(y_true, y_pred) + 1e-07)

def fscore_keras(y_true, y_pred):
    score = tf.py_function(func=fscore, inp=[y_true, y_pred], Tout=tf.float32, name='fscore_keras')
    return score

def maeOverFscore_keras(y_true, y_pred):
    score = tf.py_function(func=maeOverFscore, inp=[y_true, y_pred], Tout=tf.float32,  name='custom_mse') 
    return score


def mae_custom(y_true, y_pred):
    score = tf.py_function(func=AIF_mae, inp=[y_true, y_pred], Tout=tf.float32,  name='mae_custom') 
    return score

def fscore_(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true.reshape(1, -1)[0]

    y_pred = y_pred.reshape(1, -1)[0]


    y_true = np.where(y_true >= 0.5, 1, 0)

    y_pred = np.where(y_pred >= 0.5, 1, 0)

    return(f1_score(y_true, y_pred))




def fscore_classification(y_true, y_pred):
    score = tf.py_function(func=fscore_, inp=[y_true, y_pred], Tout=tf.float32, name='fscore_classification')
    return score



train_files = sorted(glob.glob('dataset/train/*.npy'))

tmp=[]
for file in tqdm(train_files):
    tmp.append(np.load(file))
    
train = np.array(tmp).astype(np.float32)
train[:,:,:,-1] = np.where(train[:,:,:,-1]<0, 0, train[:,:,:,-1])
del tmp
gc.collect()

tmp = []
for bt in tqdm([(3,8), (5,8), (1,4), (6, 8), (5, 9), (3, 9), (4, 9), (4, 8), (7, 9), (7,8), (6, 9), (4, 6), (3, 6), (4, 5), (3, 4),
          (6, 7), (1, 8), (3, 5), (4, 7), (1, 9), (2, 8), (5, 7), (2, 9), (2, 4)]):
    tmp.append(train[:,:,:,bt[0]-1]-train[:,:,:,bt[1]-1])
tmp = np.array(tmp).reshape(-1, 40, 40, 24)



train = np.append(tmp,train, axis=-1)
del tmp

skf = KFold(n_splits=5, shuffle= True, random_state=42)
kfolds = []
for train_idx, test_idx in skf.split(range(len(train))):
    kfolds.append((train_idx, test_idx))
    

    
fold = 2
X_train, y_train = train[kfolds[fold-1][0],:,:,:-1], train[kfolds[fold-1][0],:,:,-1]
X_valid, y_valid = train[kfolds[fold-1][1],:,:,:-1], train[kfolds[fold-1][1],:,:,-1]

tmp = []
for i in range(len(X_train)):
    if (y_train[i]>=0.1).sum()>=100:
        tmp.append(i)
        

X_train_aug90 = np.rot90(X_train[tmp], k=1, axes=(1,2))
y_train_aug90 = np.rot90(y_train[tmp], k=1, axes=(1,2))
X_train = np.concatenate([X_train, X_train_aug90])
y_train = np.concatenate([y_train, y_train_aug90])  

del train, X_train_aug90, y_train_aug90

batch_size=256
epochs=100
decay_st = (len(X_train)//batch_size+1)*epochs

from tensorflow.keras.layers import Dense, Input,BatchNormalization,LayerNormalization,Dropout,Conv2D,Conv2DTranspose, MaxPooling2D, Flatten,Activation, PReLU,concatenate, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2"])

poly_sche = tf.keras.optimizers.schedules.PolynomialDecay( 0.001, decay_st, end_learning_rate=1e-6, power=0.9)
opt = tf.keras.optimizers.Adam(poly_sche)

def unet():
    with strategy.scope():
        inputs = Input(shape=(40,40,38))
        inputs_ = UpSampling2D(size=(2,2))(inputs)
        conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs_)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('elu')(conv1)
        conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('elu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('elu')(conv2)
        conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('elu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('elu')(conv3)
        conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('elu')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('elu')(conv4)
        conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('elu')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('elu')(conv5)
        conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('elu')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        up6 = BatchNormalization()(up6)
        up6 = Activation('elu')(up6)
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('elu')(conv6)
        conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('elu')(conv6)

        up7 = Conv2D(256, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        up7 = BatchNormalization()(up7)
        up7 = Activation('elu')(up7)
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('elu')(conv7)
        conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('elu')(conv7)

        up8 = Conv2D(128, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        up8 = BatchNormalization()(up8)
        up8 = Activation('elu')(up8)
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Activation('elu')(conv8)
        conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Activation('elu')(conv8)

        up9 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        up9 = BatchNormalization()(up9)
        up9 = Activation('elu')(up9)
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation('elu')(conv9)
        conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation('elu')(conv9)
        conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation('elu')(conv9)
        conv9 = MaxPooling2D(pool_size=(2, 2))(conv9)
        conv9 = BatchNormalization()(conv9)
    #     classification = Conv2D(1, 1, activation = 'sigmoid', name='class')(conv9)
        regression = Conv2D(1, 1, activation = 'relu')(conv9)
        model = Model(inputs, regression)
        model.compile(loss='mae', optimizer=opt, metrics=[fscore_keras, maeOverFscore_keras, mae_custom] )
    return model

cp = ModelCheckpoint("regression_model/fold2_20200507_poly_elu_aug_38({epoch:02d})_val_maeOverFscore_keras({val_maeOverFscore_keras:.5f}).hdf5", 
                     monitor='val_maeOverFscore_keras', verbose=1, save_best_only=True, mode='min', period=3)
model = unet()
model.fit(X_train, y_train,
          validation_data = [X_valid, y_valid], 
          batch_size=batch_size, epochs=110,
          callbacks = [cp]
         )
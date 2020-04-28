import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import gc
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, confusion_matrix
import lightgbm as lgb
import random
import pickle
from tqdm import tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
import time
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint                                            # import
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from sklearn.metrics import f1_score

def AIF_mae(y_true, y_pred) :
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    y_true = y_true.reshape(1, -1)[0]
    
    y_pred = y_pred.reshape(1, -1)[0]
    
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


with open('data_generator/column.pickle','rb') as f:
    columns = pickle.load(f)

column={}
for idx, col in enumerate(columns):
    column[col] = idx

gc.collect()

train = pd.read_feather('data_generator/train_missing32f.feather')                                                  # 다시

train_1 = pd.concat([pd.read_feather('dataset/train_missing_btmean_{}of10.feather'.format(k)) for k in range(1,11)])
train_2 = pd.concat([pd.read_feather('dataset/train_missing_btmax-min_{}of10.feather'.format(k)) for k in range(1,11)])
train_3 = pd.concat([pd.read_feather('dataset/train_missing_btmax_{}of10.feather'.format(k)) for k in range(1,11)])
train_4=pd.read_feather('dataset/train_missing_32.feather')

for i in range(1,10):
    train['BT{}_mean3'.format(i)] = train_1['BT{}_mean'.format(i)].values
del train_1
for i in range(1,10):
    train['BT{}_max-min3'.format(i)] = train_2['BT{}_max-min'.format(i)].values
del train_2
for i in range(1,10):
    train['BT{}_max3'.format(i)] = train_3['BT{}_max'.format(i)].values
del train_3
for col in train_4.columns:
    train[col] = train_4[col].values
del train_4
gc.collect()

remove_column= ['BT1_mean5', 'BT2_mean5', 'BT7_max_val_diff5', 'BT8_max3',
       'BT7_mean_val_diff5', 'BT9_mean5', 'BT4_max_val_diff5', 'DPR_lat',
       'GMI_lat', 'BT5_max-min3', 'BT3_max_val_diff5', 'BT6_max-min3',
       'BT5_max_val_diff5', 'BT6_mean_val_diff5', 'BT2_MAX_val_diff5',
       'BT9_max3', 'BT1_MAX_val_diff5', 'GMI_lon', 'DPR_lon',
       'BT6_max_val_diff5', 'BT8_max5', 'BT9_max5', 'BT2_max_val_diff5',
       'BT1_max_val_diff5', 'type', 'BT7_max-min3']


from tensorflow.keras.layers import Dense, Input,BatchNormalization,LayerNormalization,Dropout,Conv2D, MaxPooling2D, Flatten,Activation, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])


from keras.utils.generic_utils import get_custom_objects
class Gelu(Activation):
    def __init__(self, activation, **kwargs):
        super(Gelu, self).__init__(activation, **kwargs)
        self.__name__='gelu'
        
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

get_custom_objects().update({'gelu': Gelu(gelu)})

adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)



def make_model(ver):
    drop=0.01
    act_func = 'relu'
    kernel_init = 'glorot_normal'
    
    unit = 256
    layer1 = 0
    layer2 = 64
    layer3 = 128
    layer4 = 256
#     with strategy.scope():
    input_data = Input(shape=(60+27+36,))
    dense_up_1 = Dense(unit+layer1, kernel_initializer=kernel_init)(input_data)
    dense_up_1 = BatchNormalization(trainable=True)(dense_up_1)
    dense_up_1 = PReLU()(dense_up_1)
    dense_up_1 = Dropout(drop)(dense_up_1)
    dense_up_2 = Dense(unit+layer2, kernel_initializer=kernel_init)(dense_up_1)
    dense_up_2 = BatchNormalization(trainable=True)(dense_up_2)
    dense_up_2 = PReLU()(dense_up_2)
    dense_up_2 = Dropout(drop)(dense_up_2)
    dense_up_3 = Dense(unit+layer3, kernel_initializer=kernel_init)(dense_up_2)
    dense_up_3 = BatchNormalization(trainable=True)(dense_up_3)
    dense_up_3 = PReLU()(dense_up_3)
    dense_up_3 = Dropout(drop)(dense_up_3)
    dense_up_4 = Dense(unit+layer4, kernel_initializer=kernel_init)(dense_up_3)
    dense_up_4 = BatchNormalization(trainable=True)(dense_up_4)
    dense_up_4 = PReLU()(dense_up_4)
    dense_up_4 = Dropout(drop)(dense_up_4)

    dense_down_1 = Dense(unit+layer3, kernel_initializer=kernel_init)(dense_up_4)
    dense_down_1 = BatchNormalization(trainable=True)(dense_down_1)
    dense_down_1 = PReLU()(dense_down_1)
    dense_down_1 = Dropout(drop)(dense_down_1)
    skip_1 = dense_down_1 + dense_up_3
    dense_down_2 = Dense(unit+layer2, kernel_initializer=kernel_init)(skip_1)
    dense_down_2 = BatchNormalization(trainable=True)(dense_down_2)
    dense_down_2 = PReLU()(dense_down_2)
    dense_down_2 = Dropout(drop)(dense_down_2)
    skip_2 = dense_down_2 + dense_up_2
    dense_down_3 = Dense(unit+layer1, kernel_initializer=kernel_init)(skip_2)
    dense_down_3 = BatchNormalization(trainable=True)(dense_down_3)
    dense_down_3 = PReLU()(dense_down_3)
    dense_down_3 = Dropout(drop)(dense_down_3)
    skip_3 = dense_down_3 + dense_up_1
    dense_down_4 = Dense(unit, kernel_initializer=kernel_init)(skip_3)
    dense_down_4 = BatchNormalization(trainable=True)(dense_down_4)
    dense_down_4 = PReLU()(dense_down_4)
    dense_down_4 = Dropout(drop)(dense_down_4)
    fc = Dense(64, kernel_initializer=kernel_init)(dense_down_4)
    fc = PReLU()(fc)
    if ver == 'regression':
        output_regression = Dense(1, name='regression')(fc)
        model = Model(input_data, output_regression)
    elif ver == 'classification':
        output_classification = Dense(1, activation='sigmoid', name='classification')(fc)
        model = Model(input_data, output_classification)
    print('new model loaded')
    return model



# select_mae = train.values[:,column['target']]>=0.1
# mae_train = train.values[select_mae]

skf = KFold(n_splits=5, shuffle= True, random_state=42)
kfolds = []
for train_idx, test_idx in skf.split(range(len(train))):
    kfolds.append((train_idx, test_idx))




def generator(x_data, y_data, batch_size):
    size = len(x_data)
    while True:
        np.random.seed(42)
        idx = np.random.permutation(size)
        x_data = x_data[idx]
        y_data = y_data[idx]
        
        for i in range(size//batch_size):
            x_batch = x_data[i*batch_size: (i+1)*batch_size]
            y_batch = y_data[i*batch_size: (i+1)*batch_size]
            
            
            diff_max_val = x_batch[:, range(column['BT1_max3'], column['BT9_max3']+1)] - x_batch[:, range(column['BT1'], column['BT9']+1)]
            diff_mean_val = x_batch[:, range(column['BT1_mean3'], column['BT9_mean3']+1)] - x_batch[:, range(column['BT1'], column['BT9']+1)]
            diff_max_min_val = x_batch[:, range(column['BT1_max-min3'], column['BT9_max-min3']+1)] - x_batch[:, range(column['BT1'], column['BT9']+1)]

            
            
            for idx1, idx2 in zip(range(column['BT1'], column['BT9']+1), range(column['BT1_max3'], column['BT9_max3']+1)):
                for idx3 in range(idx1, column['BT9']+1):
                    if idx1!=idx3:
                        x_batch = np.append(x_batch, ((x_batch[:,idx1]  - x_batch[:, idx3])/x_batch[:,idx2]).reshape(-1,1), axis=1)
                        
            x_batch = np.append(x_batch, diff_max_val, axis=1)
            x_batch = np.append(x_batch, diff_mean_val, axis=1)
            x_batch = np.append(x_batch, diff_max_min_val, axis=1)
            
            yield np.delete(x_batch,[column[col] for col in remove_column+['target']], axis=1), y_batch
            
            

            
            
batch_size=1600*16
kfold = kfolds[1]
print(kfold[:10])

y_train = train.values[:,column['target']]
y_train = np.where(y_train>=0.1, 1,0)
# train_generator = generator(mae_train[kfold[0][:]], mae_train[kfold[0][:],column['target']], batch_size)
# valid_generator = generator(mae_train[kfold[1][:]], mae_train[kfold[1][:],column['target']], batch_size)

train_generator = generator(train.values[kfold[0][:]], y_train[kfold[0]] , batch_size)
valid_generator = generator(train.values[kfold[1][:]], y_train[kfold[1]], batch_size)



cp = ModelCheckpoint("classification_model/fold2_onlymae_20200427_epoch({epoch:02d})_val_fscore_classification({val_fscore_classification:.5f}).hdf5", 
                     monitor='val_fscore_classification', verbose=1, save_best_only=True, mode='max', period=3)
rl = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, verbose=1, patience=5)


with strategy.scope():
    # model_regression = make_model('regression')
    # model_regression.compile(loss=['mae'], optimizer=adam)
    model_classification = make_model('classification')
    model_classification.compile(loss=['binary_crossentropy'], optimizer=adam, metrics=[fscore_classification])
gc.collect()
model_classification.fit(train_generator, steps_per_epoch=len(kfold[0][:])//batch_size, 
                     validation_data=valid_generator, validation_steps=len(kfold[1][:])//batch_size,
                     epochs=33, callbacks=[cp, rl]
                     )
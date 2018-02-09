
# Keras U-Net starter
안녕하세요. 이번에 DSB2018을 참여하며, 첫 주에는 지난 커널들을 분석해보기로 하였습니다. 

익숙하지 않은 keras이지만, 가장 많은 vote를 받았고, 이해하기도 편한 kernel을 들고 왔습니다.

https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277

### 목차
먼저 이 대회에 대해 알아보고, 이 대회에서 사용하는 모델을 이론적으로 살펴본 후에, 차례로 어떻게 적용하는지 확인해보도록 하겠습니다.

- [1. Data Science Bowl 2018?](#1-data-science-bowl-2018)
- [2. U-Net](#2-u-net)
- [3. 데이터 전처리](#3데이터-전처리)
- [4. Evaluation Function](#4-evaluation-function)
- [5. 네트워크 설계](#5-네트워크-설계)
- [6. Make Prediction](#6-make-prediction)
- [7. 결과 제출](#7-결과-제출)

---
## 1. Data Science Bowl 2018?

Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018/)은 다양한 세포 이미지들에서 nuclei(핵)를 찾는 competition입니다. 다양한 형태의 RGB 사진들을 받아서, 이를 구분하여 True(세포이다) 아니면 False(세포가 아니다)로 구분합니다. 예시를 보죠. ![1](../images/1.png) ![2](../images/2.png) 
뭐 이정도는 쉬운 문제죠. 하지만
![3](../images/3.png)![4](../images/4.png)

이렇게 말도 안되는 문제도 있습니다. 색깔도 그 때 그 때 다르고, 애매한 사진들도 많아서, 실제로 데이터를 보니깐 어려워보이더라구요. 하지만! 데이터가 크지 않아서 모델을 돌리는 데 시간이 오래 걸리지 않기 때문에 CNN을 배운 입장에서 시도하기 좋은 문제라고 생각했습니다. [Evaluation](https://www.kaggle.com/c/data-science-bowl-2018#evaluation)은 다음과 같습니다. 

$$
IoU(A, B) = \frac{A\cap B}{B \cup B}
$$

대부분의 kernel들은 유명한 Convolutional Neural Network들을 U-Net형식으로 묶어서 하더라구요. Structured machine learning계의 stacking과 같은 느낌인 것 같습니다.

---
## 2. U-Net
먼저, U-Net에 대해 이론적으로 알아보고, 기본적인 전처리 과정과 이를 어떻게 적용하였는지 알아보도록 하겠습니다.
U-Net 모델은 U-Net: Convolutional Networks for Biomedical Image Segmentation을 참고했고요, 최근 Ultrasound Nerve Segmentation competition에서 이를 활용한 케라스 코드가 나와서 이를 참조했습니다. this repo


```python
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed
```


```python
# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
```

---
## 3. 데이터 전처리


```python
# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')
```

    Getting and resizing train images and masks ... 


    100%|██████████| 670/670 [03:05<00:00,  3.62it/s]

    Getting and resizing test images ... 


    
    100%|██████████| 65/65 [00:01<00:00, 37.71it/s]

    Done!


    



```python
# Check if training data looks all right
ix = random.randint(0, len(train_ids))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
```


![png](output_8_0.png)



![png](output_8_1.png)


## 4. Evaluation Function

이제 앞서 소개했었던 IoU Evaluation function을 구현해보겠습니다.
mean average precision at different intersection over union (IoU) thresholds은 keras엔 기본적으로 없다고 하네요.

```python
# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
```

## 5. 네트워크 설계
한번 더 네트워크 구조를 보고 설계해봅시다.
![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)



```python
# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    input_1 (InputLayer)             (None, 128, 128, 3)   0                                            
    ____________________________________________________________________________________________________
    lambda_1 (Lambda)                (None, 128, 128, 3)   0           input_1[0][0]                    
    ____________________________________________________________________________________________________
    conv2d_1 (Conv2D)                (None, 128, 128, 16)  448         lambda_1[0][0]                   
    ____________________________________________________________________________________________________
    dropout_1 (Dropout)              (None, 128, 128, 16)  0           conv2d_1[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_2 (Conv2D)                (None, 128, 128, 16)  2320        dropout_1[0][0]                  
    ____________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)   (None, 64, 64, 16)    0           conv2d_2[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_3 (Conv2D)                (None, 64, 64, 32)    4640        max_pooling2d_1[0][0]            
    ____________________________________________________________________________________________________
    dropout_2 (Dropout)              (None, 64, 64, 32)    0           conv2d_3[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_4 (Conv2D)                (None, 64, 64, 32)    9248        dropout_2[0][0]                  
    ____________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)   (None, 32, 32, 32)    0           conv2d_4[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_5 (Conv2D)                (None, 32, 32, 64)    18496       max_pooling2d_2[0][0]            
    ____________________________________________________________________________________________________
    dropout_3 (Dropout)              (None, 32, 32, 64)    0           conv2d_5[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_6 (Conv2D)                (None, 32, 32, 64)    36928       dropout_3[0][0]                  
    ____________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)   (None, 16, 16, 64)    0           conv2d_6[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_7 (Conv2D)                (None, 16, 16, 128)   73856       max_pooling2d_3[0][0]            
    ____________________________________________________________________________________________________
    dropout_4 (Dropout)              (None, 16, 16, 128)   0           conv2d_7[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_8 (Conv2D)                (None, 16, 16, 128)   147584      dropout_4[0][0]                  
    ____________________________________________________________________________________________________
    max_pooling2d_4 (MaxPooling2D)   (None, 8, 8, 128)     0           conv2d_8[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_9 (Conv2D)                (None, 8, 8, 256)     295168      max_pooling2d_4[0][0]            
    ____________________________________________________________________________________________________
    dropout_5 (Dropout)              (None, 8, 8, 256)     0           conv2d_9[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_10 (Conv2D)               (None, 8, 8, 256)     590080      dropout_5[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_transpose_1 (Conv2DTransp (None, 16, 16, 128)   131200      conv2d_10[0][0]                  
    ____________________________________________________________________________________________________
    concatenate_1 (Concatenate)      (None, 16, 16, 256)   0           conv2d_transpose_1[0][0]         
                                                                       conv2d_8[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_11 (Conv2D)               (None, 16, 16, 128)   295040      concatenate_1[0][0]              
    ____________________________________________________________________________________________________
    dropout_6 (Dropout)              (None, 16, 16, 128)   0           conv2d_11[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_12 (Conv2D)               (None, 16, 16, 128)   147584      dropout_6[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_transpose_2 (Conv2DTransp (None, 32, 32, 64)    32832       conv2d_12[0][0]                  
    ____________________________________________________________________________________________________
    concatenate_2 (Concatenate)      (None, 32, 32, 128)   0           conv2d_transpose_2[0][0]         
                                                                       conv2d_6[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_13 (Conv2D)               (None, 32, 32, 64)    73792       concatenate_2[0][0]              
    ____________________________________________________________________________________________________
    dropout_7 (Dropout)              (None, 32, 32, 64)    0           conv2d_13[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_14 (Conv2D)               (None, 32, 32, 64)    36928       dropout_7[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_transpose_3 (Conv2DTransp (None, 64, 64, 32)    8224        conv2d_14[0][0]                  
    ____________________________________________________________________________________________________
    concatenate_3 (Concatenate)      (None, 64, 64, 64)    0           conv2d_transpose_3[0][0]         
                                                                       conv2d_4[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_15 (Conv2D)               (None, 64, 64, 32)    18464       concatenate_3[0][0]              
    ____________________________________________________________________________________________________
    dropout_8 (Dropout)              (None, 64, 64, 32)    0           conv2d_15[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_16 (Conv2D)               (None, 64, 64, 32)    9248        dropout_8[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_transpose_4 (Conv2DTransp (None, 128, 128, 16)  2064        conv2d_16[0][0]                  
    ____________________________________________________________________________________________________
    concatenate_4 (Concatenate)      (None, 128, 128, 32)  0           conv2d_transpose_4[0][0]         
                                                                       conv2d_2[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_17 (Conv2D)               (None, 128, 128, 16)  4624        concatenate_4[0][0]              
    ____________________________________________________________________________________________________
    dropout_9 (Dropout)              (None, 128, 128, 16)  0           conv2d_17[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_18 (Conv2D)               (None, 128, 128, 16)  2320        dropout_9[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_19 (Conv2D)               (None, 128, 128, 1)   17          conv2d_18[0][0]                  
    ====================================================================================================
    Total params: 1,941,105
    Trainable params: 1,941,105
    Non-trainable params: 0
    ____________________________________________________________________________________________________


Fitting해봅시다!


```python
# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, 
                    callbacks=[earlystopper, checkpointer])
```

    Train on 603 samples, validate on 67 samples
    Epoch 1/50
    592/603 [============================>.] - ETA: 2s - loss: 0.3950 - mean_iou: 0.4121Epoch 00000: val_loss improved from inf to 0.51264, saving model to model-dsbowl2018-1.h5
    603/603 [==============================] - 166s - loss: 0.3914 - mean_iou: 0.4127 - val_loss: 0.5126 - val_mean_iou: 0.4503
    Epoch 2/50
    592/603 [============================>.] - ETA: 2s - loss: 0.1820 - mean_iou: 0.5057Epoch 00001: val_loss improved from 0.51264 to 0.15816, saving model to model-dsbowl2018-1.h5
    603/603 [==============================] - 149s - loss: 0.1818 - mean_iou: 0.5065 - val_loss: 0.1582 - val_mean_iou: 0.5566
    Epoch 3/50
    592/603 [============================>.] - ETA: 2s - loss: 0.1418 - mean_iou: 0.5917Epoch 00002: val_loss improved from 0.15816 to 0.12711, saving model to model-dsbowl2018-1.h5
    603/603 [==============================] - 161s - loss: 0.1406 - mean_iou: 0.5922 - val_loss: 0.1271 - val_mean_iou: 0.6206
    Epoch 4/50
    592/603 [============================>.] - ETA: 2s - loss: 0.1319 - mean_iou: 0.6413Epoch 00003: val_loss did not improve
    603/603 [==============================] - 159s - loss: 0.1307 - mean_iou: 0.6416 - val_loss: 0.1347 - val_mean_iou: 0.6582
    Epoch 5/50
    592/603 [============================>.] - ETA: 2s - loss: 0.1146 - mean_iou: 0.6712Epoch 00004: val_loss improved from 0.12711 to 0.10697, saving model to model-dsbowl2018-1.h5
    603/603 [==============================] - 154s - loss: 0.1153 - mean_iou: 0.6714 - val_loss: 0.1070 - val_mean_iou: 0.6846
    Epoch 6/50
    592/603 [============================>.] - ETA: 3s - loss: 0.1026 - mean_iou: 0.6956Epoch 00005: val_loss improved from 0.10697 to 0.10526, saving model to model-dsbowl2018-1.h5
    603/603 [==============================] - 195s - loss: 0.1024 - mean_iou: 0.6958 - val_loss: 0.1053 - val_mean_iou: 0.7062
    Epoch 7/50
    592/603 [============================>.] - ETA: 10s - loss: 0.0947 - mean_iou: 0.7150Epoch 00006: val_loss did not improve
    603/603 [==============================] - 922s - loss: 0.0947 - mean_iou: 0.7151 - val_loss: 0.1177 - val_mean_iou: 0.7235
    Epoch 8/50
    592/603 [============================>.] - ETA: 100s - loss: 0.0949 - mean_iou: 0.7305Epoch 00007: val_loss improved from 0.10526 to 0.09623, saving model to model-dsbowl2018-1.h5
    603/603 [==============================] - 5426s - loss: 0.0943 - mean_iou: 0.7306 - val_loss: 0.0962 - val_mean_iou: 0.7366
    Epoch 9/50
    592/603 [============================>.] - ETA: 41s - loss: 0.0890 - mean_iou: 0.7425 Epoch 00008: val_loss improved from 0.09623 to 0.08743, saving model to model-dsbowl2018-1.h5
    603/603 [==============================] - 2232s - loss: 0.0888 - mean_iou: 0.7426 - val_loss: 0.0874 - val_mean_iou: 0.7480
    Epoch 10/50
    592/603 [============================>.] - ETA: 1273s - loss: 0.0884 - mean_iou: 0.7525Epoch 00009: val_loss improved from 0.08743 to 0.08054, saving model to model-dsbowl2018-1.h5
    603/603 [==============================] - 68555s - loss: 0.0883 - mean_iou: 0.7526 - val_loss: 0.0805 - val_mean_iou: 0.7571
    Epoch 11/50
    592/603 [============================>.] - ETA: 34s - loss: 0.0841 - mean_iou: 0.7613Epoch 00010: val_loss did not improve
    603/603 [==============================] - 1841s - loss: 0.0844 - mean_iou: 0.7613 - val_loss: 0.0812 - val_mean_iou: 0.7652
    Epoch 12/50
    592/603 [============================>.] - ETA: 16s - loss: 0.0844 - mean_iou: 0.7687Epoch 00011: val_loss improved from 0.08054 to 0.07918, saving model to model-dsbowl2018-1.h5
    603/603 [==============================] - 914s - loss: 0.0841 - mean_iou: 0.7687 - val_loss: 0.0792 - val_mean_iou: 0.7720
    Epoch 13/50
    592/603 [============================>.] - ETA: 10s - loss: 0.0838 - mean_iou: 0.7750Epoch 00012: val_loss did not improve
    603/603 [==============================] - 562s - loss: 0.0841 - mean_iou: 0.7750 - val_loss: 0.0797 - val_mean_iou: 0.7777
    Epoch 14/50
    592/603 [============================>.] - ETA: 37s - loss: 0.0803 - mean_iou: 0.7802Epoch 00013: val_loss improved from 0.07918 to 0.07805, saving model to model-dsbowl2018-1.h5
    603/603 [==============================] - 2037s - loss: 0.0800 - mean_iou: 0.7802 - val_loss: 0.0780 - val_mean_iou: 0.7827
    Epoch 15/50
    592/603 [============================>.] - ETA: 4s - loss: 0.0776 - mean_iou: 0.7852 Epoch 00014: val_loss did not improve
    603/603 [==============================] - 282s - loss: 0.0775 - mean_iou: 0.7852 - val_loss: 0.0785 - val_mean_iou: 0.7876
    Epoch 16/50
    592/603 [============================>.] - ETA: 2s - loss: 0.0780 - mean_iou: 0.7897Epoch 00015: val_loss improved from 0.07805 to 0.07490, saving model to model-dsbowl2018-1.h5
    603/603 [==============================] - 143s - loss: 0.0774 - mean_iou: 0.7897 - val_loss: 0.0749 - val_mean_iou: 0.7918
    Epoch 17/50
    592/603 [============================>.] - ETA: 2s - loss: 0.0754 - mean_iou: 0.7938Epoch 00016: val_loss did not improve
    603/603 [==============================] - 139s - loss: 0.0755 - mean_iou: 0.7938 - val_loss: 0.0797 - val_mean_iou: 0.7957
    Epoch 18/50
    592/603 [============================>.] - ETA: 2s - loss: 0.0771 - mean_iou: 0.7972Epoch 00017: val_loss did not improve
    603/603 [==============================] - 148s - loss: 0.0765 - mean_iou: 0.7972 - val_loss: 0.0808 - val_mean_iou: 0.7989
    Epoch 19/50
    592/603 [============================>.] - ETA: 2s - loss: 0.0744 - mean_iou: 0.8006Epoch 00018: val_loss did not improve
    603/603 [==============================] - 145s - loss: 0.0747 - mean_iou: 0.8007 - val_loss: 0.0782 - val_mean_iou: 0.8021
    Epoch 20/50
    592/603 [============================>.] - ETA: 2s - loss: 0.0737 - mean_iou: 0.8035Epoch 00019: val_loss did not improve
    603/603 [==============================] - 140s - loss: 0.0744 - mean_iou: 0.8035 - val_loss: 0.0761 - val_mean_iou: 0.8050
    Epoch 21/50
    592/603 [============================>.] - ETA: 2s - loss: 0.0757 - mean_iou: 0.8062Epoch 00020: val_loss did not improve
    603/603 [==============================] - 132s - loss: 0.0758 - mean_iou: 0.8063 - val_loss: 0.0790 - val_mean_iou: 0.8076
    Epoch 22/50
    592/603 [============================>.] - ETA: 2s - loss: 0.0798 - mean_iou: 0.8086Epoch 00021: val_loss did not improve
    603/603 [==============================] - 130s - loss: 0.0794 - mean_iou: 0.8087 - val_loss: 0.0770 - val_mean_iou: 0.8097
    Epoch 00021: early stopping


---
## 6. Make Prediction

원래 하던 것처럼, train-validation-test로 나누어서 최적화해봅시다.


```python
# Predict on train, val and test
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))
```


```python
# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()
```


```python
# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()
```

## 7. 결과 제출

어느정도 되었으니, 모델로 예측한 결과를 제출해봅시다. Image 제출에는 약간의 trick이 있어야 한다고 하네요 [이 거](https://www.kaggle.com/rakhlin/fast-run-length-encoding-python) 를 참조하여 결과를 만듭니당.


```python
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
```

각각의 test image에 대해 결과를 만들어보면 됩니다.


```python
new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
```

결과가 만들어졌습니다! 제출하면 끝이죵


```python
# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)
```

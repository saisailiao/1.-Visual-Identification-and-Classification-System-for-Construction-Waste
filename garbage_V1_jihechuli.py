#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import os,glob
from random import shuffle
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import random


# In[2]:


from PIL import Image
import cv2
import os,glob
base_path = "train_data_2"
lable_path_list = glob.glob(os.path.join(base_path, '*.txt'))
print(lable_path_list)


# In[3]:



IMAGE_DIMS = [64, 64]


# In[4]:


import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
#随机几何处理
def add_examples(img):
    random_num = random.randint(0,2)
    (h,w) = img.shape[:2]
    center = (w / 2,h / 2)
    if random_num == 0:      
        #旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
        M = cv2.getRotationMatrix2D(center,random.randint(-45,45),random.uniform(0.75, 1.3))
        new_image = cv2.warpAffine(img,M,(w,h))
        plt.imshow(new_image)
        plt.show()
    #图像翻转
    if random_num == 1: 
        new_image = cv2.flip(img,random.randint(-1,1))
    if random_num == 2: 
    #图片偏移
        M = np.float32([[1,0,random.randint(-100,100)],[0,1,random.randint(-100,100)]])
        new_image = cv2.warpAffine(img,M,(w, h))
    return new_image


# In[5]:


from random import shuffle
import random
import numpy
from tensorflow.keras.preprocessing.image import img_to_array
def get_shulffer_data(txt_list, IMAGE_DIMS, base_path,ratio):
    img_list = []
    lable_list = []
    length = len(txt_list)
    shuffle(txt_list)
    for _ in np.arange(length):
        with open(txt_list[_], "r") as f:
            data = f.readline()
            lable = data.split(',')[1].split()
            img = cv2.imread(os.path.join(base_path,data.split(',')[0]))
            #随机几何处理
            while random.uniform(0, 1) < ratio:
                new_img = add_examples(img)
                new_img = cv2.resize(new_img,(IMAGE_DIMS[0],IMAGE_DIMS[1]))
                new_img = img_to_array(new_img)
                new_img /= 255.0
                lable_list.append(lable)
                img_list.append(new_img)
                new_img = add_examples(img)
                new_img = cv2.resize(new_img,(IMAGE_DIMS[0],IMAGE_DIMS[1]))
                new_img = img_to_array(new_img)
                new_img /= 255.0
                lable_list.append(lable)
                img_list.append(new_img)
            img = cv2.resize(img,(IMAGE_DIMS[0],IMAGE_DIMS[1]))
            img = img_to_array(img)
            img /= 255.0
            lable_list.append(lable)
            img_list.append(img)
    return np.array(img_list), np.array(lable_list)

img_list, lable_list = get_shulffer_data(lable_path_list, IMAGE_DIMS, base_path, 0.8)


# In[6]:


lable_list = lable_list.reshape(len(lable_list))


# In[7]:


length_sum = len(lable_list)
length_sum


# In[8]:


lable_list_numtype = lable_list.copy()


# In[9]:


category_dir = {"14": "0","15": "1","19": "2", "20": "3", "26": "4", "30": "5", "36": "6", "37": "7","46":"8","41":"9"}


# In[10]:


for _ in np.arange(length_sum):
    lable_list[_] = category_dir[lable_list_numtype[_]]

from keras.utils import np_utils
lable_list_OneHot = np_utils.to_categorical(lable_list)


# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
# 搭面包架子
model = Sequential()


# In[12]:


model.add(Conv2D(filters=64,kernel_size=(3, 3),input_shape=(64, 64,3),
                 activation='relu', padding='same'))
model.add(Dropout(0.35))
model.add(MaxPooling2D(pool_size=(2, 2)))

#卷积层2池化层2
model.add(Conv2D(filters=256, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(Dropout(0.35))

model.add(MaxPooling2D(pool_size=(2, 2)))
#卷积层3池化层3
model.add(Conv2D(filters=512, kernel_size=(3, 3), 
                 activation='relu', padding='same'))
model.add(Dropout(0.35))
model.add(MaxPooling2D(pool_size=(2, 2)))
#建立神经网络（flatten、 隐藏层、输出层）
model.add(Flatten())
model.add(Dropout(0.35))
model.add(Dense(2500, activation='relu',kernel_regularizer = keras.regularizers.l2(0.01)))
model.add(Dropout(0.35))
model.add(Dense(1500, activation='relu', kernel_regularizer = keras.regularizers.l2(0.01)))
model.add(Dropout(0.35))
model.add(Dense(8, activation='softmax'))


# In[ ]:





# In[13]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

train_history=model.fit(img_list, lable_list_OneHot,
                        validation_split=0.2,
                        epochs=50, batch_size=300, verbose=1)


# In[ ]:





# In[ ]:





import os,cv2
import tensorflow as tf
from keras.utils import np_utils
import numpy as np
import pandas as pd
from random import shuffle
import pickle 
import numpy as np
import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 
import matplotlib.pyplot as plt

img_path='gray'

CLASS_MAP = { #any new class has to be added here
    "aphids":0,
    "gallyflies":1,
    "leaf feeding caterpillar":2,
    "scrab beetle":3
}

NUM_CLASSES = len(CLASS_MAP)


def mapper(val):
    return CLASS_MAP[val]

def load_data():
    dataset=[]
    #label=[]
    for directory in os.listdir(img_path):
        count=0
        path = os.path.join(img_path, directory)
        if not os.path.isdir(path):
            continue
        for item in os.listdir(path):
        # to make sure no hidden files get in our way
            if item.startswith("."):
                continue
            img = cv2.imread(os.path.join(path, item),cv2.IMREAD_GRAYSCALE)
        #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img=cv2.Canny(img,100,200)
            #img = cv2.resize(img, (227, 227))
        #img=cv2.flip(img,1)
        #save_path=path+'/'+str(count)+'.jpg'
        #save_path = 'edges'+'/'+directory+'/'+' '+str(count)+'.jpg'
        #cv2.imwrite(save_path, img)
            #img=np.array(img).reshape(-1,227,227,1)
            #img = np.expand_dim(img, axis=0)
            #img=img.reshape(227*227)
            dataset.append([img,directory])
            #label.append(directory)
            count=count+1
        
    return dataset

LR = 1e-3    
IMG_SIZE=227
dataset=load_data()
MODEL_NAME='cnngray-{}-{}.model'.format(LR, '6conv-basic') 
shuffle(dataset)
data, labels = zip(*dataset)
labels = list(map(mapper, labels))
labels = np_utils.to_categorical(labels)
train_x,train_y=data[:220],labels[:220]
test_x,test_y=data[220:],labels[220:]
X = np.array([i for i in train_x]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
#Y = [i[1] for i in train] 
test_xx = np.array([i for i in test_x]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
#test_y = [i[1] for i in test] 

tf.reset_default_graph() 
convnet = input_data(shape =[None, 227, 227, 1], name ='input') 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 128, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = fully_connected(convnet, 1024, activation ='relu') 
convnet = dropout(convnet, 0.4) 
  
convnet = fully_connected(convnet, 4, activation ='softmax') 
convnet = regression(convnet, optimizer ='adam', learning_rate = LR, 
      loss ='categorical_crossentropy', name ='targets') 
  
model = tflearn.DNN(convnet, tensorboard_dir ='log')
model.fit({'input': X}, {'targets': train_y}, n_epoch = 32,  
    validation_set =({'input': test_xx}, {'targets': test_y}),  
    snapshot_step = 500, show_metric = True, run_id = MODEL_NAME) 
model.save(MODEL_NAME) 
import os,cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from random import shuffle
import pickle 

img_path='edges'

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
            img=np.array(img)
            img=img.reshape(227*227)
            dataset.append([img,directory])
            #label.append(directory)
            count=count+1
        
    return dataset

def train(x,y):
    
    train_x,train_y=x[:220],y[:220]
    test_x,test_y=x[220:],y[220:]
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(train_x,train_y)

    y_knn_pred = knn_clf.predict(test_x)
    accuracy = accuracy_score(test_y, y_knn_pred)*100
    ans = [accuracy]
    knnPickle = open('knnpickle_file', 'wb') 
    pickle.dump(knn_clf, knnPickle) 
    return ans
dataset=load_data()
shuffle(dataset)
data, labels = zip(*dataset)
#print(len(y))
zz=train(data,labels)
print(zz)


    
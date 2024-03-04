import os,cv2
img_path='res'
i=0
dataset=[]
labels=['a','b','c','d']
for directory in os.listdir(img_path):
    count=0
    path = os.path.join(img_path, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # to make sure no hidden files get in our way
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.Canny(img,100,200)
        img = cv2.resize(img, (227, 227))
        #img=cv2.flip(img,1)
        #save_path=path+'/'+str(count)+'.jpg'
        save_path = 'edges'+'/'+directory+'/'+' '+str(count)+'.jpg'
        cv2.imwrite(save_path, img)
        dataset.append(img)
        count=count+1
    i=i+1
    

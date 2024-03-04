import os,cv2
img_path='res'
for directory in os.listdir(img_path):
    count=0
    path = os.path.join(img_path, directory)
    for item in os.listdir(path):
        img = cv2.imread(os.path.join(path, item))
        img1=cv2.flip(img,1)
        save_path = directory+str(count)+'.jpg'
        cv2.imwrite(save_path,img1)
        count+=1

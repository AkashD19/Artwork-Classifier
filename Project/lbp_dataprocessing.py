import cv2
# To performing path manipulations 
import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
# Utility package -- use pip install cvutils to install
import cvutils
# To read class from file
import csv
from PIL import Image
import glob
import numpy as np
from sklearn.svm import LinearSVC

image_list1 = []
image_list2 = []
image_list3 = []
image_list4 = []
# Store the path of training images in train_images
#train_images = cvutils.imlist("C:\Users\dutta\Desktop\Project\Images\Train")

#print train_images
#img1=imread(train_images[1])
#imshow('img1',img1)

for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Train\Cubism/*.jpg'): #assuming gif
    im=cv2.imread(filename,0)
    image_list1.append(im)

for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Train\Impressionism/*.jpg'): #assuming gif
    im=cv2.imread(filename,0)
    image_list2.append(im)
    
for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Train\Pop Art/*.jpg'): #assuming gif
    im=cv2.imread(filename,0)
    image_list3.append(im)
    
for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Train\Realism/*.jpg'): #assuming gif
    im=cv2.imread(filename,0)
    image_list4.append(im)
    
    
#print image_list[1]
#cv2.imshow('image',image_list[1])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
labels=[]
data=[]
def localbinarypattern(image):
     lbp=local_binary_pattern(image,24,8,method="uniform")
     (hist,_)=np.histogram(lbp.ravel(),bins=np.arange(0,27),range=(0,26))
     hist=hist.astype("float")
     hist/=(hist.sum()+(1e-7))
     return hist
    
for img in image_list1:
     h=localbinarypattern(img) 
     labels.append("Cubism")
     data.append(h)
     
for img in image_list2:
     h=localbinarypattern(img) 
     labels.append("Impressionism")
     data.append(h)
     
for img in image_list3:
     h=localbinarypattern(img) 
     labels.append("Pop Art")
     data.append(h)
     
for img in image_list4:
     h=localbinarypattern(img) 
     labels.append("Realism")
     data.append(h)

f = open("data.txt", "w") 
f.write(str([data, labels]))
f.close()   
     
#model=LinearSVC(C=100.0,random_state=42)
#model.fit(data,labels)


#for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test/*.jpg'): #assuming gif
 #   img1=cv2.imread(filename,0)
  #  h1=localbinarypattern(img1)
   # prediction=model.predict([h1])[0]
   # cv2.putText(img1, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
   # cv2.imshow("Image", img1)
   # cv2.waitKey(0)
     




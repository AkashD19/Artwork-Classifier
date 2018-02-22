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
from sklearn.neighbors import KNeighborsClassifier

training_data, responses = eval(open('data11.txt', 'r').read())
training_data = np.array(training_data, np.float32)
responses = np.array(responses)


def localbinarypattern(image):
     lbp=local_binary_pattern(image,24,8,method="uniform")
     (hist,_)=np.histogram(lbp.ravel(),bins=np.arange(0,27),range=(0,26))
     hist=hist.astype("float")
     hist/=(hist.sum()+(1e-7))
     return hist

def compute(img):
		averages = np.zeros((8,8,3))
		imgH, imgW, _ = img.shape
		for row in range(8):
			for col in range(8):
				slice = img[imgH/8 * row: imgH/8 * (row+1), imgW/8*col : imgW/8*(col+1)]
				average_color_per_row = np.mean(slice, axis=0)
				average_color = np.mean(average_color_per_row, axis=0)
				average_color = np.uint8(average_color)
				averages[row][col][0] = average_color[0]
				averages[row][col][1] = average_color[1]
				averages[row][col][2] = average_color[2]
		icon = cv2.cvtColor(np.array(averages, dtype=np.uint8), cv2.COLOR_BGR2YCR_CB)
		y, cr, cb = cv2.split(icon)
		dct_y = cv2.dct(np.float32(y))
		dct_cb = cv2.dct(np.float32(cb))
		dct_cr = cv2.dct(np.float32(cr))
		dct_y_zigzag = []
		dct_cb_zigzag = []
		dct_cr_zigzag = []
		flip = True
		flipped_dct_y = np.fliplr(dct_y)
		flipped_dct_cb = np.fliplr(dct_cb)
		flipped_dct_cr = np.fliplr(dct_cr)
		for i in range(8 + 8 -1):
			k_diag = 8 - 1 - i
			diag_y = np.diag(flipped_dct_y, k=k_diag)
			diag_cb = np.diag(flipped_dct_cb, k=k_diag)
			diag_cr = np.diag(flipped_dct_cr, k=k_diag)
			if flip:
				diag_y = diag_y[::-1]
				diag_cb = diag_cb[::-1]
				diag_cr = diag_cr[::-1]
			dct_y_zigzag.append(diag_y)
			dct_cb_zigzag.append(diag_cb)
			dct_cr_zigzag.append(diag_cr)
			flip = not flip
		return np.concatenate([np.concatenate(dct_y_zigzag), np.concatenate(dct_cb_zigzag), np.concatenate(dct_cr_zigzag)])

 
model=LinearSVC(C=150,random_state=1)
print "C=150,random_state=1"
#model=KNeighborsClassifier(n_neighbors=50)
model.fit(training_data,responses)

image_list1 = []
image_list2 = []
image_list3 = []
image_list4 = []


image1=[]
image2=[]
image3=[]
image4=[]

data=[]
labels=[]

for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Cubism/*.jpg'): #assuming gif
    im=cv2.imread(filename,0)
    image_list1.append(im)

for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Impressionism/*.jpg'): #assuming gif
    im=cv2.imread(filename,0)
    image_list2.append(im)
    
for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Pop Art/*.jpg'): #assuming gif
    im=cv2.imread(filename,0)
    image_list3.append(im)
    
for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Realism/*.jpg'): #assuming gif
    im=cv2.imread(filename,0)
    image_list4.append(im)
    
for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Cubism/*.jpg'): #assuming gif
    im=cv2.imread(filename)
    image1.append(im)

for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Impressionism/*.jpg'): #assuming gif
    im=cv2.imread(filename)
    image2.append(im)
    
for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Pop Art/*.jpg'): #assuming gif
    im=cv2.imread(filename)
    image3.append(im)
    
for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Realism/*.jpg'): #assuming gif
    im=cv2.imread(filename)
    image4.append(im)
  
for img,img1 in zip(image_list1,image1):
     h=localbinarypattern(img)
     g=compute(img1)
     labels.append("Cubism")
     data.append(h)
     data.append(g)
f = open("test_cubism.txt", "w") 
f.write(str([data, labels]))
f.close()
data=[]
labels=[]
     
for img,img1 in zip(image_list2,image2):
     h=localbinarypattern(img) 
     g=compute(img1)
     labels.append("Impressionism")
     data.append(h)
     data.append(g)
f = open("test_impressionism.txt", "w") 
f.write(str([data, labels]))
f.close()
data=[]
labels=[]
     
for img,img1 in zip(image_list3,image3):
     h=localbinarypattern(img) 
     g=compute(img1)
     labels.append("Pop Art")
     data.append(h)
     data.append(g)
f = open("test_popart.txt", "w") 
f.write(str([data, labels]))
f.close()
data=[]
labels=[]
     
for img,img1 in zip(image_list4,image4):
     h=localbinarypattern(img) 
     g=compute(img1)
     labels.append("Realism")
     data.append(h)
     data.append(g)
f = open("test_realism.txt", "w") 
f.write(str([data, labels]))
f.close()
data=[]
labels=[]


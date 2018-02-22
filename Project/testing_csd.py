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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB


training_data, responses = eval(open('data1.txt', 'r').read())
training_data = np.array(training_data, np.float32)
responses = np.array(responses)

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

#model=KNeighborsClassifier(n_neighbors=300)
#print "n_neighbors=300"
#model=LinearSVC(C=10,random_state=10)
#print "C=10,random_state=100"
#model=RandomForestClassifier(random_state=50)
#model = DecisionTreeClassifier(random_state=50)
#print "DecisionTreeClassifier(random_state=50)"
#model = AdaBoostClassifier(n_estimators=200)
#print "AdaBoostClassifier(n_estimators=200)"
model = BernoulliNB()
print "LogisticRegression(C=150)"
model.fit(training_data,responses)


count=0
c=0
for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Cubism/*.jpg'): #assuming gif
    img1=cv2.imread(filename)
    h1=compute(img1)
    prediction=model.predict([h1])[0]
    if(prediction == "Cubism"):
        count+=1
print "Cubism Count=",count
c=c+count
count=0

for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Impressionism/*.jpg'): #assuming gif
    img1=cv2.imread(filename)
    h1=compute(img1)
    prediction=model.predict([h1])[0]
    if(prediction == "Impressionism"):
        count+=1
print "Impressionism Count=",count
c=c+count
count=0

for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Pop Art/*.jpg'): #assuming gif
    img1=cv2.imread(filename)
    h1=compute(img1)
    prediction=model.predict([h1])[0]
    if(prediction == "Pop Art"):
        count+=1
print "Pop Art Count=",count
c=c+count
count=0

for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Realism/*.jpg'): #assuming gif
    img1=cv2.imread(filename)
    h1=compute(img1)
    prediction=model.predict([h1])[0]
    if(prediction == "Realism"):
        count+=1
print "Realism Count=",count
c=c+count
print c
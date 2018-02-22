import cv2
# To performing path manipulations 
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 

# Utility package -- use pip install cvutils to install

# To read class from file

import glob
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


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
#model=GaussianNB()
model=RandomForestClassifier(random_state=7)
#model=LinearSVC(C=100.0,random_state=1)
        
        
model.fit(training_data)


count=0
c=0
for filename in glob.glob('C:\Users\Nivetha\Cubism/*.jpg'): #assuming gif
    img1=cv2.imread(filename)
    h1=compute(img1)
    prediction=model.predict([h1])[0]
    if(prediction == "Pop Art"):
        c+=1
    if(prediction == "Cubism"):
        count+=1
print "Cubism Count=",count
c=c+count
count=0

for filename in glob.glob('C:\Users\Nivetha\Impressionism/*.jpg'): #assuming gif
    img1=cv2.imread(filename)
    h1=compute(img1)
    prediction=model.predict([h1])[0]
    if(prediction == "Pop Art"):
        c+=1
    if(prediction == "Impressionism"):
        count+=1
print "Impressionism Count=",count
c=c+count
count=0

for filename in glob.glob('C:\Users\Nivetha\Pop Art/*.jpg'): #assuming gif
    img1=cv2.imread(filename)
    h1=compute(img1)
    prediction=model.predict([h1])[0]
    if(prediction == "Pop Art"):
        c+=1
    if(prediction == "Pop Art"):
        count+=1
print "Pop Art Count=",count
c=c+count
count=0

for filename in glob.glob('C:\Users\Nivetha\Realism/*.jpg'): #assuming gif
    img1=cv2.imread(filename)
    h1=compute(img1)
    prediction=model.predict([h1])[0]   
    if(prediction == "Realism"):
        count+=1
print "Realism Count=",count

print "Accuracy(%)=",c/4 
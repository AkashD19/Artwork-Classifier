import cv2
# To performing path manipulations 

# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 

import glob
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier



training_data, responses = eval(open('data.txt', 'r').read())
training_data = np.array(training_data, np.float32)
responses = np.array(responses)


def localbinarypattern(image):
     lbp=local_binary_pattern(image,24,8,method="uniform")
     (hist,_)=np.histogram(lbp.ravel(),bins=np.arange(0,27),range=(0,26))
     hist=hist.astype("float")
     hist/=(hist.sum()+(1e-7))
     return hist
#model = MultinomialNB()
#model = GaussianNB()
model=RandomForestClassifier(random_state=50)
 
#model=LinearSVC(C=100.0,random_state=1)
model.fit(training_data,responses)



count=0
c=0
for filename in glob.glob('C:\Users\Nivetha\Cubism/*.jpg'): #assuming gif
    img1=cv2.imread(filename,0)
    h1=localbinarypattern(img1)
    prediction=model.predict([h1])[0]
    if(prediction == "Pop Art"):
        c+=1
    if(prediction == "Cubism"):
        count+=1
print "Cubism Count=",count
c=c+count
count=0

for filename in glob.glob('C:\Users\Nivetha\Impressionism/*.jpg'): #assuming gif
    img1=cv2.imread(filename,0)
    h1=localbinarypattern(img1)
    prediction=model.predict([h1])[0]
    if(prediction == "Pop Art"):
        c+=1
    if(prediction == "Impressionism"):
        count+=1
print "Impressionism Count=",count
c=c+count
count=0

for filename in glob.glob('C:\Users\Nivetha\Pop Art/*.jpg'): #assuming gif
    img1=cv2.imread(filename,0)
    h1=localbinarypattern(img1)
    prediction=model.predict([h1])[0]
    if(prediction == "Pop Art"):
        c+=1
    if(prediction == "Pop Art"):
        count+=1
print "Pop Art Count=",count
c=c+count
count=0

for filename in glob.glob('C:\Users\Nivetha\Realism/*.jpg'): #assuming gif
    img1=cv2.imread(filename,0)
    h1=localbinarypattern(img1)
    prediction=model.predict([h1])[0]   
    if(prediction == "Realism"):
        count+=1
print "Realism Count=",count

print "Accuracy(%)=",c/4 
     
#print training_data
#print responses

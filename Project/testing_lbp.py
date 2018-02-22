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
import matplotlib.pyplot as plt
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

training_data, responses = eval(open('data.txt', 'r').read())
training_data = np.array(training_data, np.float32)
responses = np.array(responses)


def localbinarypattern(image):
     lbp=local_binary_pattern(image,24,8,method="uniform")
     (hist,_)=np.histogram(lbp.ravel(),bins=np.arange(0,27),range=(0,26))
     hist=hist.astype("float")
     hist/=(hist.sum()+(1e-7))
     return hist
x=[]
y=[]
for i in range(1,300,10):
    #model=LinearSVC(C=150,random_state=1)
    #print "C=150,random_state=1"
    model=KNeighborsClassifier(n_neighbors=i)
    #model = DecisionTreeClassifier(random_state=50)
    #print "DecisionTreeClassifier(random_state=50)"
    #model = AdaBoostClassifier(n_estimators=200)
    #print "AdaBoostClassifier(n_estimators=200)"
     
    #model = BernoulliNB()
    #print "LogisticRegression(C=200)"
    model.fit(training_data,responses)

    count=0
    c=0
    for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Cubism/*.jpg'): #assuming gif
        img1=cv2.imread(filename,0)
        h1=localbinarypattern(img1)
        prediction=model.predict([h1])[0]
        if(prediction == "Cubism"):
            count+=1
    print "Cubism Count=",count
    c=c+count

    count=0

    for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Impressionism/*.jpg'): #assuming gif
        img1=cv2.imread(filename,0)
        h1=localbinarypattern(img1)
        prediction=model.predict([h1])[0]
        if(prediction == "Impressionism"):
            count+=1
    print "Impressionism Count=",count
    c=c+count
    count=0

    for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Pop Art/*.jpg'): #assuming gif
        img1=cv2.imread(filename,0)
        h1=localbinarypattern(img1)
        prediction=model.predict([h1])[0]
        if(prediction == "Pop Art"):
            count+=1
    print "Pop Art Count=",count
    c=c+count
    count=0

    for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\Realism/*.jpg'): #assuming gif
        img1=cv2.imread(filename,0)
        h1=localbinarypattern(img1)
        prediction=model.predict([h1])[0]
        if(prediction == "Realism"):
            count+=1
    print "Realism Count=",count
    c=c+count
    print c
    x.append(i)
    y.append(c)
plt.plot(x,y)
    #cv2.putText(img1, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
    #cv2.imshow("Image", img1)
    #cv2.waitKey(0)
     
#print training_data
#print responses

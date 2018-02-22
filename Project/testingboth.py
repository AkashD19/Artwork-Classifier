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

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

training_data, responses = eval(open('data11.txt', 'r').read())
training_data = np.array(training_data, np.float32)
responses = np.array(responses)

#model=LinearSVC(C=1,random_state=150)
#print "C=25,random_state=25 "
#model=KNeighborsClassifier(n_neighbors=250)
#print "n_neighbors=200"
#model=RandomForestClassifier(random_state=75)
#print "random_state=75"
#model = GaussianNB()
#model = AdaBoostClassifier(n_estimators=125)
#print "AdaBoostClassifier(n_estimators=125)"
model = LogisticRegression(C=10)
print "LogisticRegression(C=10)"
model.fit(training_data,responses)

count=0
c=0
test_data = eval(open('test_cubism.txt', 'r').read())
test_data=np.array(test_data,np.float32)

for i in range(0,100):
    prediction=model.predict([test_data[i]])[0]
    if(prediction == "Cubism"):
        count+=1
c=c+count
print "Cubism Count=",count

count=0
test_data = eval(open('test_impressionism.txt', 'r').read())
test_data=np.array(test_data,np.float32)

for i in range(0,100):
    prediction=model.predict([test_data[i]])[0]
    if(prediction == "Impressionism"):
        count+=1
print "Impressionism Count=",count
c=c+count
count=0
test_data = eval(open('test_popart.txt', 'r').read())
test_data=np.array(test_data,np.float32)

for i in range(0,100):
    prediction=model.predict([test_data[i]])[0]
    if(prediction == "Pop Art"):
        count+=1
print "Pop Art Count=",count
c=c+count
count=0
test_data = eval(open('test_realism.txt', 'r').read())
test_data=np.array(test_data,np.float32)

for i in range(0,100):
    prediction=model.predict([test_data[i]])[0]
    if(prediction == "Realism"):
        count+=1
print "Realism Count=",count
c=c+count
print c    



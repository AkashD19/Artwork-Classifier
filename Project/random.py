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
import random
import shutil 

im_list1=[]
for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Train\Pop Art/*.jpg'): #assuming gif
    im_list1.append(filename)
    
ran=[]
for i in range(0,100):
    r=random.randint(1,1100)
    if(r in  ran):
        continue
    else:
        ran.append(r)

for i in range(0,109):
   shutil.copy2(im_list1[ran[i]],'C:\Users\dutta\Desktop\Project\Images\Test\Pop Art')
   os.remove(im_list1[ran[i]])
    
    
    
import cv2
import glob
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.ensemble import AdaBoostClassifier

training_data, responses = eval(open('data.txt', 'r').read())
training_data = np.array(training_data, np.float32)
responses = np.array(responses)

def localbinarypattern(image):
     lbp=local_binary_pattern(image,24,8,method="uniform")
     (hist,_)=np.histogram(lbp.ravel(),bins=np.arange(0,27),range=(0,26))
     hist=hist.astype("float")
     hist/=(hist.sum()+(1e-7))
     return hist

actual = ["Cubism","Impressionism","Pop Art","Pop Art","Impressionism","Cubism","Cubism","Cubism","Impressionism","Pop Art","Pop Art","Impressionism","Realism","Realism","Realism","Realism"]
model = AdaBoostClassifier(n_estimators=125)
model.fit(training_data,responses)
i=0
for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\T/*.jpg'): #assuming gif
    image=cv2.imread(filename,0)
    img=cv2.imread(filename)
    h1=localbinarypattern(image)
    prediction=model.predict([h1])[0]
    cv2.putText(img, actual[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0), 3)
    cv2.imshow("Actual", img)
    
    cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0), 3)
    cv2.imshow("Prediction", image)
    
    i=i+1
    cv2.waitKey(0)
cv2.destroyAllWindows()
    
	 
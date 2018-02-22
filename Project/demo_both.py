import cv2
import glob
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

training_data, responses = eval(open('data11.txt', 'r').read())
training_data = np.array(training_data, np.float32)
responses = np.array(responses)


actual = ["Cubism","Impressionism","Pop Art","Pop Art","Impressionism","Cubism","Cubism","Cubism","Impressionism","Pop Art","Pop Art","Impressionism","Realism","Realism","Realism","Realism"]
model = AdaBoostClassifier(n_estimators=125)
model.fit(training_data,responses)

count=0
i=0
test_data = eval(open('test.txt', 'r').read())
test_data=np.array(test_data,np.float32)
for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\T/*.jpg'): #assuming gif
    image=cv2.imread(filename)
    img=cv2.imread(filename)
    prediction=model.predict([test_data[i]])[0]
    cv2.putText(img, actual[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0), 3)
    cv2.imshow("Actual", img)
    
    cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0), 3)
    cv2.imshow("Prediction", image)
    
    i=i+1
    cv2.waitKey(0)
cv2.destroyAllWindows()

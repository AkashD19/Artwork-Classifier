import cv2
import glob
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.ensemble import AdaBoostClassifier

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


actual = ["Cubism","Impressionism","Pop Art","Pop Art","Impressionism","Cubism","Cubism","Cubism","Impressionism","Pop Art","Pop Art","Impressionism","Realism","Realism","Realism","Realism"]
model = AdaBoostClassifier(n_estimators=125)
model.fit(training_data,responses)
i=0
for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\T/*.jpg'): #assuming gif
    image=cv2.imread(filename)
    img=cv2.imread(filename)
    h1=compute(image)
    prediction=model.predict([h1])[0]
    cv2.putText(img, actual[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0), 3)
    cv2.imshow("Actual", img)
    
    cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0), 3)
    cv2.imshow("Prediction", image)
    
    i=i+1
    cv2.waitKey(0)
cv2.destroyAllWindows()
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
model = AdaBoostClassifier(n_estimators=200)
model.fit(training_data,responses)
i=0
image_lbp=[]
image_csd=[]
data=[]
labels=[]

for filename in glob.glob('C:\Users\dutta\Desktop\Project\Images\Test\T/*.jpg'): #assuming gif
    image=cv2.imread(filename,0)
    image_lbp.append(image)
    img=cv2.imread(filename)
    image_csd.append(img)

for img1,img2 in zip(image_lbp,image_csd):
    h=localbinarypattern(img1)
    g=compute(img2)
    labels.append(actual[i])
    data.append(h)
    data.append(g)
    i=i+1
    
f = open("test.txt", "w") 
f.write(str([data, labels]))
f.close()
    
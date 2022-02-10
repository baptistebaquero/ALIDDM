import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('/Users/luciacev-admin/Desktop/Screen Shot 2022-02-09 at 7.24.19 PM.png')

kernel= np.ones((10,10),np.uint8)
kernel1= np.ones((20,20),np.uint8)
kernel2= np.ones((50,50),np.uint8)

closing1 = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel1)
closing2 = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel2)

erosion = cv.erode(closing1,kernel,iterations = 1)

f, axarr = plt.subplots(3)
axarr[0].imshow(image)
axarr[1].imshow(erosion)
axarr[2].imshow(closing1)

plt.show() 
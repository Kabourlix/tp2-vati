import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color


img = cv2.imread(r'D:\UQAC hiver 2023\8INF804-Vision artificielle\TP2_VisuArt\PreTraitement\Echantillion1Mod2_302.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit = 53.0)
gray_img_clahe = clahe.apply(img_gray)
# histogram calculation of the Clahe image
hist_gray_clahe = cv2.calcHist([gray_img_clahe],[0],None,[256],[0,256])
# convert the gray image with clahe into BGR image
img_clahe = cv2.cvtColor(gray_img_clahe,cv2.COLOR_GRAY2BGR)

smoothed = cv2.bilateralFilter(gray_img_clahe,5,25,25)
ret,thresh = cv2.threshold(smoothed, 10,255,cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
# noise removal
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 10)

# finding the sure background
sure_bg = cv2.dilate(opening,kernel,iterations=6)

# finding the sure foreground using distance transform
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

# defining the sure foreground applying a threshold at * percent of the maximum of distance transform
ret2, sure_fg = cv2.threshold(dist_transform, 0.04*dist_transform.max(),255,0)

# convert the sure foreground to uint8 type
sure_fg =np.uint8(sure_fg)

# find the unknown pixels
unknown = cv2.subtract(sure_bg,sure_fg)

# define the markers
_, markers = cv2.connectedComponents(sure_fg)

# separate the background(0=black) and the markers(0)
markers = markers +1

#if the unknown area has a value of 255, then mark it as 0
markers[unknown ==255] = 0

# applying watershed segmentation
markers = cv2.watershed(img, markers)

# Visualize the watershed: replace the pixels in the img where the markers = -1 into yellow
img[markers == -1] = (255,255,0)

# assign it to a different image
img2 = color.label2rgb(markers, bg_label=0)

plt.figure(figsize=(16,16))
plt.subplot(331)
plt.imshow(img, cmap='gray')
plt.title('Segmented image')
plt.subplot(332)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale image')
plt.subplot(333)
plt.imshow(img_clahe)
plt.title('CLAHE image')
plt.subplot(334)
plt.imshow(thresh, cmap='gray')
plt.title('Masked image')
plt.subplot(335)
plt.imshow(sure_bg, cmap='gray')
plt.title('Sure background')
plt.subplot(336)
plt.imshow(sure_fg, cmap='gray')
plt.title('Sure foreground')
plt.subplot(337)
plt.imshow(unknown, cmap='gray')
plt.title('Unknown area: difference between BG and FG')
plt.subplot(338)
plt.imshow(markers, cmap='gray')
plt.title('markers')
plt.subplot(339)
plt.imshow(img2, cmap='gray')
plt.title('Colored stones')

plt.show()

from __future__ import print_function

import PIL
import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage import io, color
from skimage.color import label2rgb
from skimage.data import astronaut
from skimage.segmentation import felzenszwalb, slic, quickshift, clear_border
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

# image original
loaded_img = io.imread('C:/Users/annie/PycharmProjects/pythonProject/Echantillion1Mod2_301.png')

# grayscale
image = loaded_img.copy()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
(T, threshInv) = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask_image = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
res = cv2.bitwise_and(image, image, mask=mask_image)

# Afficher rapidement les images
plt.figure()
plt.subplot(231)
plt.imshow(loaded_img)
plt.title('Original')
plt.subplot(232)
plt.imshow(gray_image)
plt.title('Grayscale')
plt.subplot(233)
plt.imshow(blurred)
plt.title('Blur')
plt.subplot(234)
plt.imshow(threshInv)
plt.title('Threshold')
plt.subplot(235)
plt.imshow(mask_image)
plt.title('Morph Close (MASK)')
plt.subplot(236)
plt.imshow(res)
plt.title('Image with mask')

# TODO : pour le rapport : montrer les résultats sans le mask

# convertir en float pr utiliser skimage
img = img_as_float(loaded_img)

# felzenszwalb
segments_fz = felzenszwalb(res, scale=1, sigma=0.5, min_size=600)
# TODO : image avec mask utilisé pour ne pas segmenter le fond noir
# TODO : diminution scale pour la taille des segmentations
# TODO : augmentation min size pour ne pas prendre en compte des petits points de couleur au sein d'une pierre

# slic
segments_slic = slic(img, n_segments=250, compactness=10, sigma=1, mask=mask_image)
# TODO : mask ajouté pour enlever les segmentation du fond noir

# quickShift
segments_quick = quickshift(img, kernel_size=5, max_dist=100, ratio=0.5)
# TODO : Pas de param mask, on ne peut pas utiliser l'image masqué non plus....fond noir segmenté !
# TODO : max_dist augmenté pour siminué le nombre de cluster

print("Felzenszwalb's number of segments: %d" % len(np.unique(segments_fz)))
print("Slic number of segments: %d" % len(np.unique(segments_slic)))
print("Quickshift number of segments: %d" % len(np.unique(segments_quick)))

# Afficher les trois segmentations
plt.figure()
plt.subplot(231)
plt.imshow(mark_boundaries(img, felzenszwalb(img, scale=100, sigma=0.5, min_size=100)))
plt.title("Felzenszwalbs's method 1")
plt.subplot(232)
plt.imshow(mark_boundaries(img, slic(img, n_segments=250, compactness=10, sigma=1)))
plt.title("SLIC 1")
plt.subplot(233)
plt.imshow(mark_boundaries(img, quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)))
plt.title("Quickshift 1")
plt.subplot(234)
plt.imshow(mark_boundaries(img, segments_fz))
plt.title("Felzenszwalbs's method 2")
plt.subplot(235)
plt.imshow(mark_boundaries(img, segments_slic))
plt.title("SLIC 2")
plt.subplot(236)
plt.imshow(mark_boundaries(img, segments_quick))
plt.title("Quickshift 2")

# Segmentation en plusieurs fichiers
# TODO : On utilisera seulement ceux de SLIC car c'est le meilleur

plt.figure()
plt.imshow(label2rgb(segments_slic,
                     image,
                     kind = 'avg'))
plt.title('Segment label')
plt.show()


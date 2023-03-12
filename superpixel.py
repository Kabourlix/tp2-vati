# imports
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

# Charger l'image
def load_img(path):
    return io.imread(path)


# Mini traitement de l'image
def process_img(img):
    image = img.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    (T, threshInv) = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_image = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
    res = cv2.bitwise_and(image, image, mask=mask_image)

    return [gray_image,blurred,threshInv,mask_image,res]

# Appliquer la segmentation par la méthode Felzenszwalb
# res : image de base avec mask appliqué
def seg_felzenszwalb(res):
    return felzenszwalb(res, scale=1, sigma=0.5, min_size=600)
    # TODO : image avec mask utilisé pour ne pas segmenter le fond noir
    # TODO : diminution scale pour la taille des segmentations
    # TODO : augmentation min size pour ne pas prendre en compte des petits points de couleur au sein d'une pierre

# Appliquer la segmentation par la méthode SLIC
# loaded_img : image de base
# mask_image : mask de l'image ( image binaire )
def seg_slic(loaded_img, mask_image) :
    # convertir en float pr utiliser skimage
    img = img_as_float(loaded_img)
    return slic(img, n_segments=250, compactness=10, sigma=1, mask=mask_image)
    # TODO : mask ajouté pour enlever les segmentation du fond noir

# Appliquer la segmentation par la méthode QuickShift
# loaded_img : image de base
def seg_quickshift(loaded_img) :
    # convertir en float pr utiliser skimage
    img = img_as_float(loaded_img)
    return quickshift(img, kernel_size=5, max_dist=100, ratio=0.5)
    # TODO : Pas de param mask, on ne peut pas utiliser l'image masqué non plus....fond noir segmenté !
    # TODO : max_dist augmenté pour siminué le nombre de cluster


'''MAIN SCRIPT'''

# image original
loaded_img = load_img('C:/Users/annie/PycharmProjects/pythonProject/Echantillion1Mod2_301.png')

# traitement image
gray_image,blurred,threshInv,mask_image,res = process_img(loaded_img)

# Affichage rapide de la procédure de traitement image
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

# Appeler les méthodes de segmentation
segments_fz = seg_felzenszwalb(res)
segments_slic = seg_slic(loaded_img, mask_image)
segments_quick = seg_quickshift(loaded_img)

print("Felzenszwalb's number of segments: %d" % len(np.unique(segments_fz)))
print("Slic number of segments: %d" % len(np.unique(segments_slic)))
print("Quickshift number of segments: %d" % len(np.unique(segments_quick)))

# convertir en float pr utiliser skimage
img = img_as_float(loaded_img)
# Afficher les segmentations ( 1 - avant traitement image et edit param, 2 - après traitement image et edit param )
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
                     loaded_img,
                     kind='avg'))
plt.title('Segment label (SLIC)')

plt.show()


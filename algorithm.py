import cv2
import numpy as np
import Utility as utils
import algo as algo
import os
import pandas as pd

def pretraitement(image):
    """
    This method apply a pretraitement to the image.
    :param image: Original image (np.array)
    :return: The pretraited image (np.array)
    """
    image_contrast = algo.adjust_contrast(image, 1.5)
    image_equ = algo.equalize_histogram(image_contrast)
    return image_equ

def segmentation(image):
    #TODO
    return np.array([image])


def feature_extraction(image):
    """
    Thie method extract the mean color of the segmented image.
    :param image: A segmented image (np.array)
    :return:
    """
    return None

if __name__ == "__main__":
    path = utils.get_path()
    image = utils.load_img(path)

    #Pretraitement
    treated_image = pretraitement(image)

    # Save the image
    #cv2.imwrite(path, treated_image)

    #Segmentation : saved in an array
    segmented_images = segmentation(treated_image)

    #Feature extraction and display
    for i in range(len(segmented_images)):
        feature = feature_extraction(segmented_images[i])
        #TODO : display feature
import cv2
import numpy as np
import Utility as utils


def pretraitement(image):
    """
    This method apply a pretraitement to the image.
    :param image: Original image (np.array)
    :return: The pretraited image (np.array)
    """
    #TODO
    return image

def segmentation(image):
    #TODO
    return np.array([image])


def feature_extraction(image):
    """
    Thie method extract the mean color of the segmented image.
    :param image: A segmented image (np.array)
    :return:
    """
    #TODO
    return None

if __name__ == "__main__":
    path = utils.get_path()
    image = utils.load_img(path)

    #Pretraitement
    treated_image = pretraitement(image)

    #Segmentation : saved in an array
    segmented_images = segmentation(treated_image)

    #Feature extraction and display
    for i in range(len(segmented_images)):
        feature = feature_extraction(segmented_images[i])
        #TODO : display feature
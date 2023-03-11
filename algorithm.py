import cv2
import numpy as np
import Utility as utils


def pretraitement(image):
    #TODO
    return image

def segmentation(image):
    #TODO
    return np.array([image])

def feature_extraction(image):
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
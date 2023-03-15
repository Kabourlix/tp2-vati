import cv2
import numpy as np
import Utility as utils
import algo
import pandas as pd
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.color import label2rgb
import matplotlib.pyplot as plt


def contrast_enhance(img):
    # converting to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    return enhanced_img


def pretraitement(image):
    """
    This method apply a pretraitement to the image.
    :param image: Original image (np.array)
    :return: The pretraited image (np.array)
    """
    # image_contrast = algo.adjust_contrast(image, 1.5)
    image_contrast = contrast_enhance(image)
    # image_equ = algo.equalize_histogram(image_contrast)
    return image_contrast


def segmentation(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Elevations
    elevation_map = sobel(grey)

    # Markers
    markers = np.zeros_like(grey)
    markers[grey < 10] = 1
    markers[grey > 150] = 2

    # Watershed
    segmentation = watershed(elevation_map, markers)

    # Segmented images
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    show_segmentation(image, segmentation)
    segmentation = segmentation.astype(np.uint8)

    contours, _ = cv2.findContours(segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print("Number of contours: " + str(len(contours)))

    segmented_images = []
    bounding_boxes = np.zeros((len(contours), 4))
    i = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes[i] = [x, y, w, h]
        segmented_images.append([str(i), image[y:y + h, x:x + w]])
        i += 1

    areas = bounding_boxes[:, 2] * bounding_boxes[:, 3]

    # Clear the least significant bounding boxes : the ones with an area < 2500
    bounding_boxes = bounding_boxes[areas > 2500]
    cropped = [segmented_images[i] for i in range(len(segmented_images)) if areas[i] > 2500]

    result = []
    # Add a label to each segmented image
    for i in range(len(cropped)):
        result.append(["seg " + str(i), cropped[i][1]])

    return np.array(result)


def show_segmentation(orig_img, segm):
    labeled_coins, _ = ndi.label(segm)
    image_label_overlay = label2rgb(labeled_coins, image=orig_img)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
    ax1.imshow(orig_img, cmap=plt.cm.gray, interpolation='nearest')
    ax1.contour(segm, [0.5], linewidths=1.2, colors='y')
    ax1.axis('off')
    ax1.set_adjustable('box')
    ax2.imshow(image_label_overlay, interpolation='nearest')
    ax2.axis('off')
    ax2.set_adjustable('box')
    margins = dict(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    plt.subplots_adjust(**margins)
    plt.show()


def feature_extraction(image, name):
    """
    Thie method extract the mean color of the segmented image.
    :param image: A segmented image (np.array)
    :return:
    """
    return algo.get_mean_color(image, name)


if __name__ == "__main__":
    path = utils.get_path()
    image = utils.load_img(path)

    # Pretraitement
    treated_image = pretraitement(image)

    # Save the image
    # cv2.imwrite(path, treated_image)

    # Segmentation : saved in an array
    segmented_images = segmentation(treated_image)

    # Feature extraction and display
    df = pd.DataFrame(columns=["label", "mean R", "mean G", "mean B"])
    for original_img in segmented_images:
        label, segm_img = original_img[0], original_img[1]
        feature = feature_extraction(segm_img, label)
        # Create a series from the feature
        featureS = pd.Series(feature, index=df.columns)
        # Concat the feature to the dataframe
        df = df.append(featureS, ignore_index=True)

    print(df)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def adjust_contrast(img, factor):
    """
    This function adjusts the contrast of an image.
    """
    image_c = np.zeros(img.shape, img.dtype)
    alpha = factor
    beta = 128 * (1 - alpha)
    cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta, image_c)
    return image_c

def equalize_histogram(img):
    """
    This function equalizes the histogram of an image.
    """
    image_e = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(image_e)
    hist = cv2.calcHist([channels[0]], [0], None, [256], [0, 256])
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, image_e)
    image_e = cv2.cvtColor(image_e, cv2.COLOR_YCrCb2BGR)
    hist_e = cv2.calcHist([channels[0]], [0], None, [256], [0, 256])

    # Plot the histogram
    plt.plot(hist_e)
    plt.title('Histogram after treatment')
    plt.xlim([0, 256])
    plt.ylim([0, 10000])
    plt.show()

    return image_e


def get_mean_color(img, img_name):
    """
    This function get the mean color of an image.
    """
    total_r, total_g, total_b = 0, 0, 0
    num_pixels = 0
    height, width, _ = img.shape
    
    # Calculate total color values and number of pixels
    total_r += np.sum(img[:,:,2])
    total_g += np.sum(img[:,:,1])
    total_b += np.sum(img[:,:,0])
    num_pixels += height * width
        
    # Calculate mean color
    mean_r = total_r / num_pixels
    mean_g = total_g / num_pixels
    mean_b = total_b / num_pixels

    return (img_name, mean_r, mean_g, mean_b)

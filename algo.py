import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def adjust_contrast(img, factor):
    """
    This function adjusts the contrast of an image.
    """
    image_c = np.zeros(img.shape, img.dtype)
    alpha = factor
    beta = 128 * (1 - alpha)
    cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta, image_c)
    return image_c

def adjust_brightness(img, factor):
    """
    This function adjusts the brightness of an image.
    """
    image_b = np.zeros(img.shape, img.dtype)
    alpha = factor
    beta = 128 * (1 - alpha)
    cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta, image_b)
    return image_b

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
    return hist


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


"""
Main script
"""
# Set the input and output folders
# -- TODO : change the path to your own path, create the output folder if it does not exist
input_folder = r"C:\Users\ludivine.ribet\TP2\Images"
output_folder = r"C:\Users\ludivine.ribet\TP2\PreTraitement"

hist_list = []
image_means = []

# Get a list of all the image filenames
filenames = [filename for filename in os.listdir(input_folder) if filename.endswith(".png")]

# Create a grid of plots with one row and as many columns as there are images
num_images = len(filenames)
num_cols = min(4, num_images)  # Limit the number of columns to 4 for clarity
num_rows = (num_images - 1) // num_cols + 1
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
axes = axes.flatten()

# Loop through all images in the input folder
for i, filename in enumerate(filenames):
    # Read the image
    img = cv2.imread(os.path.join(input_folder, filename))

    # Adjust the brightness
    brightness_factor = 1
    img_b = adjust_brightness(img, brightness_factor)

    # Adjust the contrast
    contrast_factor = 1.5
    img_c = adjust_contrast(img_b, contrast_factor)

    # Equalize the histogram and get the histogram values
    hist = equalize_histogram(img_c)

    # Plot the histogram for the current image
    ax = axes[i]
    ax.plot(hist)
    ax.set_title('Histogram for Image {}'.format(i+1))
    ax.set_xlim([0, 256])
    ax.set_ylim([0, 10000])

    # Get the mean color
    mean_color = get_mean_color(img_c, filename)
    image_means.append(mean_color)

    # Save the image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img_c)

# Hide any unused subplots
for j in range(i+1, num_rows*num_cols):
    axes[j].axis('off')

# Show the plot
plt.tight_layout()
plt.show()

# Print the table of mean colors
df = pd.DataFrame(image_means, columns=["Nom de l'image", 'Moyenne de R:', 'Moyenne de G:', 'Moyenne de B:'])
print(df)
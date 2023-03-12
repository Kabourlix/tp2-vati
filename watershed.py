import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color


img = cv2.imread(r'D:\UQAC hiver 2023\8INF804-Vision artificielle\TP2_VisuArt\PreTraitement\Echantillion1Mod2_302.png')

def watershed(img,clipLimit,opening_iterations,dilate_iterations, dtf):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit = clipLimit)
    gray_img_clahe = clahe.apply(img_gray)

    # convert the gray image with clahe into BGR image
    img_clahe = cv2.cvtColor(gray_img_clahe,cv2.COLOR_GRAY2BGR)

    ret,thresh = cv2.threshold(gray_img_clahe, 10,255,cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    # noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = opening_iterations)

    # finding the sure background
    sure_bg = cv2.dilate(opening,kernel,iterations=dilate_iterations)

    # finding the sure foreground using distance transform
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)

    # defining the sure foreground applying a threshold at (dtf) percent of the maximum of distance transform
    ret2, sure_fg = cv2.threshold(dist_transform, dtf*dist_transform.max(),255,0)

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
    img[markers == -1] = (0,255,0)

    # assign it to a different image
    img2 = color.label2rgb(markers, bg_label=0)
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    # show the images
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
    plt.imshow(markers)
    plt.title('markers')
    plt.subplot(339)
    plt.imshow(img2, cmap='gray')
    plt.title('Colored stones')

    plt.show()
    return img


def segmentation(img):
    # we perform successive watershed segmentations to have better result
    image_1= watershed(img,
                       clipLimit=30,
                       opening_iterations=5,
                       dilate_iterations=7,
                       dtf=0.1)
    image_2= watershed(image_1,
                       clipLimit=30,
                       opening_iterations=7,
                       dilate_iterations=8,
                       dtf=0.35)
    image_seg= watershed(image_2,
                       clipLimit=30,
                       opening_iterations=7,
                       dilate_iterations=8,
                       dtf=0.50)
    return image_seg

#segmentation(img)
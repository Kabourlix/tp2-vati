import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import platform

from enum import Enum


# class Room(Enum):
#     Unknown = 0
#     Kitchen = 1
#     Chamber = 2
#     LivingRoom = 3


def load_img(path):
    """
    Open the image and return the image
    :param path: The relative path of the image
    :return: The image to be used
    """
    if platform.system() == "Windows":
        path = path.replace("/", "\\")
    elif platform.system() == "Darwin":
        path = path.replace("\\", "/")

    try:
        if not (os.path.isfile(path)):
            raise FileNotFoundError("Image not found")
        img = cv2.imread(path)
    except FileNotFoundError as e:
        print(e)
        exit()
    finally:
        print("The path exists and the image is stored in img with shape {}".format(img.shape))
    return img


# def find_room(path):
#     """
#     Get the room name from the path : a simpler version than get_room
#     :param path: The relative path of the image
#     :return: The room via enum
#     """
#
#     if platform.system() == "Darwin":
#         path = path.replace("\\", "/")
#
#     room_name = path.split("/")[-2]
#     print("The room is {}".format(room_name))
#
#     return room_name
#
#
# def get_room_via_path(path):
#     """
#     Get the room name from the path
#     :param path: The relative path of the image
#     :return: The room via enum
#     """
#     if platform.system() == "Windows":
#         path = path.replace("/", "\\")
#     elif platform.system() == "Darwin":
#         path = path.replace("\\", "/")
#     room = Room.Unknown
#     room_name = "Unknown"
#     try:
#         if not (os.path.isdir(path)):
#             raise FileNotFoundError("Folder not found")
#         room_name = path.split("/")[-1]
#         if room_name == "Chamber":
#             room = Room.Chamber
#         elif room_name == "Kitchen":
#             room = Room.Kitchen
#         elif room_name == "LivingRoom":
#             room = Room.LivingRoom
#     except FileNotFoundError as e:
#         print(e)
#         exit()
#     finally:
#         print("The path exists and the room is {}".format(room_name))
#     return room


def get_path():
    """
    Handle the command line arguments
    :return: The path of the image
    """
    import argparse
    # Parse to get two paths of images
    # cmd shall be : python test.py -i <path1> -r <path2>
    parser = argparse.ArgumentParser(description='Get img and a ref to detect objects')
    parser.add_argument('-i', '--img', help='The path of the image to analyze', required=True)
    args = vars(parser.parse_args())
    return args['img']


def is_saved(save_arg):
    if save_arg is None:
        return False
    if save_arg.lower() == "y":
        return True
    else:
        return False


def get_room(letter):
    letter = letter.upper()
    if letter == "C":
        return "Chambre"
    elif letter == "K":
        return "Cuisine"
    elif letter == "S":
        return "Salon"
    else:
        return ValueError("The letter is not valid")


def quick_plot(img, title="", figsize=None, cmap="Greys", binary=True):
    """
    Plot the image
    :param binary: True if the image is binary so black and white
    :param figsize: The size of the figure
    :param title: the title of the image
    :param cmap: The color map of the image
    :param img: The image to be plotted
    """
    plt.figure()
    # Trasncript image from bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap=cmap, interpolation=None if binary else 'nearest')
    plt.title(title)
    print("The image is plotted")
    plt.show()


def quick_plot2(img1, img2, title1="", title2="", figsize=None, cmap="Greys", binary=True):
    """
    Plot the image
    :param cmap:
    :param figsize:
    :param title2:
    :param title1:
    :param img2:
    :param img1:
    :param img: The image to be plotted
    """
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1, cmap=cmap, interpolation=None if binary else 'nearest')
    plt.title(title1)
    plt.subplot(122)
    plt.imshow(img2, cmap=cmap, interpolation=None if binary else 'nearest')
    plt.title(title2)
    plt.show()


def quick_plot_any(imgs, titles, dim, figsize=None, cmap="Greys", binary=True):
    """
    Plot images according to the dimension
    :param binary:
    :param cmap:
    :param figsize: The size of the figure
    :param imgs: The images to be plotted
    :param titles: The titles of the images
    :param dim: The dimension of the plot
    :return: None
    """
    if len(imgs) != len(titles):
        raise ValueError("The length of imgs and titles are not equal")
    if len(imgs) > dim[0] * dim[1]:
        raise ValueError("The length of imgs > dim are not equal")
    plt.figure(figsize=figsize)
    for i in range(len(imgs)):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(imgs[i], cmap=cmap, interpolation=None if binary else 'nearest')
        plt.title(titles[i])
    plt.show()


# def get_subtracted_threshold_naive(img, ref, threshold, t_param=cv2.THRESH_BINARY):
#     """
#     Get the thresholded image
#     :param img: The image to be thresholded
#     :param ref: The reference image
#     :param threshold: The threshold
#     :return: The thresholded image
#     """
#     diff = cv2.absdiff(img, ref)
#     _, thresholded = cv2.threshold(diff, threshold, 255, t_param)
#     return thresholded
#
#
# def get_subtracted_threshold(img, ref, threshold):
#     """
#     Get the thresholded image using "multi channel thresholding"
#     :param img: The image to be thresholded
#     :param ref: The reference image
#     :param threshold: The threshold
#     :return: The thresholded image (foreground mask)
#     """
#     diff = cv2.absdiff(img, ref)
#     return ((diff ** 2).sum(axis=2) > threshold).astype(np.uint8) * 255
#
#
# def draw_contours(image, color=(0, 255, 0), thickness=2):
#     """
#     Draw the contours of the image (no alteration)
#     :param thickness: The thickness of the contours
#     :param color: The color of the contours
#     :param image: The image to draw the contours from
#     :return: The image with contours and the contours
#     """
#     img = image.copy()
#     contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(img, contours, -1, color, thickness)
#     return img, contours
#
#
# def draw_contours_in_place(image, color=(0, 255, 0), thickness=2):
#     """
#     Draw the contours of the image (alteration)
#     :param thickness: The thickness of the contours
#     :param color: The color of the contours
#     :param image: The image to draw the contours on
#     :return: the contours
#     """
#     contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for contour in contours:
#         cv2.drawContours(image, [contour], -1, color, thickness)
#     return contours
#
#
# def draw_bounding_boxes(image, contours, color=(0, 255, 0), thickness=2):
#     """
#     Draw the bounding boxes of the image (no alteration)
#     :param thickness: The thickness of the bounding box
#     :param color: The color of the bounding box
#     :param contours: Contours of the image
#     :param image: The image to draw the bounding boxes from
#     :return: The image with bounding boxes
#     """
#     img = image.copy()
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
#     return img


def getMask(image, room):
    mask0 = np.zeros(image.shape[:2], np.uint8)

    if room == "C":
        # Ci dessous est le masque temporaire de la chambre
        pointsMask = np.array(
            [(2376, 3992), (2048, 3980), (2145, 3778), (2232, 3560), (2298, 3314), (2304, 3112), (2195, 2981),
             (2007, 2858), (1786, 2692), (1617, 2557), (1489, 2446), (1370, 2363), (1171, 2430), (1008, 2505),
             (852, 2585), (730, 2688), (571, 2771), (503, 2827), (456, 2696), (412, 2573), (347, 2458), (303, 2323),
             (268, 2208), (209, 2093), (165, 1982), (131, 1839), (84, 1744), (56, 1653), (9, 1550), (16, 1439),
             (3, 1265), (125, 1245), (228, 1221), (306, 1308), (387, 1411), (549, 1403), (643, 1384), (687, 1328),
             (777, 1332), (877, 1356), (912, 1423), (999, 1439), (1099, 1455), (1227, 1502), (1345, 1495), (1408, 1514),
             (1483, 1522), (1536, 1491), (1561, 1395), (1517, 1340), (1608, 1284), (1680, 1340), (1673, 1459),
             (1708, 1510), (1783, 1550), (1857, 1570), (2023, 1491), (2170, 1415), (2291, 1368), (2394, 1324),
             (2501, 1273), (2597, 1233), (2675, 1177), (2672, 1086), (2604, 1023), (2566, 944), (2504, 888),
             (2426, 884), (2373, 821), (2351, 781), (2338, 706), (2310, 603), (2379, 551), (2507, 547), (2560, 476),
             (2532, 373), (2588, 341), (2635, 440), (2647, 503), (2713, 515), (2735, 464), (2694, 369), (2750, 337),
             (2844, 357), (2909, 369), (2975, 373), (2988, 440), (2997, 535), (2994, 622), (2994, 694), (3000, 781),
             (3100, 789), (3184, 797), (3340, 844), (3471, 872), (3556, 900), (3796, 971), (4005, 1039), (4127, 1090),
             (4217, 1114), (4324, 1146), (4458, 1189), (4558, 1229), (4723, 1280), (4832, 1316), (4973, 1364),
             (5051, 1403), (5145, 1475), (5179, 1491), (5170, 1538), (5151, 1610), (5126, 1669), (5060, 1657),
             (4957, 1598), (4867, 1558), (4770, 1570), (4767, 1653), (4870, 1705), (4979, 1736), (5123, 1796),
             (5210, 1835), (5344, 1919), (5516, 2018), (5613, 2046), (5632, 2224), (5591, 2359), (5541, 2525),
             (5482, 2656), (5435, 2803), (5373, 2957), (5304, 3148), (5270, 3278), (5226, 3449), (5182, 3619),
             (5088, 3762), (5017, 3897), (4964, 3988), (4779, 3992), (4536, 3960), (4246, 3984), (3840, 3960),
             (3503, 3980), (3250, 3980), (2919, 3972), (2591, 3976)])
    elif (room == "K"):
        # Ci dessous est le masque temporaire de la cuisine
        pointsMask = np.array(
            [(796, 3988), (815, 3941), (940, 3921), (1046, 3806), (1083, 3750), (1111, 3627), (1155, 3469),
             (1199, 3362), (1246, 3199), (1308, 3060), (1345, 2957), (1383, 2886), (1414, 2775), (1436, 2688),
             (1436, 2620), (1470, 2585), (1473, 2517), (1545, 2474), (1655, 2474), (1708, 2402), (1733, 2339),
             (1807, 2339), (1870, 2272), (1895, 2165), (1907, 2050), (1917, 1986), (2035, 1966), (2157, 1970),
             (2270, 1978), (2363, 1982), (2441, 1950), (2504, 1927), (2622, 1927), (2697, 1919), (2794, 1919),
             (2906, 1927), (2997, 1931), (3128, 1943), (3303, 1943), (3368, 1939), (3465, 1939), (3612, 1923),
             (3677, 1915), (3743, 2018), (3818, 2081), (3896, 2165), (3955, 2252), (4040, 2383), (4089, 2482),
             (4130, 2541), (4139, 2632), (4130, 2712), (4102, 2795), (4093, 2882), (4089, 2969), (4133, 3072),
             (4193, 3160), (4274, 3251), (4336, 3366), (4402, 3449), (4464, 3540), (4539, 3671), (4639, 3659),
             (4795, 3639), (4876, 3635), (4973, 3635), (5029, 3635), (5185, 3635), (5273, 3623), (5326, 3608),
             (5376, 3600), (5401, 3524), (5426, 3461), (5463, 3508), (5554, 3508), (5632, 3497), (5763, 3469),
             (5856, 3465), (5956, 3449), (5966, 3580), (5956, 3699), (5959, 3814), (5966, 3980), (5769, 3992),
             (5547, 3992), (5282, 3988), (5113, 3992), (4911, 3984), (4701, 3996), (4464, 3984), (4230, 3992),
             (4015, 3992), (3724, 3992), (3490, 3988), (3290, 3988), (3062, 3996), (2825, 3996), (2622, 3988),
             (2354, 3980), (2073, 3988), (1854, 3984), (1645, 3976), (1389, 3984), (1239, 3984), (1077, 3984),
             (937, 3992)])
    elif (room == "S"):
        # Ci dessous est le masque temporaire du salon
        pointsMask = np.array(
            [(6, 3976), (6, 3853), (12, 3762), (12, 3584), (9, 3366), (6, 3195), (16, 2985), (62, 3092), (91, 3207),
             (190, 3231), (240, 3195), (240, 3104), (225, 3021), (203, 2906), (184, 2779), (159, 2680), (237, 2652),
             (300, 2533), (356, 2454), (403, 2553), (437, 2605), (534, 2601), (546, 2549), (499, 2474), (475, 2426),
             (553, 2406), (631, 2398), (702, 2398), (765, 2398), (821, 2379), (880, 2355), (933, 2343), (1043, 2339),
             (1139, 2331), (1274, 2307), (1364, 2260), (1461, 2236), (1542, 2236), (1630, 2228), (1698, 2196),
             (1776, 2172), (1829, 2172), (1904, 2188), (1951, 2176), (2026, 2236), (2070, 2287), (2113, 2355),
             (2154, 2414), (2201, 2454), (2188, 2498), (2179, 2533), (2213, 2597), (2263, 2628), (2316, 2660),
             (2363, 2692), (2451, 2664), (2516, 2648), (2579, 2628), (2591, 2569), (2597, 2533), (2663, 2517),
             (2769, 2505), (2869, 2482), (2925, 2470), (3006, 2450), (3144, 2418), (3237, 2394), (3312, 2375),
             (3400, 2355), (3478, 2339), (3549, 2307), (3596, 2283), (3615, 2208), (3615, 2165), (3687, 2145),
             (3765, 2113), (3843, 2097), (3958, 2065), (4043, 2050), (4136, 2038), (4205, 2018), (4314, 2006),
             (4395, 1994), (4452, 1970), (4511, 1998), (4595, 2006), (4670, 1982), (4708, 1919), (4761, 1891),
             (4817, 1871), (4867, 1891), (4923, 1915), (4951, 1943), (4945, 2018), (4886, 2038), (4879, 2113),
             (4807, 2121), (4733, 2165), (4676, 2220), (4623, 2264), (4617, 2323), (4667, 2375), (4729, 2402),
             (4783, 2446), (4845, 2486), (4935, 2533), (4979, 2549), (5014, 2612), (5014, 2676), (5067, 2712),
             (5117, 2731), (5160, 2712), (5223, 2727), (5294, 2759), (5351, 2807), (5429, 2834), (5532, 2874),
             (5594, 2902), (5650, 2934), (5725, 2969), (5822, 3025), (5869, 3049), (5944, 3092), (5941, 3187),
             (5950, 3275), (5950, 3382), (5953, 3477), (5969, 3576), (5956, 3655), (5959, 3730), (5966, 3865),
             (5950, 3992), (5800, 3988), (5663, 3988), (5529, 3988), (5385, 3984), (5232, 3992), (5023, 3996),
             (4811, 3984), (4517, 3980), (4233, 3984), (3977, 3988), (3696, 3984), (3406, 3988), (3190, 3988),
             (2872, 3980), (2629, 3984), (2441, 3988), (2270, 3988), (2067, 3972), (1839, 3968), (1676, 3980),
             (1436, 3972), (1252, 3976), (1049, 3984), (852, 3988), (581, 3976), (446, 3976), (262, 3976), (162, 3976)])
    else:
        print("Can't find the room. There is no mask")
        return mask0

    cv2.fillPoly(mask0, np.array([pointsMask]), 255)  # remplir l'interieur de ce masque par du blanc
    return mask0


def get_masked_img(image, mask0):
    newImg = image.copy()
    return cv2.bitwise_and(newImg, newImg, mask=mask0)


def hist_equalization(img):
    # convert the image being treated in grayscale
    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # histogram calculation of the original image
    hist_gray_original = cv2.calcHist([gray_original], [0], None, [256], [0, 256])
    # Create a matrix to be added to the image for saturation increase
    M = np.ones(img.shape, dtype="uint8") * 85
    # the matrix is added to the image
    added_img = cv2.add(img, M)
    # convert the resulting image to grayscale
    gray_image = cv2.cvtColor(added_img, cv2.COLOR_BGR2GRAY)
    # gray image histogram calculation
    hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # CLAHE applied to the gray_img
    clahe = cv2.createCLAHE(clipLimit=4.0)
    gray_img_clahe = clahe.apply(gray_image)
    # histogram calculation of the Clahe image
    hist_gray_clahe = cv2.calcHist([gray_img_clahe], [0], None, [256], [0, 256])
    # convert the gray image with clahe into BGR image
    img_clahe = cv2.cvtColor(gray_img_clahe, cv2.COLOR_GRAY2BGR)
    # img clahe becomes the new img
    img = img_clahe
    # show the histograms
    """plt.figure(figsize=(10,16))
    plt.subplot(321)
    plt.imshow(gray_original, cmap='gray')
    plt.title('Original gray image')
    plt.subplot(322)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist_gray_original, color='m')
    plt.title('Grayscale image histogram')
    plt.subplot(323)
    plt.imshow(gray_image, cmap='gray')
    plt.title('gray image saturee')
    plt.subplot(324)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist_gray, color='m')
    plt.title('Grayscale image histogram')
    plt.subplot(325)
    plt.imshow(gray_img_clahe, cmap='gray')
    plt.title('gray image saturee clahe')
    plt.subplot(326)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.title('Gray clahe histogram')
    plt.plot(hist_gray_clahe, color='m')
    plt.show()"""
    return img


def highlightItem(image, x1, y1, x2, y2, index):
    # Dessiner un carré vert
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    # Chercher taille du rectangle de text
    # Afficher texte
    cv2.rectangle(image, (x1 - 2, y1 - 70), (x2 + 6, y1), (0, 255, 0), -1)
    cv2.putText(image, 'Object num ' + str(index), (x1 + 20, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 2)


def draw_bounding_boxes_in_place(image, contours, color=(0, 255, 0), thickness=4, threshold=-1):
    """
    Draw the bounding boxes of the image
    :param threshold:
    :param thickness: The thickness of the bounding box
    :param image: The image to draw the bounding boxes
    :param contours: Contours of the image
    :param color: The color of the bounding box
    :return: None
    """
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    # sort the bounding boxes by area
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[2] * x[3], reverse=True)
    # #Plot an histogram of the bounding boxes area
    # areas = [box[2] * box[3] for box in bounding_boxes]
    # plt.hist(areas, bins=20)
    # plt.show()
    if threshold != -1:
        bounding_boxes = [box for box in bounding_boxes if box[2] * box[3] > threshold]

    for i, box in enumerate(bounding_boxes):
        x, y, w, h = box
        highlightItem(image, x, y, x + w, y + h, i)  # this is what is replaced by annie
        # cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness) # this is the original code

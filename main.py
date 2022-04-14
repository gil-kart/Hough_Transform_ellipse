
import Hough_Functions
#q1 script
## image 1
#path_image = r'ellipses/images.jpg'
#upbound = 60
#lowbound = 40
#cuber = [90, 90]
#a_b_normelize = [8, 1]
#gaussian = 1
#Canny = [748, 1268]
#number_of_points = 1000
#poind_rad = 7
#Hough_Functions.Run_Hough(path_image, upbound, lowbound, cuber,
#                          a_b_normelize, gaussian, Canny, number_of_points, poind_rad)
#
## image 2
#path_image = r'ellipses/Headline-Pic.jpg'
#upbound = 180
#lowbound = 10
#cuber = [400, 400]
#a_b_normelize = [3, 1]
#gaussian = 23
#Canny = [93, 133]
#number_of_points = 1700
#poind_rad = 11
#Hough_Functions.Run_Hough(path_image, upbound, lowbound, cuber,
#                          a_b_normelize, gaussian, Canny, number_of_points, poind_rad)
#
#
## image 3
#path_image = r'ellipses/nEKGD2wNiwqrTOc63kiWZT7b4.png'
#upbound = 320
#lowbound = 150
#cuber = [500, 110]
#a_b_normelize = [3, 1]
#gaussian = 5
#Canny = [407, 9]
#number_of_points = 800
#poind_rad = 13
#Hough_Functions.Run_Hough(path_image, upbound, lowbound, cuber,
#                          a_b_normelize, gaussian, Canny, number_of_points, poind_rad)
#
#
#
## image 4
#path_image = r'ellipses/truck.jpg'
#upbound = 210
#lowbound = 150
#cuber = [420, 110]
#a_b_normelize = [2, 1]
#gaussian = 5
#Canny = [416, 24]
#number_of_points = 1700
#poind_rad = 13
#Hough_Functions.Run_Hough(path_image, upbound, lowbound, cuber,
#                         a_b_normelize, gaussian, Canny, number_of_points, poind_rad)
#
#
#
#
##image 5
#path_image = r'ellipses/hammer-tissot-big.jpg'
#upbound = 35
#lowbound = 10
#cuber = [60, 60]
#a_b_normelize = [3, 1]
#gaussian = 5
#Canny = [93, 115]
#number_of_points = 7000
#poind_rad = 15
#Hough_Functions.Run_Hough(path_image, upbound, lowbound, cuber,
#                        a_b_normelize, gaussian, Canny, number_of_points, poind_rad)
#
#
#
##image 6
#path_image = r'ellipses/1271488188_2077d21f46_b.jpg'
#upbound = 110
#lowbound = 65
#cuber = [215, 215]
#a_b_normelize = [2, 1]
#gaussian = 11
#Canny = [49, 0]
#number_of_points = 6600
#poind_rad = 15
#Hough_Functions.Run_Hough(path_image, upbound, lowbound, cuber,
#                        a_b_normelize, gaussian, Canny, number_of_points, poind_rad)
#

import cv2
import numpy as np
import os
import random
import math


def load_images_from_folder(folder):  # load images without target image
    image_return = []
    for filename in os.listdir(folder):
        img_load = cv2.imread(os.path.join(folder, filename))
        if img_load is not None:
            if filename != "target.jpg":
                image_return.append(img_load)
            else:
                image_return_t = img_load
    return image_return, image_return_t


def image_to_gray(color_images):
    gray_image = []
    for img_color in color_images:
        gray_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        gray_image.append(gray_img)
    return gray_image


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #path = r'C:\Users\saars\PycharmProjects\Computer_Vision\hw1\Q2\cameleon'
    path = r'C:\Users\saars\PycharmProjects\Computer_Vision\hw1\Q2\einstein'
    images, image_target = load_images_from_folder(path)  # load images from folder
    gray_images = image_to_gray(images)  # change source images to gray
    gray_target_image = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)  # change target image to gray
    sift = cv2.SIFT_create()  # create sift
    keypoints_t, descriptors_t = sift.detectAndCompute(gray_target_image, None)  # detect features in target image
    cols = gray_target_image.shape[1]  # cols in target image
    rows = gray_target_image.shape[0]  # rows in target image
    count_matrix = np.ones((rows, cols, 3))  # will hold how much source images contribute to pixel
    color_matrix = np.zeros((rows, cols, 3))  # will sum all the pixel values of source images who contribute to target image
    for i in range(len(gray_images)):  # start run on every source picture
        cols_source = gray_images[i].shape[1]  # cols in target image
        rows_source = gray_images[i].shape[0]  # rows in target image
        check = np.ones((rows_source, cols_source))  # will hold one if pixel in target image after warp if not contain zero in pixel index
        matches_good = []  # will hold all the good matches after ratio test
        keypoints_1, descriptors_1 = sift.detectAndCompute(gray_images[i], None)  # detect features in source image
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_1, descriptors_t, k=2)  # finds matches between source and target image
        for m, n in matches:  # do ratio test on matches
            if m.distance < 0.8*n.distance:
                matches_good.append(m)
    #  finish ratio test and start ransac as we asked for
        best_model_inliers = -1
        best_M = None  # best matrix transform we found
        threshold = 0.5
        itr = len(matches_good)
        for j in range(itr):
            counter = 0  # count how much pixels under threshold
            random_matches = random.sample(matches_good, 4)  # Sample 4 random matches
            sources = np.float32([keypoints_1[random_matches[i].queryIdx].pt for i in range(4)]).reshape(-1, 1, 2)  # receive the 4 samples matches(x,y) index of source image
            targets = np.float32([keypoints_t[random_matches[i].trainIdx].pt for i in range(4)]).reshape(-1, 1, 2)  # receive the 4 samples matches(x,y) index of target image
            M, mask = cv2.findHomography(sources, targets, cv2.INTER_CUBIC)   # find the matrix transform
            for m in matches_good:
                m_source = keypoints_1[m.queryIdx].pt
                m_target_old = keypoints_t[m.trainIdx].pt
                m_target_new = np.matmul(M, [m_source[0], m_source[1], 1])
                #  start calc distances of points
                if m_target_new[2] != 0:
                    m_target_new[0] = m_target_new[0] / m_target_new[2]
                    m_target_new[1] = m_target_new[1] / m_target_new[2]
                euclidean_distance = math.sqrt(((m_target_old[0] - m_target_new[0]) ** 2) + ((m_target_old[1] - m_target_new[1]) ** 2))
                if euclidean_distance < threshold:
                    counter += 1
            if counter > best_model_inliers:
                best_model_inliers = counter
                best_M = M
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches_good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_t[m.trainIdx].pt for m in matches_good]).reshape(-1, 1, 2)
        dst_true = cv2.warpPerspective(images[i], best_M, (cols, rows), flags=cv2.INTER_CUBIC)  # true image after transform
        check_dst = cv2.warpPerspective(check, best_M, (cols, rows), flags=cv2.INTER_CUBIC)  # check where our pixels need to be added if 1 so it need to be if not 0
        for w_val in range(rows):
            for h_val in range(cols):  # check if our pixel add to our target picture
                if check_dst[w_val][h_val] != 0:  # pixel need to be calculate
                    if dst_true[w_val][h_val].all() != 0:
                        count_matrix[w_val][h_val] = count_matrix[w_val][h_val] + 1  # add 1 to count we at the end will divide the color value by this
                        color_matrix[w_val][h_val] = np.add(color_matrix[w_val][h_val], dst_true[w_val][h_val])  # good pixel so we add the color of him
    color_matrix_1 = np.add(color_matrix, image_target)  # add target pic to our pictures
    color_matrix = np.divide(color_matrix_1, count_matrix)
    cv2.imshow("Clean Target Image", color_matrix.astype(np.uint8))
    cv2.imshow("Source Target Image", image_target)
    cv2.waitKey(0)

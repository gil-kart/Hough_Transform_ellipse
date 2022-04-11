import random
from math import ceil

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(ellipses_color, a, b, theta, _color, flag_thick, center):
    image = ellipses_color

    axesLength = (a, b)
    angle = theta

    startAngle = 791

    endAngle = 141

    # Red color in BGR
    color = _color

    # Line thickness of 5 px
    thickness = flag_thick

    window_name = 'Image'
    # Using cv2.ellipse() method
    # Draw an ellipse with blue line borders of thickness of -1 px
    image = cv2.ellipse(image, center, axesLength, angle,
                        startAngle, endAngle, color, thickness)

    # Displaying the image
    plt.imshow(image)
    plt.show()
    return image


def get_votes_indexes(var, dict_x):
    for index, i_range in dict_x.items():
        if var <= i_range:
            return index
    return 8


#
# def get_votes_indexes_neg(var, dict_x):
#     for i, i_range in dict_x.items():
#         if var <= i_range:
#             return i
#     return 8


def count_votes(a_round, b_round, theta, center_coordinates, ellipses_edge, d3_mat):
    votes = 0
    for x in range(center_coordinates[0] - 180, center_coordinates[0] + 180):
        if x > ellipses_edge.shape[0] or x < 0:
            continue
        y = int(np.sqrt((1 - (((x) ** 2) / a_round ** 2)) * b_round ** 2))
        if np.isnan(y):
            continue
        if y > ellipses_edge.shape[1] or y < 0:
            continue
        if ellipses_edge[x][y] == 1:
            votes += 1
    return votes


def calc_cur(votes_mat, i, j):
    res_ret = 0
    for row in range(i - 12, i + 12):
        for col in range(j - 12, j + 12):
            res_ret += votes_mat[row][col]
    return res_ret


def calc_a_b_theta(B, C, D):
    if ((B + 1) - np.sqrt((B - 1) ** 2 + 4 * D ** 2)) == 0:  # prevent division by zero
        return -1, -1, -1
    equ1 = (-2 * C) / ((B + 1) - np.sqrt(((B - 1) ** 2) + 4 * (D ** 2)))
    equ2 = (-2 * C) / ((B + 1) + np.sqrt(((B - 1) ** 2) + 4 * (D ** 2)))
    a = np.sqrt(equ1)  # calculate a, b and theta from the article
    b = np.sqrt(equ2)
    if (1 - B) == 0:  # prevent division by zero
        return -1, -1, -1
    theta_rad = np.arctan2((2 * D), (1 - B))
    if a < 0 or b < 0:
        return -1, -1, -1
    a_round = int(a / 6)
    b_round = int(b)
    theta = -0.5 * np.degrees(theta_rad)
    return a_round, b_round, theta

def distance_not_too_small(top_three_index, votes_mat, point, top_three):
    check_point = np.array(point)
    for i, index in enumerate(top_three_index):
        cur_point = np.array(index)
        dist = np.linalg.norm(check_point-cur_point)
        if dist < 40:
            if votes_mat[index[0]][index[1]] < votes_mat[point[0]][point[1]]:
                top_three_index[i] = point
            return False
    return True



def update_dicts(dict_b, dict_c, dict_d, cur_max_index):
    # ------------------- this method scales the ranges of B,C,D --------------------------------

    # for index, cur_dict in enumerate([dict_b, dict_c]):  # for each dictionary we update the keys
    #     if cur_max_index[index] == 8:  # if it's the last index we want to go to higher values
    #         max_dict = cur_dict[cur_max_index[index]] + int(cur_dict[cur_max_index[index]] / 3)
    #         min_dict = cur_dict[cur_max_index[index]]
    #     else:
    #         max_dict = cur_dict[cur_max_index[index]]
    #         if cur_max_index[index] > 0:
    #             min_dict = cur_dict[cur_max_index[index] - 1]
    #         else:
    #             min_dict = cur_dict[cur_max_index[index]] - int(abs(cur_dict[cur_max_index[index]] - 1) / 5)
    #     adder = int((max_dict - min_dict) / 9)
    #     if cur_max_index[index] == 0:
    #         max_dict = cur_dict[cur_max_index[index]]
    #         min_dict = cur_dict[cur_max_index[index]] - int(abs(cur_dict[cur_max_index[index]] - 1) / 5)
    #     else:
    #         max_dict = cur_dict[cur_max_index[index]]
    #         min_dict = cur_dict[cur_max_index[index] - 1]
    #     adder = int((max_dict - min_dict) / 9)
    #     for key, val in cur_dict.items():
    #         cur_dict[key] = min_dict
    #         min_dict += adder

    # if dict_d[cur_max_index[2]] ** 2 > dict_b[0]:
    #     max_dict_d = dict_b[0]
    #     min_dict_d = -dict_b[0]
    # else:
    #     if cur_max_index[2] == 0:
    #         max_dict_d = dict_d[cur_max_index[2] + 1]
    #         min_dict_d = dict_d[cur_max_index[2]]
    #     else:
    #         max_dict_d = dict_d[cur_max_index[2]]
    #         min_dict_d = dict_d[cur_max_index[2] - 1]
    #
    # adder = int(abs((max_dict_d + min_dict_d) / 9))
    # for key, val in dict_d.items():
    #     dict_d[key] = min_dict_d
    #     min_dict_d += adder
    # return

    # ------------------- this method keep B,C,D at the original range of the beginning --------------------------------
    for index, cur_dict in enumerate([dict_b, dict_c, dict_d]):  # for each dictionary we update the keys
        if cur_max_index[index] == 0:  # if maximal index is at the smallest bucket
            max_dict_val = cur_dict[cur_max_index[index] + 1]
            min_dict = cur_dict[cur_max_index[index]]
        else:  # maximal index is between 1 and 8
            max_dict_val = cur_dict[cur_max_index[index]]
            min_dict = cur_dict[cur_max_index[index] - 1]

        if index == 2:  # if we are checking dict_d
            if dict_d[cur_max_index[2]] ** 2 >= dict_b[0]:  # making sure that D is in the range  - sqrt(b), sqrt(b)
                max_dict_val = int(np.sqrt(dict_b[8]))
                min_dict = -1 * int(np.sqrt(dict_b[8]))
        adder = int((max_dict_val - min_dict) / 9)  # divide the bucket to 9 new parts
        for key, val in cur_dict.items():
            cur_dict[key] = min_dict
            min_dict += adder
            if cur_dict[key] > max_dict_val or key == 8:
                cur_dict[key] = max_dict_val
    return


def find_min_val_and_index(top_three_index, votes_mat):
    min_val = votes_mat[top_three_index[0][0]][top_three_index[0][1]]
    ret_indexes = top_three_index[0]
    for index in top_three_index:
        if votes_mat[index[0]][index[1]] <= min_val:
            min_val = votes_mat[index[0]][index[1]]
            ret_indexes = [index[0], index[1]]
    return ret_indexes, min_val



def remove_and_update(top_three, top_three_index, votes_mat, new_index):
    min_index, min_val = find_min_val_and_index(top_three_index, votes_mat)
    top_three_index.remove(min_index)
    top_three_index.append(new_index)
    # #  ----------- making sure that d is in the range  - sqrt(b), sqrt(b) ----------
    # if dict_d[cur_max_index[2]] ** 2 > dict_b[8]:
    #     max_dict_d = int(np.sqrt(dict_b[8]))
    #     min_dict_d = -1 * int(np.sqrt(dict_b[8]))
    # elif cur_max_index[2] == 0:
    #     max_dict_d = dict_d[1]
    #     min_dict_d = dict_d[0]
    # else:
    #     max_dict_d = dict_d[cur_max_index[2]]
    #     min_dict_d = dict_d[cur_max_index[2] - 1]
    # adder = int((max_dict_d - min_dict_d) / 9) + 1
    # for key, val in dict_d.items():
    #     if key == 8:
    #         dict_d[key] = max_dict_d
    #     else:
    #         dict_d[key] = min_dict_d
    #         min_dict_d += adder
    #         if dict_d[key] > max_dict_d:
    #             dict_d[key] = max_dict_d

    #
    # if cur_max_index[0] > 0:
    #     max_b = dict_b[cur_max_index[0]]
    #     min_b = dict_b[cur_max_index[0] - 1]
    #     adder = int((max_b - min_b) / 9)
    #     for key, val in dict_b.items():
    #         dict_b[key] = min_b
    #         min_b += adder
    # if cur_max_index[1] > 0:
    #     max_c = dict_c[cur_max_index[1]]
    #     min_c = dict_c[cur_max_index[1] - 1]
    #     adder = int((max_c - min_c) / 9)
    #     for key, val in dict_c.items():
    #         dict_c[key] = min_c
    #         min_c += adder
    # else:
    #     max_c = dict_c[cur_max_index[1]]
    #     min_c = dict_c[cur_max_index[1]] - abs(dict_c[cur_max_index[0]])
    #     adder = int((max_c - min_c) / 9)
    #     for key, val in dict_c.items():
    #         dict_c[key] = min_c
    #         min_c += adder
    #
    # if cur_max_index[2] > 0:
    #     max_d = dict_d[cur_max_index[2]]
    #     min_d = dict_d[cur_max_index[2] - 1]
    #     adder = int((max_d - min_d) / 9)
    #     for key, val in dict_b.items():
    #         dict_d[key] = min_d
    #         min_d += adder
    # else:
    #     max_d = dict_d[cur_max_index[2]]
    #     min_d = dict_d[cur_max_index[2]] - abs(dict_d[cur_max_index[0]])
    #     adder = int((max_d - min_d) / 9)
    #     for key, val in dict_b.items():
    #         dict_d[key] = min_d
    #         min_d += adder
    #
    # return


# reading the image
path_image = r'ellipses/images.jpg'
ellipses_color = cv2.imread(path_image)
ellipses_gray = cv2.cvtColor(ellipses_color, cv2.COLOR_BGR2GRAY)
# ellipses_edge = cv2.Canny(ellipses_gray, 400, 450)
# plt.imshow(ellipses_edge)
# plt.show()
#
# ellipses_edge = cv2.Canny(ellipses_gray, 800, 850)
# plt.imshow(ellipses_edge)
# plt.show()

# ellipses_gray = cv2.blur(ellipses_gray, (6, 6), cv2.BORDER_DEFAULT)
# plt.imshow(ellipses_gray)
# plt.show()

ellipses_edge = cv2.Canny(ellipses_gray, 900, 1150)
plt.imshow(ellipses_edge)
plt.show()



# empty_img_zero = np.zeros(ellipses_edge.shape)
# center_coordinates = (207, 227)
# a, b, theta = calc_a_b_theta(11, -27973, -2)
# empty_img_col = show_image(empty_img_zero, a, b, theta, (255, 130, 130), -1)
# path_empty =  r'ellipses/elipse_test5.jpg'
# cv2.imwrite(path_empty, empty_img_col)
# im = cv2.imread(path_image)
# im2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# empty_img = cv2.Canny(im2, 400, 450)
# plt.imshow(empty_img)
# plt.show()
# plt.imshow(im)
# plt.show()

# center_coordinates = (336, 319) # (204, 128)
# a, b, theta = calc_a_b_theta(32, -1578632, 3)
# show_image(ellipses_color, a, b, theta)

# image = ellipses_color
# # center_coordinates
#
# axesLength = (1825, 204)
# angle = 157
#
# # angle = 90
# startAngle = 791
#
# endAngle = 141
#
# # Red color in BGR
# color = (255, 0, 255)
#
# # Line thickness of 5 px
# thickness = 5
# center_coordinates = (343, 321)
# window_name = 'Image'
# # Using cv2.ellipse() method
# # Draw a ellipse with blue line borders of thickness of -1 px
# image = cv2.ellipse(image, center_coordinates, axesLength, angle,
#                     startAngle, endAngle, color, thickness)
#
# # Displaying the image
# plt.imshow(image)
# plt.show()
# dasfdas = 0
# center_coordinates = (331, 321)
# a, b, theta = calc_a_b_theta(197632, -512000127, -9288)
# show_image(ellipses_color, a, b, theta)
# center_coordinates = (204, 128)
# a, b, theta = calc_a_b_theta(8, -19212, 2)
# show_image(ellipses_color, a, b, theta)

gradients = {}  # find direction of edges for each pixel
for i in range(1, len(ellipses_edge) - 1):
    for j in range(1, len(ellipses_edge[0]) - 1):
        cur_gradient = [ellipses_edge[i + 1][j] - ellipses_edge[i - 1][j],
                        ellipses_edge[i][j + 1] - ellipses_edge[i][j - 1]]
        gradients[str(i) + ',' + str(j)] = np.power(np.tan(cur_gradient[1] / cur_gradient[0]), -1)

list_of_points_on_edges = []  # saving only gradients of edges
for i in range(1, len(ellipses_edge)):
    for j in range(1, len(ellipses_edge[0])):
        if ellipses_edge[i][j] != 0:
            d = i * np.cos([gradients[str(i) + ',' + str(j)]]) - j * np.sin([gradients[str(i) + ',' + str(j)]])
            list_of_points_on_edges.append([d[0], i, j])
line_list = []

random_index_list = list(range(0, len(list_of_points_on_edges) - 1))

#finding middle point for one ellipse
#for _ in range(
#        int(len(list_of_points_on_edges))):  # creating all lines that are created according to simon's article
#    for _ in range(int(len(list_of_points_on_edges) / 10 )):
for _ in range(100):  # creating all lines that are created according to simon's article
    for _ in range(800):
        i = random.choice(random_index_list)
        j = random.choice(random_index_list)
        if i != j and list_of_points_on_edges[i][0] != list_of_points_on_edges[j][0]:
            # creating x1, x2, y1, y2, epsilon1, epsilon2
            point_1 = list_of_points_on_edges[i]
            point_2 = list_of_points_on_edges[j]
            y1 = point_1[2]
            x1 = point_1[1]
            y2 = point_2[2]
            x2 = point_2[1]
            epsi1 = 100 * -1 / point_1[0]  # slopes of the tangent to point1
            epsi2 = 100 * -1 / point_2[0]  # slopes of the tangent to point2

            check = abs(x1 - x2)
            check2 = abs(y1 - y2)
            check3 = abs(epsi2 - epsi1)
            # if points are too far away, we don't use them

            dist = np.linalg.norm(np.array((x1, y1)) - np.array((x2, y2)))
            if dist > 80:
                continue

            # using the article's formula to create t1, t2, m1, m2
            t1 = (y1 - y2 - x1 * epsi1 + x2 * epsi2) / (epsi2 - epsi1)
            t2 = (epsi2 * epsi1 * (x2 - x1) - y2 * epsi1 + y1 * epsi2) / (epsi2 - epsi1)
            m1 = (x1 + x2) / 2
            m2 = (y1 + y2) / 2

            # saving each line using two parameters a and b
            a = (t2 - m2) / (t1 - m1)
            b = (m2 * t1 - m1 * t2) / (t1 - m1)
            if not np.isnan(a) and not np.isnan(b):
                if [(a, b)] not in line_list:
                    if x1 < x2:
                        start_point = [x1, y1]
                        end_point = [x2, y2]
                    else:
                        start_point = [x2, y2]
                        end_point = [x1, y1]
                    line_list.append([[a, b], [start_point, end_point]])




pixels_votes_dict = {}
votes_mat = np.zeros(ellipses_edge.shape)
for i in range(len(votes_mat)):
    for j in range(len(votes_mat[0])):
        pixels_votes_dict[str(i) + ', ' + str(j)] = []

for line in line_list:
    for x in range(line[1][0][0], line[1][1][0]):
        y = x * line[0][0] + line[0][1]
        ranger_y_high = max(line[1][0][1], line[1][1][1])
        ranger_y_low = min(line[1][0][1], line[1][1][1])
        if ranger_y_low <= y < ranger_y_high:  # and 0 <= y < votes_mat.shape[1]:
            votes_mat[int(x)][int(y)] += 1
            pixels_votes_dict[str(int(x)) + ', ' + str(int(y))].append([line[1][0][0], line[1][0][1]])

center_of_votes = pixels_votes_dict['126, 81']
center_of_votes2 = pixels_votes_dict['49, 150']

top_three = [votes_mat[0][0], votes_mat[0][1], votes_mat[0][2]]
top_three_index = [[0, 0], [0, 1], [0, 2]]
centers = []
indexes = [0, 0]
max_size = 0
local_maiximums = []


def find_local_maximum(i, j, votes_mat, cube):
    cur_max = 0
    ret = [0, 0]
    for row in range(i - cube, i):
        for col in range(j - cube, j):
            if votes_mat[row][col] > cur_max:
                cur_max = votes_mat[row][col]
                ret = [row, col]
    if cur_max == 0:
        return False, cur_max, ret
    return True, cur_max, ret


def takethird(elem):
    return elem[1]

cube = 50
for i in range(cube, len(votes_mat), cube):
    for j in range(cube, len(votes_mat[0]), cube):
        there_are_counts, val, local_maximum = find_local_maximum(i, j, votes_mat, cube)
        if there_are_counts:
            local_maiximums.append([local_maximum, val])
local_maiximums.sort(key=takethird)

# for i in range(12, len(votes_mat) - 12):
#     for j in range(12, len(votes_mat[0]) - 12):
#         res = calc_cur(votes_mat, i, j)
#         if res <= min(top_three):
#             continue
#         if distance_not_too_small(top_three_index, votes_mat, [i, j], top_three):
#             remove_and_update(top_three, top_three_index, votes_mat, [i,j])

for index in local_maiximums:

    check = index[0][1]
    check2 = index[0][0]

    ellipses_gray = cv2.circle(ellipses_gray, (index[0][1], index[0][0]), radius=5, color=(0, 0, 0),
                       thickness=-1)

#
# image = cv2.circle(ellipses_gray, (top_three_index[0][1], top_three_index[0][0]), radius=1, color=(0, 0, 0), thickness=-1)
# image = cv2.circle(image, (top_three_index[1][1], top_three_index[1][0]), radius=1, color=(0, 0, 0), thickness=-1)
# image = cv2.circle(image, (top_three_index[2][1], top_three_index[2][0]), radius=1, color=(0, 0, 0), thickness=-1)

plt.figure()
plt.imshow(ellipses_gray)
plt.show()
zaop = 0


# plt.figure()
# plt.imshow(votes_mat)
#
# ellipses_edge[top_five_index[len(top_five_index) - 1][0]][top_five_index[len(top_five_index) - 1][1]] = 255
# plt.show()
# zxcpo = 0
# plt.imshow(ellipses_edge)
# plt.show()
# zxcpo = 0
#
# d3_mat = np.zeros((9, 9, 9))
#
# cur_max = 0
# cur_max_index = [0, 0, 0, False]
#
center_coordinates = (top_three_index[len(top_three_index) - 1][0], top_three_index[len(top_three_index) - 1][1])

b_start = 1
b_add = 17
dict_b = {0: b_start,
          1: b_start + b_add,
          2: b_start + 2 * b_add,
          3: b_start + 3 * b_add,
          4: b_start + 4 * b_add,
          5: b_start + 5 * b_add,
          6: b_start + 6 * b_add,
          7: b_start + 7 * b_add,
          8: b_start + 8 * b_add}

dict_c = {0: -1000000,
          1: -10000,
          2: -1000,
          3: 1000,
          4: 2500,
          5: 5000,
          6: 7500,
          7: 11000,
          8: 15000}

add_me = int((np.sqrt(dict_b[8]) / 9) * 2) + 1
d_val = int(-np.sqrt(dict_b[8]))
dict_d = {0: d_val + 1,
          1: d_val + add_me,
          2: d_val + 2 * add_me,
          3: d_val + 3 * add_me,
          4: d_val + 4 * add_me,
          5: d_val + 5 * add_me,
          6: d_val + 6 * add_me,
          7: d_val + 7 * add_me,
          8: d_val + 8 * add_me - 1}

cur_max = 0
center_coordinates = (83, 128)
center_coordinates2 = (148, 49)
# (275, 228) # (336, 319)  # (204, 128)  #
d3_mat = np.zeros((9, 9, 9))
cur_max_index = [0, 0, 0]
centers = [center_coordinates, center_coordinates2]
#for center in [center_coordinates, center_coordinates2]:
for center_i, center_vote in enumerate([center_of_votes, center_of_votes2]):
    for number_of_range_fixes in range(20):
        # for _ in range(int(len(list_of_points_on_edges) / 20)):
        # for number_of_points in range(2605):
        for vote_point in center_vote:
            #i = random.choice(random_index_list)
            #point = list_of_points_on_edges[i]
            point = vote_point
            #        for B in range(dict_b[0], dict_b[8] + 1, int(((dict_b[1] - dict_b[0])) / 20) + 1):
            #            for D in range(-1 * int(np.sqrt(B)),
            #                           int(np.sqrt(B)), int(((dict_d[1] - dict_d[0])) / 20) + 1):  # iterate over all possible value of B,D
            for B in range(dict_b[0], dict_b[8] + 1):
                for D in range(dict_d[0], dict_d[8]):  # iterate over all possible value of B,D
                    if B <= D ** 2:
                        continue
                    #x = point[1] - center_coordinates[1]  # move to center
                    #y = point[2] - center_coordinates[0]
                    x = point[0] - center_coordinates[0]  # move to center
                    y = point[1] - center_coordinates[1]
                    C = -(np.power(x, 2) + B * np.power(y, 2) + (
                                2 * D * x * y))  # find C according to B and D's constraints

                    d1 = get_votes_indexes(B, dict_b)  # count votes for every maximal index
                    d2 = get_votes_indexes(C, dict_c)  # in each dimension of the vote matrix
                    d3 = get_votes_indexes(D, dict_d)

                    d3_mat[d1][d2][d3] += 1  # add count to current index in vote matrix

                    if d3_mat[d1][d2][d3] > cur_max:  # update maximal index
                        cur_max = d3_mat[d1][d2][d3]
                        cur_max_index = [d1, d2, d3]
                    # if ((B + 1) - np.sqrt((B - 1) ** 2 + 4 * D ** 2)) == 0:  # prevent division by zero
                    #     continue
                    #
                    # equ1 = (-2 * C) / ((B + 1) - np.sqrt(((B - 1) ** 2) + 4 * (D ** 2)))
                    # equ2 = (-2 * C) / ((B + 1) + np.sqrt(((B - 1) ** 2) + 4 * (D ** 2)))
                    # a = np.sqrt(equ1)                                       # calculate a, b and theta from the article
                    # b = np.sqrt(equ2)
                    # if (1 - B) == 0:   # prevent division by zero
                    #     continue
                    # thet = theta = 0.5 * np.power(np.tan((2 * D) / (1 - B)))
                    # neg_flag = False
                    # if theta < 0:
                    #     neg_flag = True
                    #     theta *= -1
                    #
                    # if a > 800 or b > 800 or theta * 100 >= 300:
                    #     continue
                    # if a < 0 or b < 0 or theta < 0:
                    #     continue
                    # a_round = int(a)
                    # b_round = int(b)
                    # theta_round_mul = int(theta * 100)
                    # ####################
                    # if np.isnan(a) or np.isnan(b) or np.isnan(thet):
                    #     continue
                    # votes = count_votes(a_round, b_round, thet, center_coordinates, ellipses_edge, d3_mat)
                    # ###################
                    # try:
                    #    d3_mat[a_round][b_round][theta_round_mul] += 1              # add count to bucket
                    #    if d3_mat[a_round][b_round][theta_round_mul] > cur_max:
                    #        cur_max = d3_mat[a_round][b_round][theta_round_mul]
                    #        cur_max_index = [a_round, b_round, theta_round_mul, neg_flag]
                    # except:
                    #    continue
                    # D += 300
                    # C += 1000
                    # try:
                    #     d3_mat[int(B)][int(C)][int(D)] += 1
                    #     if d3_mat[int(B)][int(C)][int(D)] > cur_max:
                    #         cur_max = d3_mat[int(B)][int(C)][int(D)]
                    #         cur_max_index = [int(B), int(C), int(D)]
                    # except:
                    #     continue

        update_dicts(dict_b, dict_c, dict_d, cur_max_index)
        cur_max = 0
        d3_mat = np.zeros((9, 9, 9))

    B, C, D = dict_b[cur_max_index[0]], dict_c[cur_max_index[1]], dict_d[cur_max_index[2]]
    if B < D ** 2:
        if D > 0:
            D = -1 * (np.sqrt(B) - 1)
        else:
            D = np.sqrt(B) - 1
    a, b, theta = calc_a_b_theta(B, C, D)
    show_image(ellipses_color, a, b, theta, (20, 100, 255), 3, centers[center_i])
    dict_b = {0: b_start,
              1: b_start + b_add,
              2: b_start + 2 * b_add,
              3: b_start + 3 * b_add,
              4: b_start + 4 * b_add,
              5: b_start + 5 * b_add,
              6: b_start + 6 * b_add,
              7: b_start + 7 * b_add,
              8: b_start + 8 * b_add}

    dict_c = {0: -100000,
              1: -10000,
              2: -1000,
              3: 1000,
              4: 2500,
              5: 5000,
              6: 7500,
              7: 11000,
              8: 15000}

    add_me = int((np.sqrt(dict_b[8]) / 9) * 2) + 1
    d_val = int(-np.sqrt(dict_b[8]))
    dict_d = {0: d_val + 1,
              1: d_val + add_me,
              2: d_val + 2 * add_me,
              3: d_val + 3 * add_me,
              4: d_val + 4 * add_me,
              5: d_val + 5 * add_me,
              6: d_val + 6 * add_me,
              7: d_val + 7 * add_me,
              8: d_val + 8 * add_me - 1}

dasfdas = 0
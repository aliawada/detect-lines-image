import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import math
import pdb
import random


def randColor():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    rgb = [r, g, b]

    return rgb


def hough_transform(img_bin, theta_res=1, rho_res=1):
    nR, nC = img_bin.shape
    theta = np.linspace(-90, 0, int(np.ceil(90.0 / theta_res)) + 1)
    theta = np.concatenate((theta, -theta[len(theta) - 2::-1]))

    D = np.sqrt((nR - 1) ** 2 + (nC - 1) ** 2)
    q = np.ceil(D / rho_res)
    nrho = 2 * q + 1
    rho = np.linspace(int(-q * rho_res), int(q * rho_res), int(nrho))
    H = np.zeros((len(rho), len(theta)))
    for rowIdx in range(nR):
        for colIdx in range(nC):
            if img_bin[rowIdx, colIdx]:
                for thIdx in range(len(theta)):
                    rhoVal = colIdx * np.cos(theta[thIdx] * np.pi / 180.0) + \
                             rowIdx * np.sin(theta[thIdx] * np.pi / 180)
                    rhoIdx = np.nonzero(np.abs(rho - rhoVal) == np.min(np.abs(rho - rhoVal)))[0]
                    H[rhoIdx[0], thIdx] += 1
    return rho, theta, H


def top_n_rho_theta_pairs(ht_acc_matrix, n, rhos, thetas):
    '''
    @param hough transform accumulator matrix H (rho by theta)
    @param n pairs of rho and thetas desired
    @param ordered array of rhos represented by rows in H
    @param ordered array of thetas represented by columns in H
    @return top n rho theta pairs in H by accumulator value
    @return x,y indexes in H of top n rho theta pairs
    '''
    flat = list(set(np.hstack(ht_acc_matrix)))
    flat_sorted = sorted(flat, key=lambda n: -n)
    coords_sorted = [(np.argwhere(ht_acc_matrix == acc_value)) for acc_value in flat_sorted[0:n]]
    rho_theta = []
    x_y = []
    for coords_for_val_idx in range(0, len(coords_sorted), 1):
        coords_for_val = coords_sorted[coords_for_val_idx]
        for i in range(0, len(coords_for_val), 1):
            n, m = coords_for_val[i]  # n by m matrix
            rho = rhos[n]
            theta = thetas[m]
            rho_theta.append([rho, theta])
            x_y.append([m, n])  # just to unnest and reorder coords_sorted
    return [rho_theta[0:n], x_y]


def valid_point(pt, ymax, xmax):
    '''
    @return True/False if pt is with bounds for an xmax by ymax image
    '''
    x, y = pt
    if x <= xmax and x >= 0 and y <= ymax and y >= 0:
        return True
    else:
        return False


def round_tup(tup):
    '''
    @return closest integer for each number in a point for referencing
    a particular pixel in an image
    '''
    x, y = [int(round(num)) for num in tup]
    return (x, y)


def draw_rho_theta_pairs(target_im, pairs):
    '''
    @param opencv image
    @param array of rho and theta pairs
    Has the side-effect of drawing a line corresponding to a rho theta
    pair on the image provided
    '''
    im_y_max, im_x_max, channels = np.shape(target_im)
    for i in range(0, len(pairs), 1):
        point = pairs[i]
        rho = point[0]
        theta = point[1] * np.pi / 180  # degrees to radians
        # y = mx + b form
        m = -np.cos(theta) / np.sin(theta)
        b = rho / np.sin(theta)
        # possible intersections on image edges
        left = (0, b)
        right = (im_x_max, im_x_max * m + b)
        top = (-b / m, 0)
        bottom = ((im_y_max - b) / m, im_y_max)

        pts = [pt for pt in [left, right, top, bottom] if valid_point(pt, im_y_max, im_x_max)]
        if len(pts) == 2:
            p_lines.append([round_tup(pts[0]), round_tup(pts[1])])
            cv2.line(target_im, round_tup(pts[0]), round_tup(pts[1]), (0, 0, 255), 1)


# ----------------------------------------------------------------------------------------------------- #
# main
img_orig = cv2.imread('parallel-lines.png')
img = img_orig[:, :, ::-1]  # color channel plotting mess http://stackoverflow.com/a/15074748/2256243

bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(bw, threshold1=0, threshold2=50, apertureSize=3)
rhos, thetas, H = hough_transform(edges)

rho_theta_pairs, x_y_pairs = top_n_rho_theta_pairs(H, 22, rhos, thetas)
im_w_lines = img.copy()
p_lines = []
draw_rho_theta_pairs(im_w_lines, rho_theta_pairs)

img_parallel_lines = img.copy()

i = 0
# draw parallel lines
for line1 in p_lines:
    line1_y2 = line1[1][1]
    line1_y1 = line1[0][1]
    line1_x2 = line1[1][0]
    line1_x1 = line1[0][0]
    slope1 = (line1_y2 - line1_y1) / (line1_x2 - line1_x1)
    for line2 in p_lines:
        if not np.array_equal(line1, line2):
            line2_y2 = line2[1][1]
            line2_y1 = line2[0][1]
            line2_x2 = line2[1][0]
            line2_x1 = line2[0][0]
            slope2 = (line2_y2 - line2_y1) / (line2_x2 - line2_x1)
            if slope1 == slope2:
                i = i + 1
                rgb = randColor()
                cv2.line(img_parallel_lines, (line1_x1, line1_y1), (line1_x2, line1_y2), rgb, 2)
                cv2.line(img_parallel_lines, (line2_x1, line2_y1), (line2_x2, line2_y2), rgb, 2)
                cv2.putText(img_parallel_lines, "%d" % i, (line1_x1, line1_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2,
                            cv2.LINE_AA)
                cv2.putText(img_parallel_lines, "%d" % i, (line2_x2, line2_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2,
                            cv2.LINE_AA)
    #             stop = True
    #             break
    # if stop:
    #     break

# also going to draw circles in the accumulator matrix
for i in range(0, len(x_y_pairs), 1):
    x, y = x_y_pairs[i]
    cv2.circle(img=H, center=(x, y), radius=12, color=(0, 0, 0), thickness=1)

plt.subplot(151), plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(152), plt.imshow(edges, cmap='gray')
plt.title('Image Edges'), plt.xticks([]), plt.yticks([])
plt.subplot(153), plt.imshow(H)
plt.title('Hough Transform Accumulator'), plt.xticks([]), plt.yticks([])
plt.subplot(154), plt.imshow(im_w_lines)
plt.title('Detected Lines'), plt.xticks([]), plt.yticks([])
plt.subplot(155), plt.imshow(img_parallel_lines)
plt.title('Parallel Lines'), plt.xticks([]), plt.yticks([])


plt.show()
cv2.waitKey()

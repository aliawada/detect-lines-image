import cv2
import numpy as np
import random


# utility functions
def randColor():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    rgb = [r, g, b]

    return rgb


img = cv2.imread("parking.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

low_threshold = 50
high_threshold = 150

edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 10  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 5  # minimum number of pixels making up a line
max_line_gap = 3  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on
parallel_line_image = np.copy(img)

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

for line in lines:
    cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 0, 0), 5)

# Draw the lines on the  image
lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

i = 0
# draw parallel lines
for line1 in lines:
    line1_y2 = line1[0][3]
    line1_y1 = line1[0][1]
    line1_x2 = line1[0][2]
    line1_x1 = line1[0][0]
    slope1 = (line1_y2 - line1_y1) / (line1_x2 - line1_x1)
    for line2 in lines:
        if not np.array_equal(line1, line2):
            line2_y2 = line2[0][3]
            line2_y1 = line2[0][1]
            line2_x2 = line2[0][2]
            line2_x1 = line2[0][0]
            slope2 = (line2_y2 - line2_y1) / (line2_x2 - line2_x1)
            if slope1 == slope2:
                i = i + 1
                rgb = randColor()
                cv2.line(parallel_line_image, (line1_x1, line1_y1), (line1_x2, line1_y2), rgb, 2)
                cv2.line(parallel_line_image, (line2_x1, line2_y1), (line2_x2, line2_y2), rgb, 2)
                cv2.putText(parallel_line_image, "%d" % i, (line1_x1, line1_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2,
                            cv2.LINE_AA)
                cv2.putText(parallel_line_image, "%d" % i, (line2_x2, line2_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2,
                            cv2.LINE_AA)
    #             stop = True
    #             break
    # if stop:
    #     break

cv2.imshow("Source", img)
cv2.imshow("Lines", line_image)
cv2.imshow("Lines Edges", lines_edges)
cv2.imshow("Parallel Lines", parallel_line_image)

cv2.waitKey()

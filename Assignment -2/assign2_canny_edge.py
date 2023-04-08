import cv2
import numpy as np
import copy
from scipy import signal

pic = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(pic,(3,3),cv2.BORDER_DEFAULT)

row, column = blur.shape

#Defining 5x5 sobel kernels

sobel_x_matx = np.array([[1,4,6,4,1], [2,8,12,8,2], [0,0,0,0,0], [-2,-8,-12,-8,-2], [-1,-4,-6,-4,-1]])
sobel_y_matx = np.array([[1,2,0,-2,-1], [4,8,0,-8,-4], [6,12,0,-12,-6], [4,8,0,-8,-4],[1,2,0,-2,-1]])


# Convolve the image with the Sobel kernels
sobel_x_img = cv2.filter2D(blur, -1, sobel_x_matx)/100
sobel_y_img = cv2.filter2D(blur, -1, sobel_y_matx)/100

# Calculating Sobel magnitute and orientation_matrix
sobel_img = np.sqrt(np.square(sobel_x_img) + np.square(sobel_y_img))
orientation_matrix = np.arctan2(sobel_x_img,sobel_y_img)

# Finding non-Maximum Supression
non_max_img = np.zeros([row,column])

degree = orientation_matrix * 180. / np.pi  # Converting Radians to Degrees
degree[degree < 0] += 180

for i in range(1,row-1):
    for j in range(1,column-1):
        
        x1 = 255
        x2 = 255
        
        #angle ~ 0
        if (0 <= degree[i,j] < 22.5) or (157.5 <= degree[i,j] <= 180):
            x1 = sobel_img[i, j+1]
            x2 = sobel_img[i, j-1]
        #angle ~ 45
        elif (22.5 <= degree[i,j] < 67.5):
            x1 = sobel_img[i+1, j-1]
            x2 = sobel_img[i-1, j+1]
        #angle ~ 90
        elif (67.5 <= degree[i,j] < 112.5):
            x1 = sobel_img[i+1, j]
            x2 = sobel_img[i-1, j]
        #angle ~ 135
        elif (112.5 <= degree[i,j] < 157.5):
            x1 = sobel_img[i-1, j-1]
            x2 = sobel_img[i+1, j+1]

        if (sobel_img[i,j] >= x1) and (sobel_img[i,j] >= x2):
            non_max_img[i,j] = sobel_img[i,j]
        else:
            non_max_img[i,j] = 0

# Threshold
minimum_threshold_ratio=0.1
maximum_threshold_ratio=0.8

max_threshold = non_max_img.max() * maximum_threshold_ratio
min_threshold = non_max_img.max() * minimum_threshold_ratio

threshold_img = np.zeros([row,column])

weak = np.int32(100)
strong = np.int32(255)

strong_i, strong_j = np.where(non_max_img >= max_threshold)
zeros_i, zeros_j = np.where(non_max_img < min_threshold)

weak_i, weak_j = np.where((non_max_img <= max_threshold) & (non_max_img >= min_threshold))

threshold_img[strong_i, strong_j] = strong
threshold_img[weak_i, weak_j] = weak

image_after_thresholding = copy.deepcopy(threshold_img)

# Edge Linking
for i in range(1, row-1):
    for j in range(1, column-1):
        if (threshold_img[i,j] == weak):
            if ((threshold_img[i+1, j-1] == strong) or (threshold_img[i+1, j] == strong) or (threshold_img[i+1, j+1] == strong)
                    or (threshold_img[i, j-1] == strong) or (threshold_img[i, j+1] == strong)
                    or (threshold_img[i-1, j-1] == strong) or (threshold_img[i-1, j] == strong) or (threshold_img[i-1, j+1] == strong)):
                    threshold_img[i, j] = strong
            else:
                 threshold_img[i, j] = 0

while(True):
    cv2.imshow("Blurred image",blur)
    cv2.imshow("sobel_x_img",sobel_x_img)
    cv2.imshow("sobel_y_img",sobel_y_img)
    cv2.imshow("sobel_img",sobel_img)
    cv2.imshow("Non maximum supressed Image",non_max_img)
    cv2.imshow("Image After applying threshold ",image_after_thresholding)
    cv2.imshow("Final Image",threshold_img)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
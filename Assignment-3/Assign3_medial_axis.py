import cv2
import numpy as np
import copy

original_image = cv2.imread('cvSmall.png', cv2.IMREAD_GRAYSCALE) # Load the image and convert to grayscale

ret, binary_image = cv2.threshold(original_image, 210, 255, cv2.THRESH_BINARY) # Threshold the image to obtain a binary image

binary_image = 255 - binary_image # Invert the binary image to make object pixels 1 and background pixels 0

binary_image_copy = copy.deepcopy(binary_image) # Create a copy of the thresholded image for visualization purposes

medial_axis_image = np.zeros_like(binary_image) # Initialize the medial_axis_image image as an all-zero array of the same size as the thresholded image

kernel = np.ones((3,3), np.uint8) # Define the kernel for erosion and dilation operations

for pixel_idx in range(0, binary_image.shape[0]*binary_image.shape[1], 1): # Perform the medial axis transform using a for loop
   
    eroded_image = cv2.erode(binary_image, kernel, iterations=1) # Erode the binary image using the kernel
    dilated_eroded_image = cv2.dilate(eroded_image, kernel, iterations=1) # Dilate the eroded image using the kernel to obtain a temporary image
    temp_image = cv2.subtract(binary_image, dilated_eroded_image) # Obtain the difference between the binary image and the dilated eroded image
    medial_axis_image = cv2.bitwise_or(medial_axis_image, temp_image) # Use a bitwise OR operation to add the temporary image to the medial_axis_image image
    binary_image = eroded_image.copy() # Update the thresholded image to be the eroded image

    if cv2.countNonZero(binary_image) == 0: # Break out of the loop if the thresholded image is completely eroded
        break

while(True): # Display the original image, the thresholded image, and the resulting skeleton image
    cv2.imshow('original image', original_image)
    cv2.imshow("binary_image_copy ",binary_image_copy)
    cv2.imshow("medial_axis_skeleton_image",medial_axis_image)

    if cv2.waitKey(1) == 27: # Wait for the user to press the 'ESC' key to exit
        break

cv2.destroyAllWindows() # Destroy all open windows
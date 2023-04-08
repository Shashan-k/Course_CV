import cv2
import numpy as np

pic = cv2.imread('Lenna.png') # Loading original image
gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY) # Convering image in grayscale

levels = 5
error_image = []

def gaussian_pyramid(image, levels):
    img_pyramid = [image]
    for i in range(levels-1):
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        img_pyramid.append(image)
        
    return img_pyramid

#1. Calling Gaussian pyramid function with level 5
pyramid = gaussian_pyramid(gray, 5)
cv2.imshow('Downsampled to 32 x 32 ', pyramid[levels-1]) # Displaying the last downsampled image (32x32)

# To display the error image
for i in range(len(pyramid)):
    if i > 0:
        error = cv2.subtract(pyramid[i-1], cv2.resize(pyramid[i], (pyramid[i-1].shape[1], pyramid[i-1].shape[0])))
        cv2.imshow('Error ' + str(i-1), error)
        error_image.append(error)


#2. Reconstructing the original image using 32 X 32 and error images

reconstructed_img = pyramid[levels-1] # Storing 32 X 32 for reconstruction
for j in range(3,-1,-1):
    
    reconstructed_img = cv2.resize(reconstructed_img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC) # Reconstructing the original image by upsampling
    
    reconstructed_img = cv2.add(reconstructed_img, error_image[j]) # Adding the error image 
    cv2.imshow('reconstructed_img ' + str(j), reconstructed_img)

cv2.waitKey(0)
cv2.destroyAllWindows()




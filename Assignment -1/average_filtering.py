import cv2
import numpy as np

pic = cv2.imread('Lenna.png') # Loading original image
gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY) # Convering image in grayscale

rows,columns = gray.shape
#print("Shape of gray",r,c)

final_matx_3x3 = np.zeros([rows,columns])
#print("3x3",final_matx_3x3.shape)

# For 3 * 3

kernal_matx_3x3 = np.ones(3)/3 
#print(kernal_matx)

kernal_length = len(kernal_matx_3x3)

for i in range(rows):
    for j in range(columns-(kernal_length-1)):
        sum = 0
        for k in range(kernal_length):
            sum = sum + (gray[i][j+k] * kernal_matx_3x3[k])
        final_matx_3x3[i][j+(kernal_length//2)] = sum  # Generalizing the pixel value based on kernal length
            
for p in range(columns):
    for q in range(rows-((kernal_length-1))):
        sum = 0
        for r in range(kernal_length):
            sum = sum + (final_matx_3x3[q+k][p] * kernal_matx_3x3[r])
        final_matx_3x3[q+(kernal_length//2)][p] = sum  # Generalizing the pixel value based on kernal length
        
final_matx_3x3 = final_matx_3x3.astype(np.uint8)

## For 5 * 5
kernal_matx_5x5 = np.ones(5)/5 

final_matx_5x5 = np.zeros([rows,columns])

kernal_length = len(kernal_matx_5x5)

for i in range(rows):
    for j in range(columns-(kernal_length-1)):
        sum = 0
        for k in range(kernal_length):
            sum = sum + (gray[i][j+k] * kernal_matx_5x5[k])
        final_matx_5x5[i][j+(kernal_length//2)] = sum # Generalizing the pixel value based on kernal length
            
for p in range(columns):
    for q in range(rows-((kernal_length-1))):
        sum = 0
        for r in range(kernal_length):
            sum = sum + (final_matx_5x5[q+k][p] * kernal_matx_5x5[r])
        final_matx_5x5[q+(kernal_length//2)][p] = sum   # Generalizing the pixel value based on kernal length
        
final_matx_5x5 = final_matx_5x5.astype(np.uint8)


## For 7x7

kernal_matx_7x7 = np.ones(7)/7 

final_matx_7x7 = np.zeros([rows,columns])

kernal_length = len(kernal_matx_7x7)

for i in range(rows):
    for j in range(columns-(kernal_length-1)):
        sum = 0
        for k in range(kernal_length):
            sum = sum + (gray[i][j+k] * kernal_matx_7x7[k])
        final_matx_7x7[i][j+(kernal_length//2)] = sum # Generalizing the pixel value based on kernal length
            
for p in range(columns):
    for q in range(rows-((kernal_length-1))):
        sum = 0
        for r in range(kernal_length):
            sum = sum + (final_matx_7x7[q+k][p] * kernal_matx_7x7[r])
        final_matx_7x7[q+(kernal_length//2)][p] = sum   # Generalizing the pixel value based on kernal length
        
final_matx_7x7 = final_matx_7x7.astype(np.uint8)


## For 9x9

kernal_matx_9x9 = np.ones(9)/9 

final_matx_9x9 = np.zeros([rows,columns])

kernal_length = len(kernal_matx_9x9)

for i in range(rows):
    for j in range(columns-(kernal_length-1)):
        sum = 0
        for k in range(kernal_length):
            sum = sum + (gray[i][j+k] * kernal_matx_9x9[k])
        final_matx_9x9[i][j+(kernal_length//2)] = sum # Generalizing the pixel value based on kernal length
            
for p in range(columns):
    for q in range(rows-((kernal_length-1))):
        sum = 0
        for r in range(kernal_length):
            sum = sum + (final_matx_9x9[q+k][p] * kernal_matx_9x9[r])
        final_matx_9x9[q+(kernal_length//2)][p] = sum   # Generalizing the pixel value based on kernal length
        
final_matx_9x9 = final_matx_9x9.astype(np.uint8)


## For 11X11

kernal_matx_11x11 = np.ones(11)/11 

final_matx_11x11 = np.zeros([rows,columns])

kernal_length = len(kernal_matx_11x11)

for i in range(rows):
    for j in range(columns-(kernal_length-1)):
        sum = 0
        for k in range(kernal_length):
            sum = sum + (gray[i][j+k] * kernal_matx_11x11[k])
        final_matx_11x11[i][j+(kernal_length//2)] = sum # Generalizing the pixel value based on kernal length
            
for p in range(columns):
    for q in range(rows-((kernal_length-1))):
        sum = 0
        for r in range(kernal_length):
            sum = sum + (final_matx_11x11[q+k][p] * kernal_matx_11x11[r])
        final_matx_11x11[q+(kernal_length//2)][p] = sum   # Generalizing the pixel value based on kernal length
        
final_matx_11x11 = final_matx_11x11.astype(np.uint8)

while(True):
    cv2.imshow("Originl_pic", pic)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("After 3x3 Average Filter", final_matx_3x3)
    cv2.imshow("After 5x5 Average Filter", final_matx_5x5)
    cv2.imshow("After 7x7 Average Filter", final_matx_7x7)
    cv2.imshow("After 9x9 Average Filter", final_matx_9x9)
    cv2.imshow("After 11x11 Average Filter", final_matx_11x11)
    
    if cv2.waitKey(1) == 27:
       break

cv2.destroyAllWindows()


import cv2
import numpy as np
import math

def gaussian1D(m, s):
    r = np.random.normal(m, s, 1)  # s: standard deviation
    return r

def gaussian2D(m, s):
    r1 = np.random.normal(m, s, 1)  # s: standard deviation
    r2 = np.random.normal(m, s, 1)  # s: standard deviation
    return np.array([r1[0], r2[0]])


def calculateLikelihood(color, sigma):
    d = math.sqrt( (color[2]-255)**2 + (color[1]**2) + (color[0]**2) )
    return math.sqrt(1 / (2 * math.pi * sigma**2)) * math.exp(-(d**2) / (2 * sigma**2))

def display(numParticles, particles, frame, color):
    for j in range(numParticles):
        a = int(particles[0, j])
        b = int(particles[1, j])
        cv2.circle(frame, (a, b), 2, color, -1)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
	
cap = cv2.VideoCapture('D:\Projects\CV\CV_assignment\Person.wmv')

imgH = 480
imgW = 640

targetColor = np.array([0, 0, 255])  # red
sigma = 70
posNoise = 15
velNoise = 5

numParticles = 5500

particles = np.array([
    np.random.randint(0, imgW, numParticles),
    np.random.randint(0, imgH, numParticles),
    3 * np.random.randn(numParticles) + 3,
    3 * np.random.randn(numParticles)
])

weights = np.ones(numParticles) / numParticles
predMat = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])


if not cap.isOpened():
    print("Error opening video stream or file")
	
frameCount = 0
while cap.isOpened():
    ret, frame = cap.read()
    rawFrame = np.copy(frame)
    if ret:
        frameCount += 1
        if frameCount < 30:
            continue

        for i in range(numParticles):
            x = int(particles[0, i])
            y = int(particles[1, i])
            if x >= 0 and x < imgW and y >= 0 and y < imgH:
                pixel_color = frame[y, x]
                color_likelihood = calculateLikelihood(pixel_color, sigma)
                weights[i] = color_likelihood

        weights /= np.sum(weights)

        indices = np.random.choice(np.arange(numParticles), size=numParticles, replace=True, p=weights)
        particles = particles[:, indices]

        weights = np.ones(numParticles) / numParticles

        display(numParticles, particles, frame, (255, 0, 0))

        for i in range(numParticles):
            particles[:2, i] = np.dot(predMat[:2, :2], particles[:2, i]) + gaussian2D(0, posNoise)
            particles[2:, i] = np.dot(predMat[2:, 2:], particles[2:, i]) + gaussian2D(0, velNoise)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

img = cv2.imread('picture_6.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

counter, hierarchy = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, counter, -1, (0, 255, 0), 2)
cv2.imshow('res', img)

print(hierarchy)
while True:
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()

import cv2
import numpy as np

img = cv2.imread('picture_4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.boxFilter(gray, -1, (5, 5))

ret, binary = cv2.threshold(
    gray, 205, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

counter, hierarchy = cv2.findContours(
    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

epsilon = 0.001 * cv2.arcLength(counter[0], True)
n = len(counter)
approx_counter = []

for i in range(n):
    approx = cv2.approxPolyDP(counter[i], epsilon, True)
    approx_counter.append(approx)

cv2.drawContours(img, approx_counter, -1, (0, 255, 0), 2)

while True:
    cv2.imshow('res', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()

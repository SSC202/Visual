import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('apple_tree.jpg')
plate = cv2.imread('apple.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

w, h = plate_gray.shape[::-1]
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    method = eval(meth)

    res = cv2.matchTemplate(img_gray, plate_gray, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1]+h)
    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
    
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('ERes Image'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
    

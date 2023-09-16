import cv2

event = [i for i in dir(cv2) if 'EVENT' in i]
print(event)
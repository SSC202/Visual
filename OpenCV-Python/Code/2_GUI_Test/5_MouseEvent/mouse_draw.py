import cv2
import numpy as np

# 左键按下时变为True
drawing = False
# 模式选择
mode = False
ix,iy = -1,-1

"""
    鼠标事件回调函数
"""
def draw(event,x,y,flags,param):
    global drawing,mode,ix,iy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    # event 检测鼠标移动，flags 检测鼠标左键是否按下
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if mode == True:
            cv2.circle(img,(x,y),10,(0,255,0),-1,cv2.LINE_AA)
        elif mode == False:
            cv2.rectangle(img,(x-10,y-10),(x+10,y+10),(0,0,255),-1,cv2.LINE_AA)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow('img')
cv2.setMouseCallback('img',draw)

while(1):
    cv2.imshow('img',img)
    key = cv2.waitKey(1)
    if key == ord('m'):
        mode = not mode
    elif key == 27:
        break

cv2.destroyAllWindows()
    
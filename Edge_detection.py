import cv2
import numpy as np

def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (11, 11), 10)
    img_canny = cv2.Canny(img_blur, 0, 65) #img_blur is the threshold, 0
    kernel = np.ones((19, 19))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=4)
    img_erode = cv2.erode(img_dilate, kernel, iterations=4)
    return img_erode

def draw_contours(img):
    contours, hierarchies = cv2.findContours(process(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.004 * peri, True)
    cv2.drawContours(img, [approx], -1, (255, 255, 0), 2)

path = r'C:\Users\Asbjo\Documents\GitHub\BO23EB-08\Resources\testbilde.jpg'
img = cv2.imread(path)
h, w, c = img.shape

draw_contours(img)


img = cv2.resize(img, (w // 2, h // 2))

cv2.imshow("Image", img)
cv2.waitKey(0)
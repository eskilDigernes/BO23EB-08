import cv2
import numpy as np

def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (11,11), 7)
    img_canny = cv2.Canny(img_blur, 0,70)
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
   
   

img = cv2.imread(r"Resources\blackspots\iphone\IMG_3268_test.jpg")
h, w, c = img.shape

img = cv2.resize(img, (w // 2, h // 2))

draw_contours(img)


exampleimage=img

cv2.imshow("Image", exampleimage)
cv2.waitKey(0)



# img = cv2.imread(r"Resources\blackspots\iphone\blackspot_test.jpg")
# h, w, c = img.shape

# img = cv2.resize(img, (w // 2, h // 2))

# draw_contours(img)

# exampleimage2=img

# cv2.imshow("Image2", exampleimage2)
# cv2.waitKey(0)





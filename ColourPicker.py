import  cv2
import numpy  as  np


def empty(a): # empty function for trackbars
    pass

def stackImages(scale, imgArray): # see ch6 for explanation
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver=hor
    return ver

#path = 'resources/Full_Image_set/IMG_7434.jpg'
#path = 'resources/Full_Image_set/IMG_3282.jpg'
#path = 'resources/Full_Image_set/IMG_7428.jpg'
path = 'resources/Full_Image_set/IMG_3260.jpg'




cv2.namedWindow('TrackBars') # create a window for trackbars
cv2.resizeWindow('TrackBars', 640, 240) # resize the window to 640x240

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# "Important! Define these values FIRST.
# find the values for your image using the trackbars. and then place them here.
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

cv2.createTrackbar('Hue Min', 'TrackBars', 165, 179, empty)
cv2.createTrackbar('Hue Max', 'TrackBars', 179, 179, empty)
cv2.createTrackbar('Sat Min', 'TrackBars',119, 255, empty)
cv2.createTrackbar('Sat Max', 'TrackBars', 255, 255, empty)
cv2.createTrackbar('Val Min', 'TrackBars', 41, 255, empty)
cv2.createTrackbar('Val Max', 'TrackBars', 243, 255, empty)

while True: # create a loop to keep the trackbars open
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)       # convert the image to HSV
    h_min = cv2.getTrackbarPos('Hue Min', 'TrackBars')  # get the value of the trackbar each time it moves
    h_max = cv2.getTrackbarPos('Hue Max', 'TrackBars')
    s_min = cv2.getTrackbarPos('Sat Min', 'TrackBars')
    s_max = cv2.getTrackbarPos('Sat Max', 'TrackBars')
    v_min = cv2.getTrackbarPos('Val Min', 'TrackBars')
    v_max = cv2.getTrackbarPos('Val Max', 'TrackBars')
    print(h_min, h_max,s_min,s_max,v_min, v_max)        # prints the values to the consol

    lower = np.array([h_min, s_min, v_min])             # create an array with the lower values
    upper = np.array([h_max, s_max, v_max])             # create an array with the upper values
    mask = cv2.inRange(imgHSV, lower, upper)            # create a mask with the lower and upper values
    imgResult = cv2.bitwise_and(img, img, mask=mask)    # create a new image with the mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find the contours


    for cnt in contours: # loop through the contours
        x, y, w, h = cv2.boundingRect(cnt) # get the x, y, width and height of the contour
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # draw a rectangle around the contour
    # cv2.imshow('Original', img)                       # show the original image, ect.
    # cv2.imshow("HSV", imgHSV)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Result", imgResult)



    imgStack = stackImages(.2, ([img, imgHSV, mask]))
    cv2.imshow("Stacked Images", imgStack)
    # cv2.imshow('Original', img)


    cv2.waitKey(1)                                       # wait for 1ms, CRUCIAL while in a loop, otherwise it will freeze.
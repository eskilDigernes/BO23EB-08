# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# #path = 'resources/redspots/iPhone/rediPhone1.png'
# path = 'resources/CoveredAnodes/A1.png'
# img = cv2.imread(path)
# cv2.imshow('Original', img)
#
# colour = ('b','g','r')          # create a tuple of colours
# for i,col in enumerate(colour):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# #plt.title('Image with airburn')
# plt.grid(True, linestyle='--')
# plt.show()
#
#
#
# cv2.waitKey(0)
#

import numpy as np
import matplotlib.pyplot as plt

x = np.logspace(-2, 2, 100000)
y = np.sqrt((1.28 - 1.28*np.cos(x))/(1.36 - 1.2*np.cos(x)))





plt.plot(x, y),plt.grid(True, linestyle='--')
plt.show()








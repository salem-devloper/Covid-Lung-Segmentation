#appeler les package:


import numpy as np

import cv2

import imutils
import os

#lire l'image:
path = "E:/2 MASTER/Memoire/07-06-2021 (croped)/croped pneumonia/dataset/croped_lung"
out = "E:/2 MASTER/Memoire/07-06-2021 (croped)/croped pneumonia/dataset"

img = cv2.imread(path + "/PNEUMONIA(1).jpg")

#Assurez-vous que le type de image est niveau de gris: 
gray = cv2.cvtColor (img, cv2.COLOR_RGB2GRAY)

#seuiller l'image, puis effectuer une série d'érosions + #dilatations pour éliminer les petites zones de bruit

thresh = cv2.threshold (gray, 45, 255, cv2. THRESH_BINARY) [1]

thresh = cv2.erode (thresh, None, iterations=2)

thresh = cv2.dilate (thresh, None, iterations=2)

#trouver des contours dans l'image seuillée, puis saisir le plus grand: 
cnts = cv2.findContours (thresh.copy (), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
cnts = imutils.grab_contours (cnts)

c= max(cnts, key=cv2.contourArea)

#trouver les points extrêmes:

extLeft = tuple(c[c[:, :, 0].argmin()][0])

extRight= tuple (c[c[:, :, 0].argmax()][0])

extTop = tuple(c[c[:, :, 1].argmin()][0])

extBot = tuple(c[c[:, :, 1].argmax()][0])

#création de recadrage d'image: 
ADD_PIXELS = 0

new_img= img [extTop[1]-ADD_PIXELS: extBot[1] +ADD_PIXELS, extLeft[0]-ADD_PIXELS: extRight[0]+ADD_PIXELS].copy()

cv2.imwrite(os.path.join(out,"PNEUMONIA(1).jpg"),new_img)
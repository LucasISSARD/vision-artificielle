# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 08:25:15 2022

@author: Lucas ISSARD

Implémente l'algorithme de Viola & Jones pour la détection de voitures
"""
# Librairies
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Import de l'image
img = cv2.imread('test1_3.png', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Import du classifieur pré-entraîné pour la détection de voitures
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml') 

# Détection des voitures dans notre image
cars = car_cascade.detectMultiScale(img_gray, scaleFactor = 1.03, minNeighbors = 6)

# Dessin des bounding boxes
for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

# Affichage de l'image
cv2.imshow('test.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
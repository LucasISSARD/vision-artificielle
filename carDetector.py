# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 08:25:15 2022

@author: Lucas ISSARD

Implémente l'algorithme de Viola & Jones pour la détection de voitures
"""

# Librairies
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Chemin de la vidéo ( /!\ sur Windows, remplacer les \ par des / sans oublier le / final)
#video_path = "D:/Documents/GitHub/vision-artificielle/dataset/2015/"
#video_path = "D:/Documents/GitHub/vision-artificielle/dataset/Road/"
video_path = "D:/Documents/GitHub/vision-artificielle/dataset/Road2/"

# Import du classifieur pré-entraîné pour la détection de voitures
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml') 

counter = 200                           # Première image à traiter
files = next(os.walk(video_path))[2]    # Compte le nombre de fichiers dans le video_path
while counter < len(files):             # Pour toutes les images du video_path
    # Acquisition des images dans la vidéo
    zero_pad = ""
    for i in range(10 - len((str)(counter))):
        zero_pad = zero_pad + "0"
    img_path = video_path + zero_pad + (str)(counter) + ".png"
    print(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détection des voitures dans notre image
    cars = car_cascade.detectMultiScale(img_gray, scaleFactor = 1.04, minNeighbors = 8, minSize=(20, 20), maxSize=(150, 150))

    # Dessin des bounding boxes
    i = 0
    for (x,y,w,h) in cars:
        cv2.drawMarker(img,(x+round(w/2),y+round(w/2)),color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=2)
        cv2.putText(img, (str)(i), (x+round(w/2)+8,y+round(w/2)+8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0,0,255), thickness=2)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        i = i + 1

    # Affichage de l'image
    cv2.imshow("video", img)
    counter = counter + 1
    cv2.waitKey(1)
    time.sleep(0.1)

# Fermeture des fenêtres
cv2.destroyAllWindows()
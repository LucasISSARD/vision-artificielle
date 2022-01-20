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

frame = 200                           # Première frame à traiter
files = next(os.walk(video_path))[2]  # Compte le nombre de fichiers dans le video_path
while frame < len(files):             # Pour toutes les frames de la vidéo (pour tous les fichiers du video_path)
    # Acquisition des images dans la vidéo (calcul du nom du fichier désiré + acquisition de l'image dans img)
    zero_pad = ""
    for i in range(10 - len((str)(frame))):
        zero_pad = zero_pad + "0"
    img_path = video_path + zero_pad + (str)(frame) + ".png"
    print("Img_path : ", img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Détection des voitures dans l'image (1.04 - 8 - 30 - 180)
    cars = car_cascade.detectMultiScale(img, scaleFactor = 1.04, minNeighbors = 8, minSize=(20, 20), maxSize=(180, 180))

    # Suppression des doublons et affichage des bounding boxes
    # TODO : améliorer en fusionnant les zones plutôt qu'en supprimant la plus petite
    i = 0
    for (x,y,w,h) in cars:
        I = 0
        for (X, Y, W, H) in cars:
            # Si deux détections se chevauchent (si son centre est dans la box d'une autre), on supprime la plus petite (TODO)
            if x+round(w/2) >= X and x+round(w/2) <= X+W and y+round(w/2) >= Y and y+round(w/2) <= Y+H and x != X and y != Y:
                if cars[i][2] < cars[I][2]:
                    cars[i] = (0,0,0,0)
                else:
                    cars[I] = (0,0,0,0)
            I = I + 1
        if cars[i][0] != 0 and cars[i][1] != 0:
            cv2.drawMarker(img,(x+round(w/2),y+round(w/2)),color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=2)
            #cv2.putText(img, (str)(i), (x+round(w/2)+8,y+round(w/2)+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,255), thickness=1)
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        print(i, cars[i])
        i = i + 1
    print("-------------")

    # TODO : Suivi des voitures
    # Si une voiture est détecté sur plusieurs frames de suite, on appelle un algorithme de tracking qui va la suivre le plus longtemps possible
    # On utilise l'algorithme CSRT (corrélation) car c'est celui qui est le plus précis (et le plus lent mais c'est pas grave)
    


    # Affichage de l'image
    cv2.imshow("video", img)

    frame = frame + 1
    cv2.waitKey(1)
    time.sleep(0.1)

# Fermeture des fenêtres
cv2.destroyAllWindows()
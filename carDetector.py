# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 08:25:15 2022

@author: Lucas ISSARD

Implémente l'algorithme de Viola & Jones pour la détection des voitures
Implémente l'algorithme CSRT pour le suivi des voitures
"""
## TODO List :
# - Améliorer la gestion des doublons dans la détection des voitures
# - Gérer l'ajout automatique de nouveaux traceurs
# - Gérer la suppression automatique des traceurs sortis du champ

# Librairies
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fonctions
def acqFrame (video_path, frame):
    # Calcul du nom du fichier désiré
    zero_pad = ""
    for i in range(10 - len((str)(frame))):
        zero_pad = zero_pad + "0"
    img_path = video_path + zero_pad + (str)(frame) + ".png"
    print("Img_path : ", img_path)
    # Acquisition de l'image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return img

def detectCars (img): # TODO : améliorer la gestion des doublons en fusionnant les zones plutôt qu'en supprimant la plus petite
    # Détection des voitures dans l'image
    cars = car_cascade.detectMultiScale(img, scaleFactor = 1.04, minNeighbors = 8, minSize=(20, 20), maxSize=(180, 180))

    # Suppression des doublons et affichage des bounding boxes
    i = 0
    for (x,y,w,h) in cars:
        I = 0
        for (X, Y, W, H) in cars:
            # Si deux détections se chevauchent (si son centre est dans la box d'une autre), on supprime la plus petite
            if x+round(w/2) >= X and x+round(w/2) <= X+W and y+round(w/2) >= Y and y+round(w/2) <= Y+H and x != X and y != Y:
                if cars[i][2] < cars[I][2]:
                    cars[i] = (0,0,0,0)
                else:
                    cars[I] = (0,0,0,0)
            I = I + 1
        if cars[i][0] != 0 and cars[i][1] != 0:
            cv2.drawMarker(img,(x+round(w/2),y+round(w/2)),color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=2)
            cv2.putText(img, (str)(i), (x+round(w/2)+8,y+round(w/2)+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,255), thickness=1)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        print(i, cars[i])
        i = i + 1
    print("-------------")
    return cars

# Paramètres
video_path = "D:/Documents/GitHub/vision-artificielle/dataset/Road2/"   # Chemin de la vidéo ( /!\ sur Windows, remplacer les \ par des / sans oublier le / final )
#video_path = "D:/Documents/GitHub/vision-artificielle/dataset/2011_09_26_2/2011_09_26_drive_0005_sync/image_02/data/"
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')              # Classifieur pré-entraîné
first_frame = 200                                                       # Première frame à traiter
last_frame = len(next(os.walk(video_path))[2])                          # Dernière frame à traiter (fin de la vidéo)

first = True

trackers = cv2.legacy.MultiTracker_create() # Crée le traceur multiple

# Programme principal
frame = first_frame
while frame < last_frame:

    # Acquisition de l'image
    img = acqFrame(video_path, frame)

    # Détection des voitures
    cars = detectCars(img)


    # Suivi des voitures
    # TODO : Gérer la suppression des trackers sortis du champ
    #        Gérer l'ajout automatique de nouveaux trackers
    if first == True:
        n = 0
        ofs = 20    # Réduit la taille de la zone
        box = (cars[n][0]+ofs, cars[n][1]+ofs, cars[n][2]-2*ofs, cars[n][3]-2*ofs)   # Trace la box
        trackers.add(cv2.legacy.TrackerCSRT_create(), img, box)                      # Ajoute un tracker
        n=3
        ofs = 10
        box = (cars[n][0]+ofs, cars[n][1]+ofs, cars[n][2]-2*ofs, cars[n][3]-2*ofs)   # Trace la box
        trackers.add(cv2.legacy.TrackerCSRT_create(), img, box)                      # Ajoute un tracker
        first = False
    else :
        success, boxes = trackers.update(img) # Met à jour les traceurs
        if success:
            for bbox in boxes:  # Affiche toutes les boxes
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                x = int(p1[0] + (p2[0] - p1[0])/2)
                y = int(p2[1] + (p1[1] - p2[1])/2)
                cv2.drawMarker(img,(x,y),color=(255,0,0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
        else:
            print("Echec du suivi")


    # Affichage de l'image
    cv2.imshow("video", img)

    frame = frame + 1
    cv2.waitKey(1)
    time.sleep(0.1)

# Fermeture des fenêtres
cv2.destroyAllWindows()
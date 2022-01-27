# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 08:25:15 2022

@author: Lucas ISSARD

Implémente l'algorithme de Viola & Jones pour la détection des voitures
Implémente l'algorithme CSRT ou MedianFlow pour le suivi des voitures
"""
## TODO List :
# - Gérer la création des traceurs
# - Gérer la suppression des traceurs

# Librairies
import os
from telnetlib import NOP
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paramètres globaux
show_detected = False   # Afficher sur l'image toutes les entités détectées (rouge)
show_tracked = True     # Afficher sur l'image les voitures en train d'être suivies (bleu)

# Paramètres du détecteur
video_path = "D:/Documents/GitHub/vision-artificielle/dataset/road/"    # Chemin de la vidéo ( /!\ sur Windows, remplacer les \ par des / sans oublier le / final )
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')              # Classifieur pré-entraîné
first_frame = 200                                                       # Première frame à traiter
last_frame = len(next(os.walk(video_path))[2])                          # Dernière frame à traiter (fin de la vidéo)

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

def detectCars (img):
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
        if cars[i][0] != 0 and cars[i][1] != 0 and show_detected == True:
            cv2.drawMarker(img,(x+round(w/2),y+round(w/2)),color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=2)
            cv2.putText(img, (str)(i), (x+round(w/2)+8,y+round(w/2)+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,255), thickness=1)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        print(i, cars[i])
        i = i + 1
    print("-------------")
    return cars

# Variables
first = True
trackers = []   # Liste des traceurs
liste = []      # Liste contenant les voitures détectées, format : [ x , y , w  , h , on/off ]

# Programme principal
frame = first_frame
while frame < last_frame:

    # Acquisition de l'image
    img = acqFrame(video_path, frame)

    # Détection des voitures
    cars = detectCars(img)

    # Suivi des voitures
    if first == True:
        first = False
    else :

        # Détection des NOUVELLES voitures
        for i in range(len(cars)-1):    # Pour toutes les "voitures" fraîchement détectées
            ignore = False

            # Si la voiture en cours d'étude est un doublon (valeurs mises à 0), on l'ignore
            if cars[i][2] == 0:
                ignore = True
                break

            # Si la voiture en cours d'étude est déjà dans la liste, on l'ignore
            th1 = 30
            for k in range(len(liste)):
                if cars[i][0] >= liste[k][0]-th1 and cars[i][0] <= liste[k][0]+th1 and cars[i][1] >= liste[k][1]-th1 and cars[i][1] <= liste[k][1]+th1:
                    ignore = True
            
            if ignore == False:
                for j in range(len(cars_history)-1):
                    # Si c'est la deuxième fois de suite qu'on détecte cette voiture
                    th = 20     # seuil
                    if cars[i][0] >= cars_history[j][0]-th and cars[i][0] <= cars_history[j][0]+th and cars[i][1] >= cars_history[j][1]-th and cars[i][1] <= cars_history[j][1]+th:
                        # Alors c'est bien une nouvelle voiture détectée, on l'ajoute à la liste et on lui colle un traceur
                        liste.append([cars[i][0], cars[i][1], cars[i][2], cars[i][3], 1])           # On l'ajoute à la liste des voitures vraiment détectées
                        ofs = int(0.2*cars[i][2])                                                   # On réduit la taille de la box de 20% pour aider la détection
                        box = (cars[i][0]+ofs, cars[i][1]+ofs, cars[i][2]-2*ofs, cars[i][3]-2*ofs)  # On trace sa box
                        trackers.append(cv2.legacy.TrackerMedianFlow_create())
                        trackers[-1].init(img, box)     # -1 = dernier tracker de la liste
                        break

        print("TRACKERS = ", trackers)
        print("LISTE = ", liste)

        tracker_to_remove = []
        list_to_remove = []
        for i in range(len(trackers)):
            success, box = trackers[i].update(img) # Met à jour les traceurs
            if success:
                if show_tracked == True:
                        p1 = (int(box[0]), int(box[1]))
                        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                        x = int(p1[0] + (p2[0] - p1[0])/2)
                        y = int(p2[1] + (p1[1] - p2[1])/2)
                        cv2.drawMarker(img,(x,y),color=(255,0,0), markerType=cv2.MARKER_CROSS, thickness=2)
                        cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
            else:
                # TODO : Améliorer la gestion de la perte d'objets pour éviter de vider complètement la liste à chaque fois qu'on perd qu'un seul traceur
                print("Echec du suivi tacker n°", i)
                tracker_to_remove.append(trackers[i])   # Ajoute le traceur actuel dans la liste des traceurs à supprimer
                list_to_remove.append(liste[i])         # Ajoute l'élément de la liste concerné dans la liste des éléments de la liste à supprimer (wow)

        # Et on supprime !
        for j in tracker_to_remove:
            trackers.remove(j)
        for j in list_to_remove:
            liste.remove(j)

        
            
    cars_history = cars     # On stock les voitures actuelles dans l'historique

    # Affichage de l'image
    cv2.imshow("video", img)

    frame = frame + 1
    cv2.waitKey(1)
    time.sleep(0.1)

# Fermeture des fenêtres
cv2.destroyAllWindows()
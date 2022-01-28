# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 08:25:15 2022

@author: Lucas ISSARD

Implémente l'algorithme de Viola & Jones pour la détection des voitures
Implémente l'algorithme MedianFlow pour le suivi des voitures
"""
## TODO List :
# - Ajuster les paramètres pour un fonctionnement optimal

# Librairies
import os
from telnetlib import NOP
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

import Local_config
import Fonctions

# Paramètres
show_detected = False   # Afficher sur l'image toutes les entités détectées (rouge)
show_tracked = True     # Afficher sur l'image les voitures en train d'être suivies (vert)
show_rectangle = False  # Afficher sur l'image les rectangles entourant les voitures suivies (vert)
video_path = Local_config.video_path_02   # Chemin de la vidéo ( /!\ sur Windows, remplacer les \ par des / sans oublier le / final )
video_paths= Local_config.video_path_03
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')              # Choix du classifieur pré-entraîné
first_frame = 260                                                     # Première frame à traiter
last_frame = len(next(os.walk(video_path))[2])                          # Dernière frame à traiter (fin de la vidéo)

vect_T1,vect_T2,vect_D1,vect_D2,mat_Rect_1,mat_Rect_2,mat_K1,mat_K2,mat_R1,mat_R2 = Fonctions.calib('000020.txt') # Chargement des matrices


# Fonctions
def acqFrame (video_path, frame):
    # Calcul du nom du fichier désiré
    zero_pad = ""
    for i in range(10 - len((str)(frame))):
        zero_pad = zero_pad + "0"
    img_path = video_path + zero_pad + (str)(frame) + ".png"
    #print("Img_path : ", img_path)
    # Acquisition de l'image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return img

def detectCars (img):
    # Détection des voitures dans l'image
    cars = car_cascade.detectMultiScale(img, scaleFactor = 1.04, minNeighbors = 8, minSize=(20, 20), maxSize=(180, 180))

    #print("CARS (voitures détectées) =")
    i = 0
    for (x,y,w,h) in cars:
        I = 0
        for (X, Y, W, H) in cars:
            # Si deux détections se chevauchent sur UNE MEME IMAGE (si son centre est dans la box d'une autre), on supprime la plus petite
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
        #print(i, cars[i])
        i = i + 1

    return cars

def topleft2center (liste):
    carsC = []
    for (x,y,w,h) in liste:
        carsC.append([x + round(w/2), y + round(h/2)])
    return carsC

def trackCars (cars, cars_history):
    # Détection des NOUVELLES voitures
    for i in range(len(cars)):    # Pour toutes les "voitures" fraîchement détectées
        ignore = False

        # Si la voiture en cours d'étude est un doublon (valeurs mises à 0), on l'ignore
        if cars[i][2] == 0:
            ignore = True
            break

        # Si la voiture en cours d'étude est déjà dans la liste, on l'ignore
        th1 = 30
        x_cent = cars[i][0]+round(cars[i][2]/2)
        y_cent = cars[i][1]+round(cars[i][3]/2)
        for k in range(len(liste)):
            if x_cent >= liste[k][0]-th1 and x_cent <= liste[k][0]+liste[k][2]+th1 and y_cent >= liste[k][1]-th1 and y_cent <= liste[k][1]+liste[k][3]+th1:
                ignore = True

        if ignore == False:
            for j in range(len(cars_history)):
                th = 20     # seuil
                # Si c'est la deuxième fois de suite qu'on détecte cette voiture
                if cars[i][0] >= cars_history[j][0]-th and cars[i][0] <= cars_history[j][0]+th and cars[i][1] >= cars_history[j][1]-th and cars[i][1] <= cars_history[j][1]+th:
                    # Alors c'est bien une nouvelle voiture détectée, on l'ajoute à la liste et on lui colle un traceur
                    ofs = int(0.3*cars[i][2])                                                   # On réduit la taille de la box de 30% pour aider la détection
                    liste.append([cars[i][0]+ofs, cars[i][1]+ofs, cars[i][2]-2*ofs, cars[i][3]-2*ofs])           # On l'ajoute à la liste des voitures vraiment détectées
                    box = (cars[i][0]+ofs, cars[i][1]+ofs, cars[i][2]-2*ofs, cars[i][3]-2*ofs)  # On trace sa box
                    trackers.append(cv2.legacy.TrackerMedianFlow_create())
                    trackers[-1].init(img, box)     # -1 = dernier tracker de la liste
                    break

    # Suivi des voitures
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
                    liste[i] = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]           # Mise à jour de la liste
                    cv2.drawMarker(img,(x,y),color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=2)
                    if show_rectangle == True:
                        cv2.rectangle(img, p1, p2, (0, 255, 0), 2, 1)
        else:
            # TODO : Améliorer la gestion de la suppression, c'est pas très propre ça
            tracker_to_remove.append(trackers[i])   # Ajoute le traceur actuel dans la liste des traceurs à supprimer
            list_to_remove.append(liste[i])         # Ajoute l'élément de la liste concerné dans la liste des éléments de la liste à supprimer (wow)
    # Suppression des trackers perdus
    for j in tracker_to_remove:
        trackers.remove(j)
    for j in list_to_remove:
        liste.remove(j)

    return liste

# Variables
trackers = []   # Liste des traceurs
liste = []      # Liste contenant les voitures détectées, format : [ x , y , w  , h , on/off ]

# Initialisation

w = 20   # taille du masque de corrélation (2*w+1)*(2*w+1)
seuil = 0.2 # seuil de corrélation

frame = first_frame
img = acqFrame(video_path, frame)   # Acquisition de la première image
cars_history = detectCars(img)      # Détection des voitures
cv2.imshow("video", img)            # Affichage de la première image
frame = frame + 1
cv2.waitKey(1)
time.sleep(0.01)

# Programme principal
while frame < last_frame:

    u=[]
    v=[]

    img = acqFrame(video_path, frame)       # Acquisition de l'image
    img_2=acqFrame(video_paths, frame)
    cars = detectCars(img)                  # Détection des voitures
    liste = trackCars(cars, cars_history)   # Suivi des voitures
    cars_history = cars                     # Stockage des voitures actuelles dans l'historique
    out = topleft2center(liste)             # Calcul de la sortie (liste des positions (x,y) des voitures)

    for e in out:
        u.append(e[0])
        v.append(e[1])


    Dist=Fonctions.correspondance_sans_epipo(u,v,img,img_2,w,seuil,vect_T1,vect_T2)


    # Affichage des résultats
    cv2.imshow("video", img)
    plt.pause(0.001)
    plt.clf()

    print("Dist= ",Dist)

    # Inter-image
    frame = frame + 1


    cv2.waitKey(1)
    time.sleep(0.05)

# Fermeture des fenêtres
cv2.destroyAllWindows()
plt.close()
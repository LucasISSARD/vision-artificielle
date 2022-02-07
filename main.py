# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 08:25:15 2022

@author: Lucas ISSARD, Etienne TORZINI, Quentin BERNARD

Implémente la détection de véhicule, le suivi de voiture et la Triangulation :
- Implémente l'algorithme de Viola & Jones pour la détection des voitures
- Implémente l'algorithme MedianFlow pour le suivi des voitures
- Implémente la mise en correspondance de 2 images stéréoscopiques rectifiées
- Implémente la triangulation avec affichage en 2D

Les fonctions  sont dans le fichier Fonctions. Pour exécuter le code, lancer le main.
Le fichier Local_config contient les chemins des images et du fichier calibration.
"""

# Bibliothèques
import os
import cv2
import time
from matplotlib import pyplot as plt
import Fonctions
import Local_config

# Chemins
video_path_cam_2 = Local_config.path + 'image_02/data/' # Images de la caméra 2
video_path_cam_3 = Local_config.path + 'image_03/data/' # Images de la caméra 3

# Paramètres
first_frame = 260                                       # Première frame à traiter
last_frame  = len(next(os.walk(video_path_cam_2))[2])   # Dernière frame à traiter (fin de la vidéo)
w = 20                                                  # Taille du masque de corrélation (2*w+1)*(2*w+1)
seuil = 0.2                                             # Seuil de corrélation

# Chargement des matrices de calibration
vect_T1, vect_T2, vect_D1, vect_D2, mat_Rect_1, mat_Rect_2, mat_K1, mat_K2, mat_R1, mat_R2 = Fonctions.calib()

# Programme principal
frame = first_frame                                 # Départ à first_frame
img_2 = Fonctions.acqFrame(video_path_cam_2, frame) # Acquisition de la première image (caméra 2)
cars_history = Fonctions.detectCars(img_2)          # Détection des voitures
cv2.imshow("video", img_2)                          # Affichage de la première image
frame = frame + 1                                   # Incrémentation du numéro de la frame en cours
cv2.waitKey(1)                                      # Attente
time.sleep(0.01)                                    #  |

while frame < last_frame :  # Pour toutes les images de first_frame à last_frame
    u=[]
    v=[]

    img_2 = Fonctions.acqFrame(video_path_cam_2, frame)     # Acquisition de l'image de la caméra 2
    img_3 = Fonctions.acqFrame(video_path_cam_3, frame)     # Acquisition de l'image de la caméra 3
    cars  = Fonctions.detectCars(img_2)                     # Détection des voitures
    liste = Fonctions.trackCars(cars, cars_history,img_2)   # Suivi des voitures
    cars_history = cars                                     # Stockage des voitures actuelles dans l'historique
    out = Fonctions.topleft2center(liste)                   # Calcul de la sortie (liste des positions (x,y) des voitures)

    for e in out:
        u.append(e[0])
        v.append(e[1])

    Dist = Fonctions.correspondance_sans_epipo(u, v, img_2, img_3, w, seuil, vect_T1, vect_T2)

    # Affichage des résultats
    cv2.imshow("video", img_2)
    plt.pause(0.001)
    plt.clf()
    print("Dist= ", Dist)

    # Traiement inter-image
    frame = frame + 1
    cv2.waitKey(1)
    time.sleep(0.05)

# Fermeture des fenêtres
cv2.destroyAllWindows()
plt.close()
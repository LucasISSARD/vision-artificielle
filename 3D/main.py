# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 08:25:15 2022

@author: Lucas ISSARD, Etienne TORZINI, Quentin BERNARD

Implémente la détection de véhicule, le suivi de voiture et la Triangulation :
- Implémente l'algorithme de Viola & Jones pour la détection des voitures
- Implémente l'algorithme MedianFlow pour le suivi des voitures
- Implémente la mise en correspondance de 2 images rectifiées
- Implémente la triangulation avec affichage 2D

Les fonctions sont dans le fichier Fonctions. Pour exécuter le code, lancer le main.
"""

# Librairies
import os
from telnetlib import NOP
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

import Local_config # Les chemins des fichiers de calibrations et des images sont dans ce fichier
import Fonctions # L'ensemble des fonctions sont dans ce fichier

# Paramètres
video_path_cam_2 = Local_config.video_path_02   # Chemin de la vidéo ( /!\ sur Windows, remplacer les \ par des / sans oublier le / final )
video_path_cam_3= Local_config.video_path_03
first_frame = 260                                                             # Première frame à traiter
last_frame = len(next(os.walk(video_path_cam_2))[2])                          # Dernière frame à traiter (fin de la vidéo)
vect_T1,vect_T2,vect_D1,vect_D2,mat_Rect_1,mat_Rect_2,mat_K1,mat_K2,mat_R1,mat_R2 = Fonctions.calib('000020.txt') # Chargement des matrices de calibration
w = 20   # taille du masque de corrélation (2*w+1)*(2*w+1)
seuil = 0.2 # seuil de corrélation

# Main
Fonctions.main(first_frame,last_frame,w,seuil,vect_T1,vect_T2,video_path_cam_3,video_path_cam_2)



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
Le fichier Local_config contient les chemins des images et du fichier calibration.
"""

import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


## Local config

chemin='C:/Users/quent/Desktop/Cours 5A/VISION_PROJET'
#video_path = "C:/Users/quent/Desktop/Cours 5A/VISION_PROJET/road/"    # Chemin de la vidéo ( /!\ sur Windows, remplacer les \ par des / sans oublier le / final )

video_path_02 = "C:/Users/quent/Desktop/Cours 5A/VISION_PROJET/2011_09_26/image_02/data/"
video_path_03 = "C:/Users/quent/Desktop/Cours 5A/VISION_PROJET/2011_09_26/image_03/data/"
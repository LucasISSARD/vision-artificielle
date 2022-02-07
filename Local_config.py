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

## Local config
path = 'H:/GitHub/vision-artificielle/2011_09_26/'      # /!\ TODO : Adaptez le chemin du dataset ici
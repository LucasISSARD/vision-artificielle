#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

import Local_config
import Fonctions_sup

### Main
img_filename_1,img_filename_2 = Fonctions_sup.image('000020_10.png') # Chargement des images
vect_T1,vect_T2,vect_D1,vect_D2,mat_Rect_1,mat_Rect_2,mat_K1,mat_K2,mat_R1,mat_R2 = Fonctions_sup.calib('000020.txt') # Chargement des matrices
img_i_1,img_i_2 = Fonctions_sup.Trace_Image(img_filename_1,img_filename_2) # Affiche les images et crée les figures


## Correpondance entre 2 points sur 2 images différentes avec la droite épipolaire
# Prenons le cas où l'on détecte sur image 2 et que l'on fait la correspondance sur l'image 3
# Calcul des matrices essentielles et fondamentales
E, F = Fonctions_sup.Calcul_matrice_E_F(mat_R1,mat_R2,vect_T1,vect_T2,mat_K1,mat_K2)
u,v = 363,197 # Test à une position
A,B,C = Fonctions_sup.Calcul_droite_epi(u,v,F) # calcul et affiche la droite épipolaire

w = 75   # taille du masque de corrélation (2*w+1)*(2*w+1)
seuil = 0.2 # seuil de corrélation
j_max,i_max = Fonctions_sup.cherche_point_droite_epi(w,seuil,A,B,C,img_i_1,img_i_2,u,v)


## Correpondance entre 2 points sur 2 images différentes sans la droite épipolaire
#
# w=10
# seuil=0.5

# u= [182,250,363,462,504,602,645,333,661,1014]
# v= [224,211,196,187,183,186,183,125,186,106]
#
# d_X=[]
# d_Z=[]
# dist=[]
# for i in range (np.size(u)):
#     i_max=Fonctions_sup.cherche_pts(img_i_1,img_i_2,u[i],v[i],w,seuil)
#     rep=Fonctions_sup.triangulation(u[i],i_max,vect_T1,vect_T2)
#
#     plt.figure(1)
#     plt.plot (u[i],v[i],'bo')
#
#     plt.figure(2)
#     plt.plot (i_max,v[i],'bo')

# ## Affichage dans un plant 2D

#     plt.figure(3)
#     plt.plot(rep[1],rep[2],'bo')

# Taille=140
# plt.figure(3)
# plt.plot(0,0,'mo')
# plt.grid()
# plt.axis([-Taille/2, Taille/2, -2, Taille])
# plt.title("Positions mesurées")
# plt.ylabel('Z (Profondeur) (m)')
# plt.xlabel('X (Largeur) (m)')

## Triangulation
Dist_cam_vehicule = Fonctions_sup.triangulation(u,j_max,vect_T1,vect_T2)
print(Dist_cam_vehicule)
plt.show()


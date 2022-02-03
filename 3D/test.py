#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from telnetlib import NOP
import time
import Local_config
import Fonctions




### Main
img_filename_1,img_filename_2 = Fonctions.image('000020_10.png') # Chargement des images
vect_T1,vect_T2,vect_D1,vect_D2,mat_Rect_1,mat_Rect_2,mat_K1,mat_K2,mat_R1,mat_R2 = Fonctions.calib('000020.txt') # Chargement des matrices
img_i_1,img_i_2 = Fonctions.Trace_Image(img_filename_1,img_filename_2) # Affiche les images et crée les figures


## Correpondance entre 2 points sur 2 images différentes sans la droite épipolaire
w=10
seuil=0.5
# u= [182,250,363,462,504,602,645,333,661,1014]
# v= [224,211,196,187,183,186,183,125,186,106]
Positions_vehicule = [[250,211,100,100,1],[363,196,100,100,1],[462,187,100,100,1],[504,183,100,100,1],[602,186,100,100,1],[645,183,100,100,1],[661,186,100,100,1]]
u,v=Fonctions.Extract_u_v(Positions_vehicule)
Fonctions.correspondance_sans_epipo(u,v,img_i_1,img_i_2,w,seuil,vect_T1,vect_T2) # Affiche resultat de correspondance et triangulation
plt.show()


#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import *

import Local_config
import Fonctions_sup

## Fonction

def param_to_vect(param,taille):
    list=np.zeros(taille)
    c=0
    for j in range(len(param)):
        if param[j]==" ":
            if param[j+1]=="-":
                list[c]=(float(param[j+1:j+14]))
            else:
                list[c]=(float(param[j+1:j+13]))
            c=c+1
    return list

def param_to_matrice(param,taille_w,taille_h):
    list=np.zeros((taille_w,taille_h))
    w=0
    h=0
    for j in range(len(param)):
        if param[j]==" ":
            if h == 3 :
                h=0
                w=w+1
            if param[j+1]=="-":
                list[w,h]=(float(param[j+1:j+14]))
            else:
                list[w,h]=(float(param[j+1:j+13]))
            h=h+1
    return list


## Chargement des images
# Charger les photos
img_filename_1 = Local_config.chemin+'/data_scene_flow/testing/image_2/000000_10.png'  # Load the photo
img_filename_2 = Local_config.chemin+'/data_scene_flow/testing/image_3/000000_10.png'  # Load the photo

# Photo 1
fig=plt.figure(figsize=(6,6))
img_i_1 = cv2.imread(img_filename_1)
plt.axis('off')
plt.imshow(cv2.cvtColor(img_i_1, cv2.COLOR_BGR2RGB))

# Photo 2
fig=plt.figure(figsize=(6,6))
img_i_2 = cv2.imread(img_filename_2)
plt.axis('off')
plt.imshow(cv2.cvtColor(img_i_2, cv2.COLOR_BGR2RGB))


#plt.show()

## Lecture de fichier de calibrage

calib_filename= Local_config.chemin+'/data_scene_flow_calib/testing/calib_cam_to_cam/000000.txt'

with open(calib_filename, "r") as filin:
    for i in range(18):
        filin.readline()
    S1 = filin.readline() # Taille image
    K1 = filin.readline() # Param calibrage
    D1 = filin.readline() # Distorsion
    R1 = filin.readline() # Matrice Rotation
    T1 = filin.readline() # Vect translation
    S_Rect1 = filin.readline() # Taille rectifier
    Rect_1 = filin.readline() # Matrice de rectification
    P_Rect_1 = filin.readline() # Matrice de Projection
    S2 = filin.readline() # Taille image
    K2 = filin.readline() # Param calibrage
    D2 = filin.readline() # Distorsion
    R2 = filin.readline() # Matrice Rotation
    T2 = filin.readline() # Vect translation
    S_Rect2 = filin.readline() # Taille rectifier
    Rect_2 = filin.readline() # Matrice de rectification
    P_Rect_2 = filin.readline() # Matrice de Projection

## Conversion en array (numpy)
vect_T2=param_to_vect(T2,3)
vect_T1=param_to_vect(T1,3)
vect_D1=param_to_vect(D1,5)
vect_D2=param_to_vect(D2,5)
mat_Rect_2 = param_to_matrice(Rect_2,3,3)
mat_Rect_1 = param_to_matrice(Rect_1,3,3)
mat_K1 = param_to_matrice(K1,3,3)
mat_K2 = param_to_matrice(K2,3,3)
mat_R1 = param_to_matrice(R1,3,3)
mat_R2 = param_to_matrice(R2,3,3)


## Appariment

#cv2.triangulatePoints()
## Rectification

#cv2.stereoRectify()

## Triangulation

##intercorelation


rep=Fonctions_sup.correlation_2D(img_i_1,img_i_2)



#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

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
plt.figure(1)
img_i_1 = cv2.imread(img_filename_1)
plt.axis('off')
plt.imshow(cv2.cvtColor(img_i_1, cv2.COLOR_BGR2RGB))

# Photo 2
plt.figure(2)
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


## Correpondance entre 2 points sur 2 images différentes
# Prenons le cas où l'on détecte sur image 2 et que l'on fait la correspondance sur l'image 3
# Calcul des matrices essentielles et fondamentales
R32 = mat_R2.dot(np.matrix.transpose(mat_R1))                              # Calcul de matrice de Rotations
t32 = vect_T2 - mat_R2.dot(np.matrix.transpose(mat_R1)).dot(vect_T1)             # Calcul de matrice de translation
t_x = np.array([[0,-t32[2],t32[1]],
                [t32[2],0,-t32[0]],
                [-t32[1],t32[0],0]])


E = t_x * R32                                                 # Calcul de la matrice Essentielle
F = np.matrix.transpose((np.linalg.inv(mat_K1))).dot(E).dot(np.linalg.inv(mat_K2))   # Calcul de la matrice Fondamentale



# Calcul de la droite épipolaire
u = int(input( " Enter value between 0 and 1242 >> "))
v = int(input( " Enter value between 0 and 375 >> "))
plt.figure(1)
plt.plot (u,v,'ro')
ABC = np.array([float(u),float(v),1.0]).dot(F)     # Droite épipolaire                                           #

A = ABC[0];B=ABC[1];C=ABC[2]
x= np.linspace(0,1200)
y= np.linspace(0,300)
plt.Line2D(ABC,'r' )

# Recherche du point correspondant sur l'image droite
w = 11        # taille du masque de corrélation (2*w+1)*(2*w+1)
seuil = 1e-29
sc_max=seuil
for j in range(w,np.size(img_i_2,0)-w):
    i = round(-(A*j+C)/B)    # round = arrondie
    if i > w + 1 and i < np.size(img_i_2,1)-w:
        sc = Fonctions_sup.correlation_2D(img_i_1[v-w:v+w,u-w:u+w,0],img_i_2[i-w:i+w,j-w:j+w,0])
        #print(sc)
        if sc > sc_max :
            sc_max = sc
            i_max = i
            j_max =j
            print("coucou")
if sc_max > seuil:
    plt.figure(2)
    plt.plot(j_max,i_max,'go')


## Rectification

#cv2.stereoRectify()

## Triangulation

plt.show()


import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import *
import scipy as sc

import Local_config


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



def calib(filename):
    calib_filename= Local_config.chemin+'/data_scene_flow_calib/testing/calib_cam_to_cam/'+filename
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

    return([vect_T1,vect_T2,vect_D1,vect_D2,mat_Rect_1,mat_Rect_2,mat_K1,mat_K2,mat_R1,mat_R2])

def image (img):
    # Charger les photos
    img_filename_1 = Local_config.chemin+'/data_scene_flow/testing/image_2/'+img  # Load the photo
    img_filename_2 = Local_config.chemin+'/data_scene_flow/testing/image_3/'+img  # Load the photo
    return([img_filename_1,img_filename_2])

def correlation_2D(I1,I2):

    I1_int=I1.astype(float)
    I2_int=I2.astype(float)

    m1=np.mean(I1_int)
    A=np.sum((I1_int-m1)**2)

    m2=np.mean(I2_int)
    B=np.sum((I2_int-m2)**2)


    if B == 0:
        rep = 0
    else :
        rep = np.sum(np.sum((I2_int-m2)*(I1_int-m1)))/sqrt(A*B)

    return (rep)

def cherche_pts(img_1,img_2,u,v,w,seuil):

    sc_max=seuil

    for i in range (w,np.size(img_i_2,1)-w):
        sc = correlation_2D(img_i_1[v-w:v+w,u-w:u+w],img_i_2[v-w:v+w,i-w:i+w])

        if sc > sc_max :
            sc_max = sc
            i_max = i
    if sc_max > seuil:
        return(i_max)
    else :
        print("non trouver")
        return(-1)

def calc_d_cam_to_cam(T2,T3):
    d_X=T2[0]+T3[0]
    d_Y=T2[1]+T3[1]
    d_Z=T2[2]+T3[2]

    d_cam=sqrt(sqrt(d_X**2+d_Z**2)**2+d_Y**2)

    return(d_cam)


def triangulation (dist_pix_im_g,dist_pix_im_d,T2,T3):

    pi=np.pi

    AB=calc_d_cam_to_cam(T2,T3)
    angle_d_ouverture=np.pi/2
    L_ecrant=1242

    L_ecrant_moitier=L_ecrant/2

    d_ecrant=L_ecrant_moitier/np.tan(angle_d_ouverture/2)

    alpha_prime= np.arctan(abs(dist_pix_im_g-L_ecrant_moitier)/d_ecrant)
    beta_prime = np.arctan(abs(dist_pix_im_d-L_ecrant_moitier)/d_ecrant)

    if dist_pix_im_g >=L_ecrant_moitier :
        alpha= pi/2-alpha_prime
    else :
        alpha= pi/2+alpha_prime

    if dist_pix_im_d >=L_ecrant_moitier :
        beta = pi/2+ beta_prime
    else :
        beta = pi/2- beta_prime

    gama = pi-alpha-beta

    BC=AB*np.sin(alpha)/np.sin(gama)

    # print(degrees(alpha))
    # print(degrees(beta))
    # print(degrees(gama))

    # print(np.cos(beta)*BC)
    # print(d_X)

    # d_Y=-T3[1]

    d_X=-np.cos(beta)*BC+T3[0]
    d_Z=np.sin(beta)*BC-T3[2]-d_ecrant*4.65*10**(-6)

    distance=sqrt(d_X**2+d_Z**2)

    return([distance,d_X,d_Z])

## test correlation 2D
# a=[[1,2,3],[4,5,6]]
# b=a
#
# m=np.mean(a)
# n=np.mean(b)
# A=np.sum((a-m)**2)
# B=np.sum((b-n)**2)
# rep = np.sum(np.sum((b-n)*(a-m)))/sqrt(A*B)
#
# rep_2=correlation_2D(a,b)
#
# print(rep)
# print(rep_2)

## test triangulation
# T0= [2.573699e-16, -1.059758e-16, 1.614870e-16]
# T2= [5.956621e-02, 2.900141e-04, 2.577209e-03]
# T3= [-4.731050e-01, 5.551470e-03,-5.250882e-03]
# AB= calc_d_cam_to_cam(T2,T3)
#
# angle_d_ouverture=np.pi/2
#
# rep=[]
#
# d_X=[]
# d_Z=[]
# taille=621
#
# dist_pix_im_g=363
# dist_pix_im_d=349
# # for dist_pix_im_d in range(taille,taille*2):
# rep=triangulation (dist_pix_im_g,dist_pix_im_d,T2,T3)
# d_X.append(rep[1])
# d_Z.append(rep[2])
#
# print(d_X)
# print(d_Z)
#
#
#
# plt.figure(1)
# plt.plot(0,0,'mo')
# plt.plot(d_X,d_Z,'bo')
# plt.show()
#
# # for dist_pix_im_g in range(taille*2,taille,-1):
# #     for dist_pix_im_d in range(taille):
# #         rep.append(triangulation (dist_pix_im_g,dist_pix_im_d,T2,T3))
# #
# # for dist_pix_im_g in range(taille*2,taille*2-1,-1):
# #     for dist_pix_im_d in range(taille,taille*2):
# #         rep.append(triangulation (dist_pix_im_g,dist_pix_im_d,T2,T3))
# #
# # dist_pix_im_d= np.linspace(0,taille**2+taille,taille**2+taille)
# # plt.figure(1)
# # plt.plot(dist_pix_im_d,rep,'r-', lw=1)
# # plt.show()
#
# # dist_pix_im_g=taille*2
# # for dist_pix_im_d in range(taille,taille*2):
# #     rep.append(triangulation (dist_pix_im_g,dist_pix_im_d,T2,T3))
# #
# # dist_pix_im_d= np.linspace(0,taille,taille)
# # plt.figure(1)
# # plt.plot(dist_pix_im_d,rep,'r-', lw=1)
# # plt.show()

## test cherche pts

# w=10
# seuil=0.5
#
#
# [vect_T1,vect_T2,vect_D1,vect_D2,mat_Rect_1,mat_Rect_2,mat_K1,mat_K2,mat_R1,mat_R2 ]=calib('000020.txt')
#
# [img_filename_1,img_filename_2]=image ('000020_10.png')
#
# # Photo 1
# plt.figure(1)
# img_i_1 = cv2.imread(img_filename_1,0)
# plt.axis('off')
# plt.imshow(cv2.cvtColor(img_i_1, cv2.COLOR_BGR2RGB))
# plt.title("Image 2 de référence")
#
# # Photo 2
# plt.figure(2)
# img_i_2 = cv2.imread(img_filename_2,0)
# plt.axis('off')
# plt.imshow(cv2.cvtColor(img_i_2, cv2.COLOR_BGR2RGB))
# plt.title("Image 3 correspondante")
#
# size=2
#
# u= [182,250,363,462,504,602,645,333,661,1014]#592, 524,  49, 1152, ,906
# v= [224,211,196,187,183,186,183,125,186,106]#148, 171,  135, 99, ,106
#
# d_X=[]
# d_Z=[]
# dist=[]
# for i in range (np.size(u)):
#     i_max=cherche_pts(img_i_1,img_i_2,u[i],v[i],w,seuil)
#     rep=triangulation (u[i],i_max,vect_T1,vect_T2)
#     # dist.append(rep[0])
#     # d_X.append(rep[1])
#     # d_Z.append(rep[2])
#
#     plt.figure(1)
#     plt.plot (u[i],v[i],'bo')
#
#     plt.figure(2)
#     plt.plot (i_max,v[i],'bo')
#
#     plt.figure(3)
#     plt.plot(rep[1],rep[2],'bo')
#     # x = np.array([rep[1]-size, rep[1]+size, rep[1]+size, rep[1]-size, rep[1]-size])
#     # y = np.array([rep[2]-size, rep[2]-size, rep[2]+size, rep[2]+size, rep[2]-size])
#     # plt.plot(x, y,'b')
#     # plt.xlim(-1, 2)
#     # plt.ylim(-1, 2)
#
#
# Taille=140
#
# plt.figure(3)
# plt.plot(0,0,'mo')
# plt.grid()
# plt.axis([-Taille/2, Taille/2, -2, Taille])
# plt.title("Positions mesurées")
# plt.ylabel('Z (Profondeur) (m)')
# plt.xlabel('X (Largeur) (m)')
#
# # X=-8.193427163744156
# # Z=28.53998265869714
# # size=2
# #
# #
# # x = np.array([X-size, X+size, X+size, X-size, X-size])
# # y = np.array([Z-size, Z-size, Z+size, Z+size, Z-size])
# # plt.plot(x, y,'b')
# # plt.xlim(-1, 2)
# # plt.ylim(-1, 2)
#
#
#
#
#
# # zero = np.zeros((np.size(u)))
# # plt.figure(4)
# # plt.plot(zero,dist,'bo')
# # plt.plot(0,0,'mo')
#
#
#
#
# # dist=[]
# # for i in range (621):
# #     rep=triangulation (621,i,vect_T1,vect_T2)
# #     dist.append(rep[0])
# #
# # zero = np.zeros((621))
# #
# # plt.figure(5)
# #
# # plt.plot(zero,dist,'bo')
# # plt.plot(0,0,'mo')
# # plt.grid()
# plt.show()

# ## autre
# #verification peut etre pas besion de R_rect
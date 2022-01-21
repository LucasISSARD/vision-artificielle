import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import *
import scipy as sc

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


    d_X=-np.cos(beta)*BC+T3[0]

    # print(np.cos(beta)*BC)
    # print(d_X)

    # d_Y=-T3[1]
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

#

# ## autre
# #verification peut etre pas besion de R_rect
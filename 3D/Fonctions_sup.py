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

def calc_d_cam_to_cam(T2,T3):
    d_X=T2[0]+T3[0]
    d_Z=T2[2]+T3[2]

    d_cam=sqrt(d_X**2+d_Z**2)

    return(d_cam)


def triangulation (dist_pix_im_g,dist_pix_im_d,L_ecrant, angle_d_ouverture,AB):

    pi=np.pi

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

    # print(degrees(alpha))
    # print(degrees(beta))
    # print(degrees(gama))

    BC=AB*np.sin(alpha)/np.sin(gama)

    # print(AB)
    # print(BC)

    distance=np.sin(beta)*BC

    return(distance)

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

## test triangulation

# T2= [5.956621e-02, 2.900141e-04, 2.577209e-03]
# T3= [-4.731050e-01, 5.551470e-03,-5.250882e-03]
# AB= calc_d_cam_to_cam(T2,T3)
#
# angle_d_ouverture=np.pi/2
#
# dist_pix_im_g=270
# dist_pix_im_d=211
# L_ecrant=1242
#
#
# rep=triangulation (dist_pix_im_g,dist_pix_im_d,L_ecrant, angle_d_ouverture,AB)
#
# print(rep)


## autre
#verification peut etre pas besion de R_rect
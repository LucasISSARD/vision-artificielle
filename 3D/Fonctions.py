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
#Fonction de convertion pour la lecture des fichiers transformation en vecteur
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
# Fonction de convertion pour la lecture des fichiers transformation en matrice
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
    calib_filename= Local_config.chemin+'/2011_09_26/calib_cam_to_cam.txt'
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


def Trace_Image(img_filename_1,img_filename_2):
    # Photo 1
    plt.figure(1)
    img_i_1 = cv2.imread(img_filename_1,0)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img_i_1, cv2.COLOR_BGR2RGB))
    plt.title("Image 2 de référence")

    # Photo 2
    plt.figure(2)
    img_i_2 = cv2.imread(img_filename_2,0)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img_i_2, cv2.COLOR_BGR2RGB))
    plt.title("Image 3 correspondante")

    return ([img_i_1,img_i_2])

def Calcul_matrice_E_F(mat_R1,mat_R2,vect_T1,vect_T2,mat_K1,mat_K2):

    R32 = mat_R2.dot(np.transpose(mat_R1))                              # Calcul de matrice de Rotations
    t32 = vect_T2 - mat_R2.dot((np.transpose(mat_R1)).dot(vect_T1))             # Calcul de matrice de translation
    t_x = np.array([[0,-t32[2],t32[1]],
                    [t32[2],0,-t32[0]],
                    [-t32[1],t32[0],0]])
    E = t_x.dot(R32)                                                                     # Calcul de la matrice Essentielle
    F = (np.transpose(np.linalg.inv(mat_K1)).dot(E)).dot(np.linalg.inv(mat_K2))   # Calcul de la matrice Fondamentale
    return ([E,F])

def Calcul_droite_epi(u,v,F):

    plt.figure(1)
    plt.plot (u,v,'mo')
    ABC = np.array([u,v,1]).dot(F)     # Droite épipolaire                                           #
    A = ABC[0];B=ABC[1];C=ABC[2]
    x= np.linspace(0,1242,1242)
    plt.figure(2)
    plt.plot(x, -(A*x+C)/B, 'r-', lw=1)
    return([A,B,C])

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


def cherche_point_droite_epi(w,seuil,A,B,C,img_i_1,img_i_2,u,v):
    # Recherche du point correspondant sur l'image droite
    sc_max=seuil
    for j in range(w,np.size(img_i_2,1)-w):
        i = round(-(A*j+C)/B)    # round = arrondie

        if i > w + 1 and i < np.size(img_i_2,0)-w:
            sc = correlation_2D(img_i_1[v-w:v+w,u-w:u+w],img_i_2[i-w:i+w,j-w:j+w])
            if sc > sc_max :
                sc_max = sc
                i_max = i
                j_max =j
                #print(sc)
    if sc_max > seuil:
        plt.figure(2)
        plt.plot(j_max,i_max,'go')
    return ([j_max,i_max])

def cherche_pts(img_1,img_2,u,v,w,seuil):
    # sans droite epipolaire
    # Simplification caméra sur la meme ligne

    sc_max=seuil

    for i in range (w,np.size(img_2,1)-w):
        sc = correlation_2D(img_1[v-w:v+w,u-w:u+w],img_2[v-w:v+w,i-w:i+w])

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

    # d_Y=-T3[1]

    d_X=-np.cos(beta)*BC+T3[0]
    d_Z=np.sin(beta)*BC-T3[2]-d_ecrant*4.65*10**(-6)

    distance=sqrt(d_X**2+d_Z**2)

    return([distance,d_X,d_Z])

def correspondance_sans_epipo(u,v,img_i_1,img_i_2,w,seuil,vect_T1,vect_T2):
    d_X=[]
    d_Z=[]
    dist=[]
    for i in range (np.size(u)):
        i_max=cherche_pts(img_i_1,img_i_2,u[i],v[i],w,seuil)
        dist.append(triangulation(u[i],i_max,vect_T1,vect_T2))

        # plt.figure(1)
        # plt.plot (u[i],v[i],'bo')
        #
        # plt.figure(2)
        # plt.plot (i_max,v[i],'bo')
        # # Affichage dans un plant 2D
        # plt.figure(3)
        # plt.plot(dist[1],dist[2],'bo')
    return dist
    # Taille=100
    # plt.figure(3)
    # plt.plot(0,0,'mo')
    # plt.grid()
    # plt.axis([-Taille/2, Taille/2, -2, Taille])
    # plt.title("Positions mesurées")
    # plt.ylabel('Z (Profondeur) (m)')
    # plt.xlabel('X (Largeur) (m)')

def Extract_u_v(Positions_vehicule):
    # Etrait la position des voitures donner par l'algorithme de détection et suivi de véhicule
    u=[]
    v=[]
    for i in range(len(Positions_vehicule)):
        u.append(Positions_vehicule[i][0])
        v.append(Positions_vehicule[i][1])
    return ([u,v])


## Detecttion de voiture
def acqFrame (video_path, frame):
    # Calcul du nom du fichier désiré
    zero_pad = ""
    for i in range(10 - len((str)(frame))):
        zero_pad = zero_pad + "0"
    img_path = video_path + zero_pad + (str)(frame) + ".png"
    print("Img_path : ", img_path)
    # Acquisition de l'image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return img

def detectCars (img):
    # Détection des voitures dans l'image
    cars = car_cascade.detectMultiScale(img, scaleFactor = 1.04, minNeighbors = 8, minSize=(20, 20), maxSize=(180, 180))

    print("Voitures détectées sur cette image :")
    i = 0
    for (x,y,w,h) in cars:
        I = 0
        for (X, Y, W, H) in cars:
            # Si deux détections se chevauchent sur UNE MEME IMAGE (si son centre est dans la box d'une autre), on supprime la plus petite
            if x+round(w/2) >= X and x+round(w/2) <= X+W and y+round(w/2) >= Y and y+round(w/2) <= Y+H and x != X and y != Y:
                if cars[i][2] < cars[I][2]:
                    cars[i] = (0,0,0,0)
                else:
                    cars[I] = (0,0,0,0)
            I = I + 1
        if cars[i][0] != 0 and cars[i][1] != 0 and show_detected == True:
            cv2.drawMarker(img,(x+round(w/2),y+round(w/2)),color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=2)
            cv2.putText(img, (str)(i), (x+round(w/2)+8,y+round(w/2)+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,255), thickness=1)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        print(i, cars[i])
        i = i + 1

    return cars


def detection_vehicule():
    # Variables
    first = True
    trackers = []   # Liste des traceurs
    liste = []      # Liste contenant les voitures détectées, format : [ x , y , w  , h , on/off ]

    # Programme principal
    frame = first_frame
    while frame < last_frame:
        # Acquisition de l'image
        img = acqFrame(video_path, frame)

        # Détection des voitures
        cars = detectCars(img)

        # Suivi des voitures
        if first == True:
            first = False
        else :

            # Détection des NOUVELLES voitures
            for i in range(len(cars)):    # Pour toutes les "voitures" fraîchement détectées
                ignore = False

                # Si la voiture en cours d'étude est un doublon (valeurs mises à 0), on l'ignore
                if cars[i][2] == 0:
                    ignore = True
                    break

                # Si la voiture en cours d'étude est déjà dans la liste, on l'ignore
                th1 = 30
                for k in range(len(liste)):
                    if cars[i][0] >= liste[k][0]-th1 and cars[i][0] <= liste[k][0]+th1 and cars[i][1] >= liste[k][1]-th1 and cars[i][1] <= liste[k][1]+th1:
                        ignore = True

                if ignore == False:
                    for j in range(len(cars_history)-1): # /!\ Faut pas le -1 normalement mais ça marche mieux avec donc bon...
                        # Si c'est la deuxième fois de suite qu'on détecte cette voiture
                        th = 20     # seuil
                        if cars[i][0] >= cars_history[j][0]-th and cars[i][0] <= cars_history[j][0]+th and cars[i][1] >= cars_history[j][1]-th and cars[i][1] <= cars_history[j][1]+th:
                            # Alors c'est bien une nouvelle voiture détectée, on l'ajoute à la liste et on lui colle un traceur
                            liste.append([cars[i][0], cars[i][1], cars[i][2], cars[i][3], 1])           # On l'ajoute à la liste des voitures vraiment détectées
                            ofs = int(0.2*cars[i][2])                                                   # On réduit la taille de la box de 20% pour aider la détection
                            box = (cars[i][0]+ofs, cars[i][1]+ofs, cars[i][2]-2*ofs, cars[i][3]-2*ofs)  # On trace sa box
                            trackers.append(cv2.legacy.TrackerMedianFlow_create())
                            trackers[-1].init(img, box)     # -1 = dernier tracker de la liste
                            break

            print("TRACKERS = ", trackers)
            print("LISTE = ", liste)

            tracker_to_remove = []
            list_to_remove = []
            for i in range(len(trackers)):
                success, box = trackers[i].update(img) # Met à jour les traceurs
                if success:
                    if show_tracked == True:
                            p1 = (int(box[0]), int(box[1]))
                            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                            x = int(p1[0] + (p2[0] - p1[0])/2)
                            y = int(p2[1] + (p1[1] - p2[1])/2)
                            cv2.drawMarker(img,(x,y),color=(255,0,0), markerType=cv2.MARKER_CROSS, thickness=2)
                            cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
                else:
                    # TODO : Améliorer la gestion de la perte d'objets pour éviter de vider complètement la liste à chaque fois qu'on perd qu'un seul traceur
                    print("Echec du suivi tacker n°", i)
                    tracker_to_remove.append(trackers[i])   # Ajoute le traceur actuel dans la liste des traceurs à supprimer
                    list_to_remove.append(liste[i])         # Ajoute l'élément de la liste concerné dans la liste des éléments de la liste à supprimer (wow)

            # Et on supprime !
            for j in tracker_to_remove:
                trackers.remove(j)
            for j in list_to_remove:
                liste.remove(j)



        cars_history = cars     # On stock les voitures actuelles dans l'historique

        # Affichage de l'image
        cv2.imshow("video", img)

        frame = frame + 1
        cv2.waitKey(1)
        time.sleep(0.1)
        print("-------------")

    # Fermeture des fenêtres
    cv2.destroyAllWindows()




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
# img_i_1,img_i_2 = Trace_Image(img_filename_1,img_filename_2)
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
# #
# #
# #
# #
# #
# # zero = np.zeros((np.size(u)))
# # plt.figure(4)
# # plt.plot(zero,dist,'bo')
# # plt.plot(0,0,'mo')
# #
# #
# #
# #
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
## Correpondance entre 2 points sur 2 images différentes avec la droite épipolaire
# Prenons le cas où l'on détecte sur image 2 et que l'on fait la correspondance sur l'image 3
# Calcul des matrices essentielles et fondamentales
# E, F = Fonctions.Calcul_matrice_E_F(mat_Rect_1,mat_Rect_2,vect_T1,vect_T2,mat_K1,mat_K2)
# u,v = 363,197 # Test à une position
# A,B,C = Fonctions.Calcul_droite_epi(u,v,F) # calcul et affiche la droite épipolaire
#
# w = 75   # taille du masque de corrélation (2*w+1)*(2*w+1)
# seuil = 0.2 # seuil de corrélation
# j_max,i_max = Fonctions.cherche_point_droite_epi(w,seuil,A,B,C,img_i_1,img_i_2,u,v)
#
# ## Triangulation
# Dist_cam_vehicule = Fonctions.triangulation(u,j_max,vect_T1,vect_T2)
# print(Dist_cam_vehicule)
## autre
#verification peut etre pas besion de R_rect
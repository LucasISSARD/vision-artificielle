# Rapport vision artificielle : Détection et suivi 3D de voitures dans des séquences routières
Notre objectif est de réaliser un programme capable de détecter et de suivre des véhicules dans des séquences vidéo issues d’une tête stéréoscopiques embarquée et de localiser ces véhicules en 3D par rapport à la voiture qui embarque la tête stéréo. L’application de cette détection peut se faire pour les véhicules autonomes. En effet, le véhicule doit connaitre son environnement et réagir en fonction de celui-ci pour éviter toutes collisions. La détection des autres véhicules mais aussi des piétons est donc nécessaire. On se focalisera sur la détection des véhicules. Notre programme sera réalisé en Python. On utilisera notamment la bibliothèque OpenCv. Cette bibliothèque est spécialisée dans le traitement d’image en temps réels. Nos programmes seront testés sur une séquence de la base de données KITTI. Cette base de données fournit des séquences vidéo multi-vues dans différents contextes (ville, autoroute,) avec les paramètres de calibrage intrinsèque et extrinsèque des caméras. 

Les véhicules autonomes ont besoins de connaitre parfaitement leur environnement afin de circuler en toute sécurité. Il existe plusieurs technologies permettant de détecter l’environnement extérieur de la voiture. Dans notre cas, nous avons à notre disposition deux caméras situer sur le toit d’une voiture qui prennent des images en même temps.
Dans ce rapport, nous nous occuperons tout d’abord de détecter une voiture sur un image et nous suivrons ce véhicule sur une suite d’image. Par la suite nous mettrons en correspondance les deux images des deux caméras afin de déterminer la position du véhicule détecter par rapport au repère origine. 

# 1. Détection de véhicules
## Etude bibliographique
## Algorithme de Viola & Jones

# 2. Suivi de véhicules
## Etude bibliographique
## Algorithme MedianFlow

# 3. Mise en correspondance et localisation 3D
Une seule caméra ne nous permet pas de voir en 3D. Ainsi, pour localiser le véhicule il est nécessaire d’ajouter une ou plusieurs caméras permettant de voir en 3 dimensions.  Le premier but est de mettre en correspondance les points sur les images réalisés en même temps mais avec des caméras à des positions différentes. Ensuite, pour déterminer la position du véhicule, on utilise la triangulation. 

## Mise en correspondance
Dans notre cas, nous avons à notre disposition des photos de 2 caméras cote à cote. Nous allons prendre le cas ou l’on utilise notre algorithme de détection de véhicule sur la photo 1 soit la caméra de gauche (repère R1). Notre algorithme nous retourne la position du véhicule en pixel. La première chose à faire de retrouver cette position sur la photo 2 soit la caméra de droite (Repère R2) afin de localiser le véhicule avec la triangulation.  
Pour notre projet, on utilise la base de données KITI comme fichier de test. Cette bibliothèque contient un ensemble d’image et de séquence en multi vues avec les différents paramètres de calibrage des caméras (K1 et K2).
Dans notre cas, le schéma ci-dessous représente notre situation. Il y a deux caméras qui prennent les images en même temps. La première à la position O1 et la deuxième à la position O2. 

<img src="D:\Documents\GitHub\vision-artificielle-Etienne\vision-artificielle\DOC\Schéma_des_deux_caméras.png" alt="width=80%" style="zoom: 50%;" />

Dans un premier temps il est nécessaire de déterminer la rotation et la translation entre les caméras. Dans la bibliothèque, on a à notre disposition les rotations et les translations entre les caméras et un repère. 

<img src="D:\Documents\GitHub\vision-artificielle-Etienne\vision-artificielle\DOC\Changement_de_vecteur.png" style="zoom:67%;" />
$$
R21=R2.R1^T
$$

$$
t31=t2-R2.R1^T.t1
$$

```
![formula](https://render.githubusercontent.com/render/math?math=t31=t2-R2.R1^T.t1)
```

Pour la mise en correspondance des points, il est possible d’utiliser la droite épipolaire qui permet de déterminer le point correspondant dans la deuxième image. Or dans ce cas nous avons à notre disposition des images déjà rectifiées. Il n’est donc pas nécessaire de calculer la droite épipolaire. Il suffit de faire une corrélation 2D selon la même ligne. Dans le cas ci-dessous, on souhaite faire la correspondance d’un point entre 2 images. Ce point appartient à la même ligne. Ainsi on peut tracer une droite qui coupe horizontalement l’image. Ensuite, on va parcourir cette droite en faisant une corrélation 2D avec l’image de référence. On pourra choisir la taille du masque qui correspond à la matrice blanche sur le dessin. En parcourant la droite avec la matrice blanche, on va garder en mémoire le maximum du résultat de la corrélation. Lorsque ce nombre est maximal soit 1, cela veut dire que les deux images sont identiques. Ainsi, en jouant sur le seuil de corrélation et le masque, on trouve le point correspondant à l’endroit ou la corrélation est maximale. On pourra donc savoir sa position exacte. Le pixel sera sur la même ligne. La colonne sera située au maximum de la corrélation.

<img src="D:\Documents\GitHub\vision-artificielle-Etienne\vision-artificielle\DOC\Mise_en_correspondance.png" style="zoom:67%;" />

Maintenant que l’on a mis en correspondant les points sur les deux images, il nous reste à calculer la réelle position de la voiture. Dans notre cas nous avons les positions en pixels. On cherche la position du point :
$$
P=(X,Y,Z)
$$

## Triangulation

La triangulation et une méthode pour calculer une position à partir de deux points dont on connait la distance. Voici un schéma qui explique la technique de triangulation. Les point A et B sont des positions que l’on connaît et le point C est la position que nous cherchons à déterminer.

<img src="D:\Documents\GitHub\vision-artificielle-Etienne\vision-artificielle\DOC\La_triangulation.png" style="zoom: 50%;" />

Comme nous connaissons les positions des points A et B nous pouvons calculer la distance AB. Et en mesurant les angle α et β nous pouvons calculer Ɣ le troisième angle du triangle ABC :
$$
Ɣ=2π - α – β
$$
Ainsi avec la loi des sinus :
$$
\frac{AB}{sin(Ɣ)}  =  \frac{BC}{sin(α)}  =  \frac{CA}{sin(β)}
$$

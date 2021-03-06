# vision-artificielle
L'objectif de ce projet est de réaliser une application en Python permettant la détection et la localisation dans l'espace de véhicules à partir des images d'une scène vue par une caméra stéréoscopique. La base de donnée utilisée est celle de KITTI.

Démonstration en vidéo : https://youtu.be/a7-0vDmLPqY

## Prérequis
- python 				3.8.10
- opencv-python         4.5.5.62
- opencv-contrib-python 4.5.5.62

## Etapes du projet
### 1. Etudes bibliographiques
- [x] Détecteurs de voiture et de piétons dans une image, choix de la méthode retenue pour ce projet
- [x] Suivi d'objets dans une séquence vidéo, choix de la méthode retenue pour ce projet

### 2. Application
- [x] Détection et suivi de voitures dans une image
	- [x] Mise en oeuvre de l'algorithme de Viola et Jones pour la détection d'objets
	- [x] Mise en oeuvre de l'algorithme MedianFlow pour le suivi d'objets
- [x] Localisation 3D des objets
	- [x] Mise en correspondance des points
	- [x] Triangulation
	- [x] Affichage en 2D 

## Tutoriel
### 1. Vérifiez vos prérequis
- Installer python, opencv-python, opencv-contrib-python dans les version adéquates

### 2. Téléchargez le dataset
- S'inscire sur le site de KITTI : http://www.cvlibs.net/datasets/kitti/
- Télécharger un dataset (ex: 2011_09_26_drive_0015 en version synchronisé et rectifié)
- Télécharger les données de calibration liées
- Organiser l'arborescence et adapter le chemin dans Local_config.py
<img src="Doc/Arborescence.PNG" width=30%/>

### 3. Exécuter le main.py


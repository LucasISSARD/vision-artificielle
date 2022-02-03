
#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


## Local config

chemin='C:/Users/quent/Desktop/Cours 5A/VISION_PROJET'
video_path = "C:/Users/quent/Desktop/Cours 5A/VISION_PROJET/road/"    # Chemin de la vidéo ( /!\ sur Windows, remplacer les \ par des / sans oublier le / final )

video_path_02 = "C:/Users/quent/Desktop/Cours 5A/VISION_PROJET/2011_09_26/image_02/data/"
video_path_03 = "C:/Users/quent/Desktop/Cours 5A/VISION_PROJET/2011_09_26/image_03/data/"
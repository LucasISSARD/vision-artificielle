

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


# Charger les photos
img_filename_1 = 'C:/Users/quent/Desktop/Cours 5A/VISION_PROJET/data_scene_flow/testing/image_2/000000_10.png'  # Load the photo
img_filename_2 = 'C:/Users/quent/Desktop/Cours 5A/VISION_PROJET/data_scene_flow/testing/image_3/000000_10.png'  # Load the photo

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


plt.show()


#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt



img_filename_1 = 'C:/Users/quent/Desktop/Cours 5A/VISION_PROJET/data_scene_flow/testing/image_2/000000_10.png'  # Load the photo
img_filename_2 = 'C:/Users/quent/Desktop/Cours 5A/VISION_PROJET/data_scene_flow/testing/image_3/000000_10.png'  # Load the photo
# Plot the Photo
fig=plt.figure(figsize=(8,8))
img_i = cv2.imread(img_filename)


plt.axis('off')
plt.imshow(cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB))
plt.show()
# Compute crypt area (aka # cells per crypt) distribution and second neighbour distribution

import numpy as np
# from sklearn.cluster import DBSCAN
from skimage import measure
# import matplotlib.pyplot as plt
from math import atan, atan2, cos, acos, pi

import glob
import sys
from os.path import join, basename, splitext, extsep

from PIL import Image

file_path = sys.argv[1]
output_path = sys.argv[2]
name = basename(file_path)[:-7]

scaling_factor = 1/0.8

im = np.array(Image.open(file_path))
nx = im.shape[0]
ny = im.shape[1]

binary = im > 0
# print(binary)

labels = measure.label(binary, background=0)
# plt.matshow(labels.T, cmap='nipy_spectral')
# print(np.amax(labels))

centroids = []
area = []
# for i,(label,color) in enumerate(zip(unique_labels,colors)):
for i in range(1,np.amax(labels)):

    idx = np.argwhere(labels==i)
    if idx.shape[0]>1:
        # idx[:,1] = ny - idx[:,1]
        # print(idx.shape)
        # print(idx)


        centroid=np.mean(idx, axis=0)
        # print(centroid)
        centroids.append(centroid)
        area.append(idx.shape[0])

centroids = np.array(centroids)
area = np.array(area)
# plt.scatter(centroids[:,0],centroids[:,1])
# plt.show()

diameter = 2*np.sqrt(area/pi)
# print("centroids shape",centroids.shape)
# print("diameter shape",diameter.shape)

a_d = []
b_d = []

for i in range(centroids.shape[0]):
    # print("************i:",i)
    distances = []
    for j in range(centroids.shape[0]):
        # print("************j:",j)

        if i == j:
            continue
        else:
            distances.append(np.linalg.norm(centroids[i,:]-centroids[j,:]))
        # print(distances[-1])
    sorted = np.argsort(np.array(distances))
    # print(distances)
    # print(sorted)

    a_d.append(sorted[0]/diameter[i])
    b_d.append(sorted[1]/diameter[i])

a_d = np.array(a_d)
b_d = np.array(b_d)


save_as = output_path + name

np.savetxt(save_as + ".diam", diameter, delimiter=",")
np.savetxt(save_as + ".ad", a_d, delimiter=",")
np.savetxt(save_as + ".bd", b_d, delimiter=",")

# fig,ax=plt.subplots(1,3,figsize=(20,10))
#
# ax[0].hist(diameter)
# ax[0].set_title('crypt diameter')
# ax[1].hist(a_d)
# ax[1].set_title('a_d')
# ax[2].hist(b_d)
# ax[2].set_title('b_d')
#
# plt.show()

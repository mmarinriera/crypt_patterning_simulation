# Compute crypt area (aka # cells per crypt) distribution and second neighbour distribution

import numpy as np
from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt
from math import atan, atan2, cos, acos, pi

import glob
import sys
from os.path import join, basename, splitext, extsep

import vtk
from vtk.util.numpy_support import vtk_to_numpy

file_path = sys.argv[1]
output_path = sys.argv[2]
name = basename(file_path)[:-7]

scaling_factor = 1/0.8

reader = vtk.vtkDataSetReader()
reader.SetFileName(file_path)
reader.ReadAllScalarsOn()  # Activate the reading of all scalars
reader.Update()
data=reader.GetOutput()

coords = vtk_to_numpy(data.GetPoints().GetData())[1:,:]*scaling_factor
diff = vtk_to_numpy(data.GetPointData().GetScalars("diff"))[1:]
is_paneth = vtk_to_numpy(data.GetPointData().GetScalars("is_paneth"))[1:]
is_pattern = vtk_to_numpy(data.GetPointData().GetScalars("is_pattern"))[1:]

crypts = coords[diff<0.4,:]
# crypts = coords[is_pattern == 1,:]

# print("crypts shape:", crypts.shape)
clustering = DBSCAN(eps=1.0, min_samples=8).fit(crypts)
labels = clustering.labels_
# print(labels)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


unique_labels = set(labels)
# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
# print(unique_labels)
# print(colors)

# plt.scatter(crypts[:,])
# centroids = np.zeros(shape=(len(unique_labels),2),dtype=np.float)
centroids = []
diameter = []
# for i,(label,color) in enumerate(zip(unique_labels,colors)):
for i,label in enumerate(unique_labels):

    if(label==-1):
        continue
    cluster = crypts[labels==label,:]
    # plt.scatter(cluster[:,0], cluster[:,1],color=color)

    centroid=np.mean(cluster, axis=0)[:2]
    centroids.append(centroid)
    dist=np.sqrt(np.sum((centroid - cluster[:,:2])**2,axis=1))
    diameter.append(2*np.amax(dist))

centroids = np.array(centroids)
diameter = np.array(diameter)
# print("centroids shape",centroids.shape)
# print("diameter shape",diameter.shape)

a_d = []
b_d = []

for i in range(len(unique_labels)-1):
    # print("************i:",i)
    distances = []
    for j in range(len(unique_labels)-1):
        # print("************j:",j)

        if i == j:
            distances.append(1000)
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

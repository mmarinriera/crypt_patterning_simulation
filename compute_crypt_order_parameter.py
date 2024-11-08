# Compute the order parameter between neighbouring crypts

import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from math import atan, atan2, cos, acos, pi

import glob
import sys
from os.path import join, basename, splitext, extsep

import vtk
from vtk.util.numpy_support import vtk_to_numpy

T = 1000
time_interval = T/100

file_pattern = sys.argv[1]
# output_path = sys.argv[2]
name = basename(file_pattern)[:-2]
# print("name",name)

sorted_list = sorted(glob.glob(file_pattern), key=lambda f: int(f.rsplit(extsep, 1)[0].rsplit("_",1)[-1]))

time = np.arange(0, T+time_interval, time_interval)

time_transform_factor = 0.075 # We assume one arbitrary time unit = 30 min

time = time_transform_factor * time

scaling_factor = 1/0.8

reader = vtk.vtkDataSetReader()
reader.SetFileName(sorted_list[-1])
reader.ReadAllScalarsOn()  # Activate the reading of all scalars
reader.Update()
data=reader.GetOutput()

coords = vtk_to_numpy(data.GetPoints().GetData())[1:,:]*scaling_factor
diff = vtk_to_numpy(data.GetPointData().GetScalars("diff"))[1:]
is_paneth = vtk_to_numpy(data.GetPointData().GetScalars("is_paneth"))[1:]
is_pattern = vtk_to_numpy(data.GetPointData().GetScalars("is_pattern"))[1:]

crypts = coords[diff<0.4,:]
# crypts = coords[is_pattern == 1,:]

print("crypts shape:", crypts.shape)
clustering = DBSCAN(eps=1.0, min_samples=8).fit(crypts)
labels = clustering.labels_
# print(labels)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
print(unique_labels)
# print(colors)

# plt.scatter(crypts[:,])
# centroids = np.zeros(shape=(len(unique_labels),2),dtype=np.float)
centroids = []
radius = []
for i,(label,color) in enumerate(zip(unique_labels,colors)):
    if(label==-1):
        continue
    cluster = crypts[labels==label,:]
    plt.scatter(cluster[:,0], cluster[:,1],color=color)

    centroid=np.mean(cluster, axis=0)[:2]
    centroids.append(centroid)
    dist=np.sqrt(np.sum((centroid - cluster[:,:2])**2,axis=1))
    radius.append(np.amax(dist))

centroids = np.array(centroids)
# print(centroids)


order_param_array = []
shortest_distance_array = []
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

    vec_0 = (centroids[sorted[0],:] - centroids[i,:])/distances[sorted[0]]
    vec_1 = (centroids[sorted[1],:] - centroids[i,:])/distances[sorted[1]]
    angle = acos(vec_0[0]*vec_1[0] + vec_0[1]*vec_1[1])
    order_param = cos(2*(angle-pi/2))
    print(angle, order_param)
    order_param_array.append(order_param)
    shortest_distance_array.append(distances[sorted[0]])


print("av order param =", np.mean(order_param_array))
print("av shortest distance =", np.mean(shortest_distance_array))

# plt.show()


from scipy.spatial import Delaunay
tri = Delaunay(centroids)
print(tri.simplices)

gridness = 0
n_tri = 0
angles=[]
distances=[]

for i in range(tri.simplices.shape[0]):
    print("******i",i)
    a = centroids[tri.simplices[i,0],:]
    b = centroids[tri.simplices[i,1],:]
    c = centroids[tri.simplices[i,2],:]

    print("a",a)
    print("b",b)
    print("c",c)

    ab = b-a
    bc = c-b
    ac = c-a

    l_ab = np.linalg.norm(ab)
    l_bc = np.linalg.norm(bc)
    l_ac = np.linalg.norm(ac)
    distances += [l_ab,l_bc,l_ac]

    # Heron's formula
    s = (l_ab+l_bc+l_ac)/2
    surf_area = (s*(s-l_ab)*(s-l_bc)*(s-l_ac)) ** 0.5

    if(surf_area < 1.0):
        continue

    sides = np.array([l_ab, l_bc, l_ac])
    print(sides)
    sides = np.sort(sides)
    print(sides)

    gridness += abs(surf_area - 0.5*(sides[0]*sides[1]))
    n_tri += 1
    print("surfarea", surf_area)
    print("rectangle",0.5*(sides[0]*sides[1]))
    print("diff",surf_area - 0.5*(sides[0]*sides[1]))

    # Measure triangle angles to check distribution
    bac=acos(np.dot(ab,ac)/(l_ab*l_ac))*180/pi
    abc=acos(np.dot(-ab,bc)/(l_ab*l_bc))*180/pi
    acb=180-bac-abc
    print("****angles******")
    print(bac,abc,acb)
    angles += [bac,abc,acb]


print("gridness =",gridness/float(n_tri))

plt.triplot(centroids[:,0], centroids[:,1], tri.simplices)
plt.plot(centroids[:,0], centroids[:,1], 'o')
plt.show()

plt.hist(angles,bins=20)
plt.xlabel("angles")
plt.show()
plt.hist(radius,bins=50)
plt.xlabel("crypt radius")
plt.show()
plt.hist(distances,bins=20)
plt.xlabel("inter-centroid distances")
plt.show()

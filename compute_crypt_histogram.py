# Compute the order parameter between neighbouring crypts

import numpy as np
# from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt
from math import atan, atan2, cos, acos, pi
from PIL import Image

# import glob
import sys
from os.path import join, basename, splitext, extsep

import vtk
from vtk.util.numpy_support import vtk_to_numpy

file_name = sys.argv[1]
output_path = sys.argv[2]
name = basename(file_name)[:-4]
# print("name",name)



w=1.0
save_as = output_path+ name + "_pix-size_" + str(w) + "_hist.tif"
#print(save_as)

reader = vtk.vtkDataSetReader()
reader.SetFileName(file_name)
reader.ReadAllScalarsOn()  # Activate the reading of all scalars
reader.Update()
data=reader.GetOutput()

coords = vtk_to_numpy(data.GetPoints().GetData())[1:,:]
diff = vtk_to_numpy(data.GetPointData().GetScalars("diff"))[1:]
is_paneth = vtk_to_numpy(data.GetPointData().GetScalars("is_paneth"))[1:]
is_pattern = vtk_to_numpy(data.GetPointData().GetScalars("is_pattern"))[1:]

# crypts = coords[is_pattern == 1,:]
crypts = coords[diff<0.4,:]
n_cells = crypts.shape[0]

# Determine the effective cell radius as the average minimal distance
# from cell to neighbour
av_diameter = []
for i in range(n_cells):
    cell = crypts[i,:]
    dist = np.sort(np.sqrt(np.sum((cell - coords)**2,axis=1)))
    # print("i",i)
    # print(dist[1:7])
    av_diameter.append(np.mean(dist[1:7]))
print("mean distance",np.mean(np.array(av_diameter)))

scaling_factor = 1/np.mean(np.array(av_diameter))
# print(crypts[:10,:])
# print(scaling_factor)
crypts *= scaling_factor

rpattern = name.split("_")[-6]
print(rpattern)
print("scaled diameter of pattern=",2*float(rpattern)*scaling_factor)

# Create histogram
x_min, x_max = np.amin(crypts[:,0]), np.amax(crypts[:,0])
y_min, y_max = np.amin(crypts[:,1]), np.amax(crypts[:,1])

bins_x = np.arange(x_min-w, x_max + w, w)
bins_y = np.arange(y_min-w, y_max + w, w)

hist = np.histogram2d(crypts[:,0],crypts[:,1],bins=[bins_x,bins_y])[0].astype(np.int8)

# plt.imshow(hist.T)
# plt.colorbar()
# plt.show()

# np.savetxt(save_as, hist, fmt='%d',delimiter=', ')
im = Image.fromarray(hist)
im.save(save_as)

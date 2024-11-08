# Compute the Mean Clonal Squared Centroid Size (MCSCS) for a growing aggregate

import numpy as np
# import matplotlib.pyplot as plt
from math import atan, atan2, pi

import glob
import sys
from os.path import join, basename, splitext, extsep

import vtk
from vtk.util.numpy_support import vtk_to_numpy

T = 1000
time_interval = T/100

file_pattern = sys.argv[1]
output_path = sys.argv[2]
name = basename(file_pattern)[:-2]
# print("name",name)

sorted_list = sorted(glob.glob(file_pattern), key=lambda f: int(f.rsplit(extsep, 1)[0].rsplit("_",1)[-1]))

time = np.arange(0, T+time_interval, time_interval)

time_transform_factor = 0.075 # We assume one arbitrary time unit = 30 min

time = time_transform_factor * time

N_t = []
stem_ratio_t = []
paneth_ratio_t = []
MSD_t = []

# Save positions at initial state
reader = vtk.vtkDataSetReader()
reader.SetFileName(sorted_list[0])
reader.ReadAllScalarsOn()  # Activate the reading of all scalars
reader.Update()
data=reader.GetOutput()

coords_0 = vtk_to_numpy(data.GetPoints().GetData())[1:,:]
diff_0 = vtk_to_numpy(data.GetPointData().GetScalars("diff"))[1:]
is_paneth_0 = vtk_to_numpy(data.GetPointData().GetScalars("is_paneth"))[1:]

n_0 = coords_0.shape[0]

for t, state in zip(time, sorted_list):
    # print("t",t,"state", state)
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(state)
    reader.ReadAllScalarsOn()  # Activate the reading of all scalars
    reader.Update()
    data=reader.GetOutput()

    coords = vtk_to_numpy(data.GetPoints().GetData())[1:,:]
    diff = vtk_to_numpy(data.GetPointData().GetScalars("diff"))[1:]
    is_paneth = vtk_to_numpy(data.GetPointData().GetScalars("is_paneth"))[1:]

    n_neighbours = vtk_to_numpy(data.GetPointData().GetScalars("n_epi_nbs")).astype(float)[1:]
    n_cells = coords.shape[0]
    N_t.append(n_cells)
    n_stem = (diff<0.9).sum()
    stem_ratio_t.append(n_stem/n_cells)
    n_paneth = np.sum(is_paneth)
    paneth_ratio_t.append(n_paneth/n_cells)

    # Compute MSD
    sum_sd = 0
    n_sum = 0
    for i in range(n_0):
        if(diff[i]<1.0 and is_paneth[i] == False ):
            sum_sd += np.linalg.norm(coords[i,:] - coords_0[i,:])**2
            n_sum += 1
    if n_sum == 0:
        MSD_t.append(0)
    else:
        MSD_t.append(sum_sd/n_sum)

# truncates the time array in case the simulated crashed mid-way or something
if len(MSD_t) < time.shape[0]:
    time = time[:len(MSD_t)]

# # Compuyte MSD deviation with respect to experimental curve
# cell_diameter_length = 20
# exp_MSD = time * (1000/(cell_diameter_length**2))/8.33
# MSD_t = np.array(MSD_t)
#
#
# plt.plot(time[time<=10],exp_MSD[time<=10],'-o', label='exp')
# plt.plot(time[time<=10],MSD_t[time<=10],'-o', label='sim')
# plt.legend()
# plt.show()
#
# # Compute sum of squares between experimental and simulated progressions
# sq_err = np.sum((exp_MSD[time<=10] - MSD_t[time<=10])**2)
# print("MSD squared error",sq_err)

save_as = output_path + name

time_series_N = np.column_stack((time, np.array(N_t)))
time_series_stem_ratio = np.column_stack((time, np.array(stem_ratio_t)))
time_series_paneth_ratio = np.column_stack((time, np.array(paneth_ratio_t)))
time_series_MSD = np.column_stack((time, MSD_t))

np.savetxt(save_as + ".N", time_series_N, delimiter=",")
np.savetxt(save_as + ".stem_ratio", time_series_stem_ratio, delimiter=",")
np.savetxt(save_as + ".paneth_ratio", time_series_paneth_ratio, delimiter=",")
np.savetxt(save_as + ".MSD", time_series_MSD, delimiter=",")

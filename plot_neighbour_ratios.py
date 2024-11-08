# Compute degree of heterotypic cell mixing and MSD

import numpy as np
import matplotlib.pyplot as plt
from math import atan, atan2, pi

import glob
import sys
from os.path import join, basename, splitext, extsep

import vtk
from vtk.util.numpy_support import vtk_to_numpy

T = 1000
time_interval = T/100

file_pattern = sys.argv[1]
name = basename(file_pattern)[:-2]
# print("name",name)

sorted_list = sorted(glob.glob(file_pattern), key=lambda f: int(f.rsplit(extsep, 1)[0].rsplit("_",1)[-1]))

time = np.arange(0, T+time_interval, time_interval)

time_transform_factor = 0.075 # We assume one arbitrary time unit = 30 min

time = time_transform_factor * time

mean_homotypic_ratio_t = []
mean_heterotypic_ratio_t = []
misplaced_ratio_t = []
n_t = []
stem_ratio_t = []
paneth_ratio_t = []
MSD_t = []
Lx_t = []
Ly_t = []

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

    n_neighbours = vtk_to_numpy(data.GetPointData().GetScalars("n_epi_nbs")).astype(np.float)[1:]
    n_homotypic = vtk_to_numpy(data.GetPointData().GetScalars("n_homotypic")).astype(np.float)[1:]
    n_heterotypic = vtk_to_numpy(data.GetPointData().GetScalars("n_heterotypic")).astype(np.float)[1:]
    n_cells = coords.shape[0]
    n_t.append(n_cells)
    n_stem = (diff<0.9).sum()
    stem_ratio_t.append(n_stem/n_cells)
    n_paneth = np.sum(is_paneth)
    paneth_ratio_t.append(n_paneth/n_cells)
    # print(n_neighbours)
    # print(n_homotypic)
    # print(n_heterotypic)
    n_homotypic = n_homotypic[n_neighbours>0]
    n_heterotypic = n_heterotypic[n_neighbours>0]
    n_neighbours = n_neighbours[n_neighbours>0]

    homotypic_ratio = n_homotypic / n_neighbours
    heterotypic_ratio = n_heterotypic / n_neighbours

    # plt.hist(heterotypic_ratio,bins=10)
    # plt.show()


    mean_homotypic_ratio = np.mean(homotypic_ratio)
    mean_heterotypic_ratio = np.mean(heterotypic_ratio)
    # print(mean_homotypic_ratio, mean_heterotypic_ratio)

    mean_homotypic_ratio_t.append(mean_homotypic_ratio)
    mean_heterotypic_ratio_t.append(mean_heterotypic_ratio)

    misplaced = (heterotypic_ratio > 0.8).sum()
    misplaced_ratio_t.append(100*misplaced/n_stem)
    # print("misplaced",misplaced)

    # Compute MSD
    sum_sd = 0
    n_sum = 0
    for i in range(n_0):
        if(diff[i]<1.0 and is_paneth[i] == False):
            sum_sd += np.linalg.norm(coords[i,:] - coords_0[i,:])**2
            n_sum += 1
    print("n_sum", n_sum)
    if n_sum == 0:
        MSD_t.append(0)
    else:
        MSD_t.append(sum_sd/n_sum)


    # Dimensions
    Lx_t.append(np.amax(coords[:,0])-np.amin(coords[:,0]))
    Ly_t.append(np.amax(coords[:,1])-np.amin(coords[:,1]))


MSD = np.array(MSD_t)

# # plt.plot(time, mean_homotypic_ratio_t, label='mean homotypic ratio')
# # plt.plot(time, mean_heterotypic_ratio_t, label='mean heterotypic ratio')
# plt.plot(time, misplaced_ratio_t,label='misplaced')
# # plt.xlim(50,850)
# # plt.ylim(0,0.025)
# plt.legend()
# plt.show()


plt.plot(time, n_t,label='n')
plt.xlabel('time')
plt.legend()
plt.show()

# plt.plot(time, Lx_t,label='lx')
# plt.plot(time, Ly_t,label='ly')
# plt.xlabel('time')
# plt.legend()
# plt.show()

plt.plot(time, stem_ratio_t,label='stem cell ratio')
plt.plot(time, paneth_ratio_t,label='paneth ratio')
plt.xlabel('time')
plt.legend()
plt.show()

# Empirical progression is 1000 um^2 every 8.33h
# We approximate one cell diameter to 20um
cell_diameter_length = 20
exp_MSD = time * (1000/(cell_diameter_length**2))/8.33

# Compute sum of squares between experimental and simulated progressions
# sum_sq = 0
# for i in range(time[time<=10].shape[0]):
#     sum_sq += (exp_MSD[i]-MSD[i])**2
#     print(time[i],(exp_MSD[i]-MSD[i])**2)
# print("sum of squares",sum_sq)

sum_sq = np.sum((exp_MSD[time<=10] - MSD[time<=10])**2)
print("sum of squares",sum_sq)

# exp_MSD = time * 2.5/8.33
plt.plot(time,exp_MSD,'-o', label='empirical progression')
plt.xlim(0,11)
plt.ylim(0,4)
# Simulation
plt.plot(time, MSD,'-o', label='MSD')
plt.legend()
plt.show()

# plt.plot(log_t, log_MSD, label='log-log t vs MSD')
# plt.show()

import numpy as np
# import imageio
import sys
from os.path import join, basename, splitext, extsep
from PIL import Image
import glob


file_pattern = sys.argv[1]
# sorted_list = sorted(glob.glob(file_pattern), key=lambda f: int(f.rsplit(extsep, 1)[0].rsplit("_",1)[-1]))
list = glob.glob(file_pattern)


for file_path in list:
    name = splitext(file_path)[0]
    print(name)

    array = np.genfromtxt(file_path,delimiter=", ")
    print(array.shape)
    # print(array)
    # imageio.imwrite(name+".tif", array)
    im = Image.fromarray(array)
    im.save(name+".tif")

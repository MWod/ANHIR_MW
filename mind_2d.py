import sys
import os
import platform

src_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_path)

import numpy as np
from matplotlib import pyplot as plt
from numpy.ctypeslib import ndpointer

import ctypes

current_path = os.path.abspath(os.path.dirname(__file__))
if platform.system().lower() == "linux":
    libs_path = os.path.join(current_path, "libs", "unix")
elif platform.system().lower() == "windows":
    raise ValueError("Windows not supported yet.")
else:
    raise ValueError("Not supported OS.")

if platform.system().lower() == "linux":
    library_path = os.path.join(libs_path, "mind_2d.so")
elif platform.system().lower() == "windows":
    raise ValueError("Windows not supported yet.")
else:
    raise ValueError("Not supported OS.")

lib = ctypes.cdll.LoadLibrary(library_path)
fun = lib.mind_ssd
fun.restype = ctypes.c_double
fun.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_double, ctypes.c_double,
]

def mind_ssd(source, target, **args):
    if np.shape(source) != np.shape(target):
        raise ValueError("Arrays must have the same shape.")

    radius = args['radius']
    sigma = args['sigma']
    src = np.ascontiguousarray(source.astype(np.float64))
    trg = np.ascontiguousarray(target.astype(np.float64))
    y_size, x_size = np.shape(source)
    x_size, y_size = int(x_size), int(y_size)
    radius_x, radius_y = int(radius[1]), int(radius[0])
    sigma_x, sigma_y = float(sigma[1]), float(sigma[0])
    result = fun(source, target, x_size, y_size, radius_x, radius_y, sigma_x, sigma_y)
    return result
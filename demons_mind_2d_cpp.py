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
    library_path = os.path.join(libs_path, "demons_mind_2d.so")
elif platform.system().lower() == "windows":
    raise ValueError("Windows not supported yet.")
else:
    raise ValueError("Not supported OS.")

lib = ctypes.cdll.LoadLibrary(library_path)
fun = lib.demons_mind_2d_so
fun.restype = None
fun.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int,
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_double,
    ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_bool,
    ctypes.c_bool, ctypes.c_bool,
]


def demons_mind_registration(source, target, spacing, update_mode="addition", gradient_mode="symmetric",
                        resolutions=1, early_stop=0, diffusion_sigma=(1.0, 1.0), fluid_sigma=(1.0, 1.0), mind_radius=(1, 1),
                        mind_sigma=(1.0, 1.0), max_iterations=10, tolerance=None,
                        initial_u_x=None, initial_u_y=None, echo=True, return_best=False, iterations=None):
    if (source.shape != target.shape):
        raise ValueError("Source and target must have the same shape.")

    y_size, x_size = np.shape(source)
    grid_x, grid_y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    grid_x = grid_x.astype(np.float64)
    grid_y = grid_y.astype(np.float64)
    es = int(early_stop)
    spacing_x, spacing_y = spacing
    dx, dy = diffusion_sigma
    fx, fy = fluid_sigma
    mx, my = mind_sigma
    mrx, mry = mind_radius
    mrx = int(mrx)
    mry = int(mry)
    u_mode = update_mode.encode('utf-8')
    g_mode = gradient_mode.encode('utf-8')

    if iterations is None:
        iters = np.ascontiguousarray(np.repeat(max_iterations, resolutions)).astype(np.int32)
    else:
        iters = np.ascontiguousarray(iterations).astype(np.int32)

    if tolerance == None:
        tolerance = 0
    if initial_u_x == None or initial_u_y == None:
        i_dfx = np.ascontiguousarray((np.zeros(np.shape(source)) + grid_x).astype(np.float64))
        i_dfy = np.ascontiguousarray((np.zeros(np.shape(source)) + grid_y).astype(np.float64))
        use_init = False
    else:
        i_dfx = np.ascontiguousarray(initial_u_x.astype(np.float64) + grid_x)
        i_dfy = np.ascontiguousarray(initial_u_y.astype(np.float64) + grid_y)
        use_init = True

    df_x = np.ascontiguousarray(np.zeros(np.shape(source), dtype=np.float64))
    df_y = np.ascontiguousarray(np.zeros(np.shape(source), dtype=np.float64))
    src = np.ascontiguousarray(source.astype(np.float64))
    trg = np.ascontiguousarray(target.astype(np.float64))

    if platform.system().lower() == "linux":
        fun(
            src, trg,
            i_dfx, i_dfy,
            df_x, df_y,
            x_size, y_size,
            resolutions,
            iters,
            es,
            fx, fy,
            dx, dy,
            mx, my,
            mrx, mry,
            spacing_x, spacing_y,
            max_iterations, tolerance,
            u_mode, g_mode,
            echo,
            return_best, use_init)
    elif platform.system().lower() == "windows":
        raise ValueError("Windows not supported yet.")
    else:
        raise ValueError("Not supported OS.")
    return df_x - grid_x, df_y - grid_y





from __future__ import division
import numpy as np
cimport numpy as np 

DTYPE = np.float32

ctypedef np.float32_t DTYPE_t

assert sizeof(float) == sizeof(np.float32_t)

cdef extern from "src/manager.hh":
    void depth2normals(DTYPE_t* normal,
                       DTYPE_t* depth,
                       DTYPE_t* intrinsic,
                       int width,
                       int height,
                       int batch)

    
def depth2normals_np(np.ndarray[DTYPE_t, ndim=2] depth,
                     np.ndarray[DTYPE_t, ndim=1] intrinsic):
    cdef int height = depth.shape[0];
    cdef int width = depth.shape[1];
    cdef int batch = 1;

    cdef np.ndarray[DTYPE_t, ndim=3] normal = np.zeros((3, height, width),
        dtype=DTYPE)

    depth2normals(&normal[0, 0, 0],
                  &depth[0, 0],
                  &intrinsic[0],
                  width, height, batch)

    return normal

# cdef extern from "src/manager.hh":
#     cdef cppclass C_GPUD2N "GPUD2N":
#         C_GPUD2N(DTYPE_t*, DTYPE_t*, DTYPE_t*, int, int, int)
#         void depth2normal()
#         void get_normal()

# cdef class GPUAdder:
#     cdef C_GPUD2N* g
#     cdef int height
#     cdef int width
#     cdef int batch

#     def __cinit__(self, np.ndarray[ndim=3, dtype=DTYPE_t] normal, 
#                         np.ndarray[ndim=2, dtype=DTYPE_t] depth,
#                         np.ndarray[ndim=1, dtype=DTYPE_t] intrinsic):

#         self.height = depth.shape[0]
#         self.width = depth.shape[1]
#         self.batch = 1
#         self.g = new C_GPUD2N(&normal[0, 0, 0], &depth[0, 0], &intrinsic[0],
#                               self.height, self.width, self.batch)
#     def depth2normal(self):
#         self.g.depth2normal()

#     def get_normal(self):
#         self.g.get_normal()

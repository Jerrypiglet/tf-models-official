from __future__ import division
from libcpp cimport bool
import numpy as np
cimport numpy as np # for np.ndarray


cdef extern from "renderMesh.h":
    void renderMesh(double* FM, int fNum, 
                    double* VM, int vNum, 
                    double* intrinsics, 
                    int height, int width, 
                    float* depth, bool *mask)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def renderMesh_np(np.ndarray[DTYPE_t, ndim=2] vertices,
                  np.ndarray[DTYPE_t, ndim=2] faces,
                  np.ndarray[DTYPE_t, ndim=1] intrinsic,
                  int height, int width):

    cdef int v_num = vertices.shape[0];
    cdef int f_num = faces.shape[0];

    vertices = vertices.transpose().copy()
    faces = faces.transpose().copy()

    cdef np.ndarray[DTYPE_t, ndim=1] color;
    cdef np.ndarray[np.float32_t, ndim=2] depth = np.zeros((height, width), dtype=np.float32);
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] mask = np.zeros((height, width), dtype=np.uint8);
    cdef bool *mask_bool = <bool*> &mask[0, 0]
    
    renderMesh(&faces[1, 0], f_num,
               &vertices[0, 0], v_num,
               &intrinsic[0],
               height, width,
               &depth[0, 0], 
               mask_bool)
    depth[mask == 0] = -1.0
    
    return depth, mask

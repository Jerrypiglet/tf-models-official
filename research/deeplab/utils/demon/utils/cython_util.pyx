from __future__ import division
import numpy as np
cimport numpy as np 

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef extern from "src/manager.hh":
    void depth2normals(DTYPE_t* normal,
                       DTYPE_t* depth,
                       DTYPE_t* intrinsic,
                       int width,
                       int height,
                       int batch)

    void min_value(DTYPE_t* min_cpu,
                   DTYPE_t* arr_cpu,
                   int length)

def min_value_np(np.ndarray[DTYPE_t, ndim=1] arr):
    cdef int length = arr.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] min_val= np.zeros(1,dtype=DTYPE)
    min_val[0] = arr[0]
    min_value(&min_val[0], &arr[0], length)
    return min_val


def depth2normals_np(np.ndarray[DTYPE_t, ndim=2] depth,
                     np.ndarray[DTYPE_t, ndim=1] intrinsic):
    cdef int height = depth.shape[0];
    cdef int width = depth.shape[1];
    cdef int batch = 1;

    cdef np.ndarray[DTYPE_t, ndim=3] normal = np.zeros((3, height, width), dtype=DTYPE)
    depth2normals(&normal[0, 0, 0],
                  &depth[0, 0],
                  &intrinsic[0],
                  width, height, batch)

    return normal


def gen_depth_map(np.ndarray[DTYPE_t, ndim=2] proj, int height, int width, int get_id):

    cdef np.ndarray[DTYPE_t, ndim=2] depth = np.zeros((height, width), dtype=DTYPE)
    cdef int pix_num = proj.shape[1]
    cdef np.ndarray[np.int32_t, ndim=2] index = np.zeros((height, width),
      dtype=np.int32)

    cdef int x
    cdef int y

    for i in range(pix_num):
        x = int(proj[0, i])
        y = int(proj[1, i])
        if 0 < x and x <= width and 0 < y and y <= height:
            cur_depth = depth[y-1][x-1]
            if cur_depth == 0 or cur_depth > proj[2, i]:
                depth[y-1][x-1] = np.float32(proj[2, i])
                index[y-1][x-1] = np.int32(i)

    if get_id == 1:
      return depth, index
    else:
      return depth


def gen_depth_map_with_ref_depth(np.ndarray[DTYPE_t, ndim=2] proj, 
        int height, int width, np.ndarray[DTYPE_t, ndim=2] ref_depth):

    cdef np.ndarray[DTYPE_t, ndim=2] depth = np.zeros((height, width), dtype=DTYPE)
    cdef int pix_num = proj.shape[1]
    cdef np.ndarray[np.int32_t, ndim=1] index = np.zeros(pix_num, dtype=np.int32)

    cdef int x
    cdef int y

    for i in range(pix_num):
        x = int(proj[0, i])
        y = int(proj[1, i])
        if 0 < x and x <= width and 0 < y and y <= height:
            cur_depth = depth[y-1][x-1]
            if cur_depth == 0 or cur_depth > proj[2, i]:
                depth[y-1][x-1] = np.float32(proj[2, i])
                index[i] = 1
            if np.abs(proj[2, i] - ref_depth[y-1][x-1]) < 0.1:
                index[i] = 1

    return depth, index


def extend_building(np.ndarray[np.int32_t, ndim=2] label_map, 
        int building_id, int sky_id):

    cdef int height = label_map.shape[0] 
    cdef int width = label_map.shape[1]
    cdef int is_building = 0
    cdef int is_sky = 0
    cdef int label
    cdef int top = int(0.06 * height)
    cdef int down = int(height * 2 / 3)
    cdef int cur

    for w in range(width):
        is_building = 0
        cur = top
        # from top down
        for h in range(down):
            label = label_map[h, w]
            if label == building_id:
                cur = h
                is_building = 1

            if 1 == is_building and label in [0, sky_id]:
                label_map[h, w] = building_id

        # from bottom up
        if w > 0.1 * width and w < 0.9 * width:
            is_building = 0
            is_sky = 0
            for h in range(h, down)[::-1]:
                label = label_map[h, w]
                if label == building_id:
                    is_building = 1
                if label == sky_id:
                    is_sky = 1

                if 1 == is_building and label == 0 and 0 == is_sky:
                    label_map[h, w] = building_id

    return label_map

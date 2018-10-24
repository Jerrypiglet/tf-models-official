#ifndef RENDER_MESH_H_
#define RENDER_MESH_H_

#include <iostream>

extern void renderMesh(double *FM, int fNum, 
                       double *VM, int vNum, 
                       double *intrinsics, 
                 int height, int width,
                 float *depth, bool *mask);

#endif

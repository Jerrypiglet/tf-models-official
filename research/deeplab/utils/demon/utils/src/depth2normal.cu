#include <stdio.h>
#include <manager.hh>

typedef float T;

// template <class T>
__device__ inline void compute3dPoint(T* p_3d, 
                                      const int& x,
                                      const int& y, 
                                      const T& depth, 
                                      const T& inv_fx,
                                      const T& inv_fy,
                                      const T& cx,
                                      const T& cy ) {
  p_3d[0] = (x+T(0.5)-cx)*inv_fx*depth;
  p_3d[1] = (y+T(0.5)-cy)*inv_fy*depth;
  p_3d[2] = depth;
}


// template <class T>
inline __device__ void cross(T* v, T* v1, T* v2) {
  v[0] = v1[1]*v2[2] - v1[2]*v2[1];
  v[1] = v1[2]*v2[0] - v1[0]*v2[2];
  v[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

// template <class T>
__device__ void normalize(T* v, int dim) {
  T mag = 0;
  for(int i = 0; i < dim; ++ i)
    mag += v[i] * v[i];

  if(mag == 0)
    return;

  mag = sqrtf(mag);
  for(int i = 0; i < dim; ++ i)
    v[i] /= mag;
}

// template <class T>
void __global__ depthtonormals_kernel(T* out,
                                      const T* depth,
                                      const T* intrinsics,
                                      int x_size,
                                      int y_size,
                                      int z_size ) {
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if( x >= x_size || y >= y_size || z >= z_size )
    return;
  
  // intrinsic is normalized intrinsics 
  const T inv_fx = 1/(intrinsics[4*z+0]*x_size);
  const T inv_fy = 1/(intrinsics[4*z+1]*y_size);
  const T cx = (intrinsics[4*z+2]*x_size);
  const T cy = (intrinsics[4*z+3]*y_size);

  const int xy_size = x_size * y_size;
  const T* depthmap = depth + z * xy_size;
  T* normal = out+3*z*xy_size;

#define DEPTH(y,x) depthmap[(y)*x_size+(x)]
#define NORMAL(c,y,x) normal[(c)*xy_size+(y)*x_size+(x)]

  if( x == 0 || y == 0 || x == x_size-1 || y == y_size-1)
  {
    NORMAL(0,y,x) = 0;
    NORMAL(1,y,x) = 0;
    NORMAL(2,y,x) = 0;
  }
  else
  {
    T d = DEPTH(y,x);
    T d_y0 = DEPTH(y-1,x);
    T d_x0 = DEPTH(y,x-1);
    T d_y1 = DEPTH(y+1,x);
    T d_x1 = DEPTH(y,x+1);
    
    if( d <= 0 || !isfinite(d) || 
        d_y0 <= 0 || !isfinite(d_y0) || 
        d_x0 <= 0 || !isfinite(d_x0) || 
        d_y1 <= 0 || !isfinite(d_y1) || 
        d_x1 <= 0 || !isfinite(d_x1)) {
      NORMAL(0,y,x) = 0;
      NORMAL(1,y,x) = 0;
      NORMAL(2,y,x) = 0;
    }
    else
    {
      T p[3], p_y0[3], p_x0[3], p_y1[3], p_x1[3];
      T normals_0[3], normals_1[3], normals_vec[3];

      compute3dPoint(p, x, y, d, inv_fx, inv_fy, cx, cy);
      compute3dPoint(p_y0, x, y-1, d_y0, inv_fx, inv_fy, cx, cy);
      compute3dPoint(p_x0, x-1, y, d_x0, inv_fx, inv_fy, cx, cy);
      compute3dPoint(p_y1, x, y+1, d_y1, inv_fx, inv_fy, cx, cy);
      compute3dPoint(p_x1, x+1, y, d_x1, inv_fx, inv_fy, cx, cy);

      T diff_x1[3], diff_y1[3], diff_x0[3], diff_y0[3];
      for(int i = 0; i < 3; i ++) {
        diff_x1[i] = p[i] - p_x1[i];
        diff_y1[i] = p_y1[i] - p[i];
        diff_x0[i] = p[i] - p_x0[i];
        diff_y0[i] = p_y0[i] - p[i];
      }

      cross(normals_1, diff_x1, diff_y1);
      cross(normals_0, diff_x0, diff_y0);
      normalize(normals_1, 3);
      normalize(normals_0, 3);
      
      for(int i = 0; i < 3; i ++) 
        normals_vec[i] = normals_1[i] + normals_0[i];

      normalize(normals_vec, 3);
      for(int i = 0; i < 3; i ++)
        NORMAL(i,y,x) = normals_vec[i];
    }
  }

#undef DEPTH
#undef NORMAL
}


// template<class T>
void depth2normals( 
    T* normal_cpu, const T* depth_cpu, const T* intrinsic_cpu,
    int x_size, int y_size, int z_size) {
  // intrinsices ( fx, fy, cx, cy)
  int length = x_size * y_size * z_size;
  int size = length * sizeof(T);
  T* depth_gpu;
  T* normal_gpu;
  T* intrinsic_gpu;
  
  cudaError_t err;
  err = cudaMalloc((void**) &depth_gpu, size); assert(err == 0);
  err = cudaMalloc((void**) &normal_gpu, 3 * size); assert(err == 0);
  err = cudaMalloc((void**) &intrinsic_gpu, 4 * sizeof(T)); assert(err == 0);

  err = cudaMemcpy(depth_gpu, depth_cpu, size, cudaMemcpyHostToDevice);
  err = cudaMemcpy(intrinsic_gpu, intrinsic_cpu, 4 * sizeof(T),
   cudaMemcpyHostToDevice);

  // compute here 
  dim3 block(64, 8, 1);
  dim3 grid;
  grid.x = DIVUP(x_size, block.x);
  grid.y = DIVUP(y_size, block.y);
  grid.z = DIVUP(z_size, block.z);

  depthtonormals_kernel<<<grid, block>>>(
      normal_gpu, depth_gpu, intrinsic_gpu, x_size, y_size, z_size );

  // copy back to cpu
  err = cudaMemcpy(normal_cpu, normal_gpu, 3 * size, cudaMemcpyDeviceToHost);
  assert(err == 0);

  cudaFree(normal_gpu);
  cudaFree(depth_gpu);
  cudaFree(intrinsic_gpu);
}



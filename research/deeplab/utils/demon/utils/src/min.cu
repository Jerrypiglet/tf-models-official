#include <stdio.h>
#include <manager.hh>

typedef float T;

__device__ float atomicMin(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


void __global__ min_value(T* out,
        T* in,
        int len) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if( x >= len)
    return;
  //atomicAdd(out, in[x]);
  atomicMin(out, in[x]);
}


// template<class> 
void min_value(T* min_cpu, 
               const T* arr_cpu, 
               int len) {
  // intrinsices ( fx, fy, cx, cy)
  T* arr_gpu;
  T* min_gpu;
  int size = len * sizeof(T);
  
  cudaError_t err;
  err = cudaMalloc((void**) &arr_gpu, size); 
  assert(err == 0);
  err = cudaMalloc((void**) &min_gpu, sizeof(T)); 
  assert(err == 0);
  err = cudaMemcpy(arr_gpu, arr_cpu, size, cudaMemcpyHostToDevice);
  assert(err == 0);
  err = cudaMemcpy(min_gpu, min_cpu, sizeof(T), cudaMemcpyHostToDevice);
  assert(err == 0);

  // compute here 
  const int threads = 64;
  const int block = DIVUP(len, threads);

  min_value<<<block, threads>>>(min_gpu, arr_gpu, len);
  
  // copy back to cpu
  // printf("val %f ", min_gpu[0]);
  err = cudaMemcpy(min_cpu, min_gpu, sizeof(T), cudaMemcpyDeviceToHost);
  assert(err == 0);

  cudaFree(min_gpu);
  cudaFree(arr_gpu);

}

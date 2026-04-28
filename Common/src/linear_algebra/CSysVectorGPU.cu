/*!
 * \file CSysVectorGPU.cu
 * \brief Implementations of Kernels and Functions for Vector Operations on the GPU
 * \author A. Raj
 * \version 8.3.0 "Harrier"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2024, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

#include "../../include/linear_algebra/CSysVector.hpp"
#include "../../include/linear_algebra/GPUComms.cuh"
#include <algorithm>
#include <vector>

template <class ScalarType>
__global__ void VecAXPYKernel(ScalarType* y, const ScalarType* x, ScalarType a, unsigned long n) {
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += a * x[i];
}

template <class ScalarType>
__global__ void VecScaleKernel(ScalarType* y, ScalarType a, unsigned long n) {
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] *= a;
}

template <class ScalarType>
__global__ void VecCopyKernel(ScalarType* y, const ScalarType* x, unsigned long n) {
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i];
}

template <class ScalarType>
__global__ void VecUpdatePKernel(ScalarType* p, const ScalarType* v, const ScalarType* r, ScalarType beta,
                                 ScalarType omega, unsigned long n) {
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = beta * (p[i] - omega * v[i]) + r[i];
}

template <class ScalarType>
__global__ void DotPartialKernel(const ScalarType* a, const ScalarType* b, ScalarType* partial, unsigned long n) {
  __shared__ ScalarType sdata[256];
  unsigned long tid = threadIdx.x;
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;

  ScalarType sum = 0;
  while (i < n) {
    sum += a[i] * b[i];
    i += blockDim.x * gridDim.x;
  }

  sdata[tid] = sum;
  __syncthreads();

  for (unsigned long s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0) partial[blockIdx.x] = sdata[0];
}

template<class ScalarType>
void CSysVector<ScalarType>::HtDTransfer() const
{
   gpuErrChk(cudaMemcpy((void*)(d_vec_val), (void*)&vec_val[0], (sizeof(ScalarType)*nElm), cudaMemcpyHostToDevice));
}

template<class ScalarType>
void CSysVector<ScalarType>::DtHTransfer() const
{
   gpuErrChk(cudaMemcpy((void*)(&vec_val[0]), (void*)d_vec_val, (sizeof(ScalarType)*nElm), cudaMemcpyDeviceToHost));
}

template<class ScalarType>
void CSysVector<ScalarType>::GPUSetVal(ScalarType val) const
{
   gpuErrChk(cudaMemset((void*)(d_vec_val), val, (sizeof(ScalarType)*nElm)));
}

template<class ScalarType>
void CSysVector<ScalarType>::GPUCopyFrom(const CSysVector<ScalarType>& x) const {
   const int block = 256;
   const int grid = KernelParameters::round_up_division(block, static_cast<int>(nElm));
   VecCopyKernel<<<grid, block>>>(d_vec_val, x.GetDevicePointer(), nElm);
   gpuErrChk(cudaPeekAtLastError());
}

template<class ScalarType>
void CSysVector<ScalarType>::GPUAXPY(ScalarType a, const CSysVector<ScalarType>& x) const {
   const int block = 256;
   const int grid = KernelParameters::round_up_division(block, static_cast<int>(nElm));
   VecAXPYKernel<<<grid, block>>>(d_vec_val, x.GetDevicePointer(), a, nElm);
   gpuErrChk(cudaPeekAtLastError());
}

template<class ScalarType>
void CSysVector<ScalarType>::GPUScale(ScalarType a) const {
   const int block = 256;
   const int grid = KernelParameters::round_up_division(block, static_cast<int>(nElm));
   VecScaleKernel<<<grid, block>>>(d_vec_val, a, nElm);
   gpuErrChk(cudaPeekAtLastError());
}

template<class ScalarType>
void CSysVector<ScalarType>::GPUUpdateP(const CSysVector<ScalarType>& v, const CSysVector<ScalarType>& r,
                                        ScalarType beta, ScalarType omega) const {
   const int block = 256;
   const int grid = KernelParameters::round_up_division(block, static_cast<int>(nElm));
   VecUpdatePKernel<<<grid, block>>>(d_vec_val, v.GetDevicePointer(), r.GetDevicePointer(), beta, omega, nElm);
   gpuErrChk(cudaPeekAtLastError());
}

template<class ScalarType>
ScalarType CSysVector<ScalarType>::GPUDot(const CSysVector<ScalarType>& x) const {
   const int block = 256;
   const int grid = std::min(1024, KernelParameters::round_up_division(block, static_cast<int>(nElm)));

   ScalarType* d_partial = nullptr;
   gpuErrChk(cudaMalloc(reinterpret_cast<void**>(&d_partial), grid * sizeof(ScalarType)));

   DotPartialKernel<<<grid, block>>>(d_vec_val, x.GetDevicePointer(), d_partial, nElm);
   gpuErrChk(cudaPeekAtLastError());

   std::vector<ScalarType> h_partial(grid, ScalarType(0));
   gpuErrChk(cudaMemcpy(h_partial.data(), d_partial, grid * sizeof(ScalarType), cudaMemcpyDeviceToHost));
   gpuErrChk(cudaFree(d_partial));

   ScalarType sum = 0;
   for (const auto& val : h_partial) sum += val;
   return sum;
}

template void CSysVector<su2double>::HtDTransfer() const;
template void CSysVector<su2double>::DtHTransfer() const;
template void CSysVector<su2double>::GPUSetVal(su2double) const;
template void CSysVector<su2double>::GPUCopyFrom(const CSysVector<su2double>&) const;
template void CSysVector<su2double>::GPUAXPY(su2double, const CSysVector<su2double>&) const;
template void CSysVector<su2double>::GPUScale(su2double) const;
template void CSysVector<su2double>::GPUUpdateP(const CSysVector<su2double>&, const CSysVector<su2double>&,
                                                su2double, su2double) const;
template su2double CSysVector<su2double>::GPUDot(const CSysVector<su2double>&) const;

#ifdef USE_MIXED_PRECISION
template void CSysVector<su2mixedfloat>::HtDTransfer() const;
template void CSysVector<su2mixedfloat>::DtHTransfer() const;
template void CSysVector<su2mixedfloat>::GPUSetVal(su2mixedfloat) const;
template void CSysVector<su2mixedfloat>::GPUCopyFrom(const CSysVector<su2mixedfloat>&) const;
template void CSysVector<su2mixedfloat>::GPUAXPY(su2mixedfloat, const CSysVector<su2mixedfloat>&) const;
template void CSysVector<su2mixedfloat>::GPUScale(su2mixedfloat) const;
template void CSysVector<su2mixedfloat>::GPUUpdateP(const CSysVector<su2mixedfloat>&,
                                                    const CSysVector<su2mixedfloat>&, su2mixedfloat,
                                                    su2mixedfloat) const;
template su2mixedfloat CSysVector<su2mixedfloat>::GPUDot(const CSysVector<su2mixedfloat>&) const;
#endif

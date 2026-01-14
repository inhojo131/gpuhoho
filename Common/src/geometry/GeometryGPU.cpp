#include "../../include/geometry/GeometryGPU.hpp"

void GeometryGPU::Allocate(unsigned long nPoint_, unsigned long nEdge_, unsigned short nDim_) {
  nPoint = nPoint_; nEdge = nEdge_; nDim = nDim_;
  /* Coordinates: nPoint * nDim */
  d_coords = GPUMemoryAllocation::gpu_alloc<su2double, true>(nPoint * nDim * sizeof(su2double));
  d_normals = GPUMemoryAllocation::gpu_alloc<su2double, true>(nEdge * nDim * sizeof(su2double));
  d_areas   = GPUMemoryAllocation::gpu_alloc<su2double, true>(nEdge * sizeof(su2double));
  d_edge_v0 = GPUMemoryAllocation::gpu_alloc<unsigned long, true>(nEdge * sizeof(unsigned long));
  d_edge_v1 = GPUMemoryAllocation::gpu_alloc<unsigned long, true>(nEdge * sizeof(unsigned long));
  d_volumes = GPUMemoryAllocation::gpu_alloc<su2double, true>(nPoint * sizeof(su2double));
}

void GeometryGPU::Upload(const su2double* coords, const su2double* normals,
                         const su2double* areas, const unsigned long* edge_v0,
                         const unsigned long* edge_v1, const su2double* volumes) {
  if (coords) GPUMemoryAllocation::gpu_memcpy(d_coords, coords, nPoint * nDim * sizeof(su2double));
  if (normals) GPUMemoryAllocation::gpu_memcpy(d_normals, normals, nEdge * nDim * sizeof(su2double));
  if (areas) GPUMemoryAllocation::gpu_memcpy(d_areas, areas, nEdge * sizeof(su2double));
  if (edge_v0) GPUMemoryAllocation::gpu_memcpy(d_edge_v0, edge_v0, nEdge * sizeof(unsigned long));
  if (edge_v1) GPUMemoryAllocation::gpu_memcpy(d_edge_v1, edge_v1, nEdge * sizeof(unsigned long));
  if (volumes) GPUMemoryAllocation::gpu_memcpy(d_volumes, volumes, nPoint * sizeof(su2double));
}

void GeometryGPU::Free() {
  GPUMemoryAllocation::gpu_free(d_coords);   d_coords = nullptr;
  GPUMemoryAllocation::gpu_free(d_normals);  d_normals = nullptr;
  GPUMemoryAllocation::gpu_free(d_areas);    d_areas = nullptr;
  GPUMemoryAllocation::gpu_free(d_edge_v0);  d_edge_v0 = nullptr;
  GPUMemoryAllocation::gpu_free(d_edge_v1);  d_edge_v1 = nullptr;
  GPUMemoryAllocation::gpu_free(d_volumes);  d_volumes = nullptr;
  nPoint = nEdge = 0; nDim = 0;
}

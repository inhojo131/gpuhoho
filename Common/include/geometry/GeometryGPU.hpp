#pragma once

/*
 * Lightweight GPU geometry buffer wrapper.
 * Holds raw device pointers and helpers to allocate, upload, and free.
 * This does not replace CGeometry; it is a staging area for future CUDA kernels.
 */

#include "../code_config.hpp"                /* su2double */
#include "../toolboxes/allocation_toolbox.hpp" /* GPUMemoryAllocation helpers */
#include <cstddef>

struct GeometryGPU {
  /* Device buffers (cudaMalloc-managed). */
  su2double* d_coords = nullptr;     /* nPoint * nDim, packed x,y,(z) */
  su2double* d_normals = nullptr;    /* nEdge * nDim */
  su2double* d_areas = nullptr;      /* nEdge */
  unsigned long* d_edge_v0 = nullptr; /* nEdge */
  unsigned long* d_edge_v1 = nullptr; /* nEdge */
  su2double* d_volumes = nullptr;    /* nPoint */

  /* Sizes for convenience. */
  unsigned long nPoint = 0;
  unsigned long nEdge = 0;
  unsigned short nDim = 0;

  GeometryGPU() = default;

  /* Allocate GPU buffers for the given sizes. */
  void Allocate(unsigned long nPoint_, unsigned long nEdge_, unsigned short nDim_);

  /* Upload host-side contiguous arrays into the device buffers. */
  void Upload(const su2double* coords, const su2double* normals,
              const su2double* areas, const unsigned long* edge_v0,
              const unsigned long* edge_v1, const su2double* volumes);

  /* Free all device buffers. Safe to call multiple times. */
  void Free();
};

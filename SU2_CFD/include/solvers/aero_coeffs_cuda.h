#pragma once

#include <cuda_runtime_api.h>

enum class GPUFlowBC : unsigned short {
  FARFIELD = 0,
  INLET = 1,
  OUTLET = 2,
  SYMMETRY = 3,
  EULER_WALL = 4
};

/* Host helper: node-based Roe flux + residual + explicit Euler update. */
extern "C" cudaError_t ComputeEulerRoeUpdateHost(const unsigned long* h_nodeEdgeOffsets,
                                                 const unsigned long* h_nodeEdges,
                                                 const unsigned long* h_edgeNode0,
                                                 const unsigned long* h_edgeNode1,
                                                 const double* h_edgeNormals,
                                                 const double* h_solution,
                                                 const double* h_dt,
                                                 const double* h_volume,
                                                 const double* h_residual_extra,
                                                 const double* h_residual_trunc,
                                                 unsigned long nPointDomain,
                                                 unsigned long nPoint,
                                                 unsigned long nEdge,
                                                 unsigned short nVar,
                                                 unsigned short nDim,
                                                 double gamma,
                                                 double* h_solution_out,
                                                 double* h_residual_out,
                                                 cudaStream_t stream = 0);

/* Host helper: node-based Roe (convective) + viscous + boundary update for NS. */
extern "C" cudaError_t ComputeNSRoeUpdateHost(const unsigned long* h_nodeEdgeOffsets,
                                              const unsigned long* h_nodeEdges,
                                              const unsigned long* h_edgeNode0,
                                              const unsigned long* h_edgeNode1,
                                              const double* h_edgeNormals,
                                              const double* h_solution,
                                              const double* h_gradU,
                                              const double* h_dt,
                                              const double* h_volume,
                                              const double* h_residual_extra,
                                              const double* h_residual_trunc,
                                              const unsigned long* h_bndOffsets,
                                              const double* h_bndNormals,
                                              const unsigned short* h_bndTypes,
                                              const double* h_bndParams,
                                              unsigned long nPointDomain,
                                              unsigned long nPoint,
                                              unsigned long nEdge,
                                              unsigned long nBnd,
                                              unsigned short nVar,
                                              unsigned short nDim,
                                              double gamma,
                                              double mu,
                                              double* h_solution_out,
                                              double* h_residual_out,
                                              cudaStream_t stream = 0);

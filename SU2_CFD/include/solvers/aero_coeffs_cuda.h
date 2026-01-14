#pragma once

// CUDA helper to zero AeroCoeffs arrays on device for testing/prototyping.
// This is declared extern "C" to match the definition in aero_coeffs_cuda.cu.
#include <cuda_runtime_api.h>

extern "C" cudaError_t SetZeroHostArrays(double* CD, double* CL, double* CSF, double* CEff,
                                         double* CFx, double* CFy, double* CFz,
                                         double* CMx, double* CMy, double* CMz,
                                         double* CoPx, double* CoPy, double* CoPz,
                                         double* CT, double* CQ, double* CMerit,
                                         int n, cudaStream_t stream);

/* Placeholder stub for GPU residual assembly. Currently unimplemented and
   returns cudaErrorNotSupported so callers can link successfully. */
extern "C" cudaError_t ComputeResidualGPU(const void* geometry,
                                          const void* solution,
                                          const void* numerics,
                                          cudaStream_t stream);

/* Placeholder viscous flux builder (see aero_coeffs_cuda.cu). */
extern "C" cudaError_t ComputeViscousFluxesGPU(const unsigned long* edgeNode0,
                                               const unsigned long* edgeNode1,
                                               const double* edgeNormal,
                                               const double* edgeArea,
                                               const double* mu_edge,
                                               const double* gradU,
                                               double* edgeFluxes,
                                               unsigned long nEdge,
                                               unsigned short nVar,
                                               cudaStream_t stream);

/* Host helper: upload flattened arrays, run viscous kernel, copy back fluxes. */
extern "C" cudaError_t ComputeViscousFluxesHost(const unsigned long* h_edgeNode0,
                                                const unsigned long* h_edgeNode1,
                                                const double* h_edgeNormal,
                                                const double* h_edgeArea,
                                                const double* h_mu_edge,
                                                const double* h_gradU,
                                                size_t gradU_count,
                                                double* h_edgeFluxes,
                                                unsigned long nEdge,
                                                unsigned short nVar,
                                                cudaStream_t stream = 0);

/* Extended viscous flux interface: per-edge primitive variables, gradients, viscosities, and heat conductivity. */
extern "C" cudaError_t ComputeViscousFluxesGPUFull(const unsigned long* edgeNode0,
                                                   const unsigned long* edgeNode1,
                                                   const double* edgeNormal,
                                                   const double* edgeArea,
                                                   const double* mu_lam,
                                                   const double* mu_turb,
                                                   const double* kappa,
                                                   const double* prim,       /* size nEdge * nPrim */
                                                   const double* gradPrim,   /* size nEdge * nGradPrim * 3 */
                                                   const double* tau_wall,   /* size nEdge */
                                                   unsigned long nEdge,
                                                   unsigned short nVar,
                                                   unsigned short nPrim,
                                                   unsigned short nGradPrim,
                                                   unsigned short nDim,
                                                   int qcr_enabled,
                                                   double* edgeFluxes,
                                                   cudaStream_t stream = 0);
extern "C" cudaError_t ComputeViscousFluxesHostFull(const unsigned long* h_edgeNode0,
                                                    const unsigned long* h_edgeNode1,
                                                    const double* h_edgeNormal,
                                                    const double* h_edgeArea,
                                                    const double* h_mu_lam,
                                                    const double* h_mu_turb,
                                                    const double* h_kappa,
                                                    const double* h_prim,
                                                    const double* h_gradPrim,
                                                    const double* h_tau_wall,
                                                    size_t gradPrim_count,
                                                    unsigned long nEdge,
                                                    unsigned short nVar,
                                                    unsigned short nPrim,
                                                    unsigned short nGradPrim,
                                                    unsigned short nDim,
                                                    int qcr_enabled,
                                                    double* h_edgeFluxes,
                                                    cudaStream_t stream = 0);

extern "C" cudaError_t SumEdgeFluxesGPU(const unsigned long* nodeEdgeOffsets,
                                        const unsigned long* nodeEdges,
                                        const unsigned long* edgeNode0,
                                        const unsigned long* edgeNode1,
                                        const double* edgeFluxes,
                                        double* linSysRes,
                                        unsigned long nPoint,
                                        unsigned short nVar,
                                        cudaStream_t stream);

// Minimal GPU state for CEulerSolver (skeleton).
struct CEulerSolverGPUState;
extern "C" CEulerSolverGPUState* CreateEulerSolverGPU(int nZone);
extern "C" void DestroyEulerSolverGPU(CEulerSolverGPUState* state);

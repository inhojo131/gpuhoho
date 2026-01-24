// Minimal CUDA example: setZero for AeroCoeffsArray on device buffers.
// This is a stub to illustrate how a GPU kernel could zero the arrays.
// Real integration requires wiring Device pointers and build system updates.

#include <cuda_runtime.h>
#include <cmath>
#include "../../include/solvers/aero_coeffs_cuda.h"
#if 0
/* -------------------------------------------------------------------------
 * Minimal viscous-flux GPU scaffold.
 *
 * This is a bare-bones kernel that walks edges and writes a placeholder
 * viscous flux (currently zero) into an edge-flux buffer.  It is intended
 * as a starting point for porting CAvgGrad_Flow::ComputeResidual:
 *
 * Required flattened host inputs (to be cudaMemcpy’ed by the caller):
 *   - edgeNode0/edgeNode1 (size nEdge)   : endpoint indices for each edge
 *   - edgeNormal[3*nEdge]                : geometric normal per edge
 *   - edgeArea (size nEdge)              : edge area/length
 *   - mu_edge (size nEdge)               : dynamic viscosity per edge
 *   - gradU[*] (per-edge gradients or per-node gradients sampled at edge)
 *   - nVar                               : number of conserved variables
 * Output:
 *   - edgeFluxes (size nEdge * nVar)     : viscous contribution per edge
 *
 * The kernel below simply zeroes the output; hook in the real viscous
 * formulas as you port the CPU implementation.
 * ------------------------------------------------------------------------- */
/* Legacy simplified viscous kernel: flux = -mu*area*(gradU·n) for each variable. */
__global__ void ViscousFluxKernel(const unsigned long* edgeNode0,
                                  const unsigned long* edgeNode1,
                                  const double* edgeNormal,
                                  const double* edgeArea,
                                  const double* mu_edge,
                                  const double* gradU,
                                  unsigned long nEdge,
                                  unsigned short nVar,
                                  double* edgeFluxes) {
  const unsigned long iEdge = blockIdx.x * blockDim.x + threadIdx.x;
  if (iEdge >= nEdge) return;

  const double nx = edgeNormal[3 * iEdge + 0];
  const double ny = edgeNormal[3 * iEdge + 1];
  const double nz = edgeNormal[3 * iEdge + 2];
  const double mu = mu_edge[iEdge];
  const double area = edgeArea[iEdge];
  const double scale = -mu * area;
  const double* grad = gradU + iEdge * static_cast<unsigned long>(nVar) * 3u;
  double* flux = edgeFluxes + iEdge * nVar;

  for (unsigned short v = 0; v < nVar; ++v) {
    const double gx = grad[v * 3u + 0];
    const double gy = grad[v * 3u + 1];
    const double gz = grad[v * 3u + 2];
    flux[v] = scale * (gx * nx + gy * ny + gz * nz);
  }
  (void)edgeNode0;
  (void)edgeNode1;
}

/* Full viscous kernel using primitive variables and gradients (simplified Navier-Stokes form). */
__global__ void ViscousFluxKernelFull(const unsigned long* edgeNode0,
                                      const unsigned long* edgeNode1,
                                      const double* edgeNormal,
                                      const double* edgeArea,
                                      const double* mu_lam,
                                      const double* mu_turb,
                                      const double* kappa,
                                      const double* prim,
                                      const double* gradPrim,
                                      const double* tau_wall,
                                      unsigned long nEdge,
                                      unsigned short nVar,
                                      unsigned short nPrim,
                                      unsigned short nGradPrim,
                                      unsigned short nDim,
                                      int qcr_enabled,
                                      double* edgeFluxes) {
  const unsigned long iEdge = blockIdx.x * blockDim.x + threadIdx.x;
  if (iEdge >= nEdge) return;

  const double nx = edgeNormal[3 * iEdge + 0];
  const double ny = edgeNormal[3 * iEdge + 1];
  const double nz = edgeNormal[3 * iEdge + 2];
  const double area = edgeArea ? edgeArea[iEdge] : 0.0;
  const double inv_area = (area > 0.0) ? (1.0 / area) : 0.0;
  const double unit_nx = nx * inv_area;
  const double unit_ny = ny * inv_area;
  const double unit_nz = nz * inv_area;
  const double mu   = mu_lam[iEdge] + (mu_turb ? mu_turb[iEdge] : 0.0);
  const double kap  = kappa ? kappa[iEdge] : 0.0;

  const double* pEdge = prim + static_cast<unsigned long>(iEdge) * nPrim;
  const double u = (nDim > 0 && nPrim > 1) ? pEdge[1] : 0.0;
  const double v = (nDim > 1 && nPrim > 2) ? pEdge[2] : 0.0;
  const double w = (nDim > 2 && nPrim > 3) ? pEdge[3] : 0.0;

  /* gradPrim flattened: ((edge*nGradPrim + var)*3 + dim) */
  const double* gBase = gradPrim + (static_cast<unsigned long>(iEdge) * nGradPrim * 3u);
  auto G = [&](unsigned short var, unsigned short dim)->double {
    return gBase[static_cast<unsigned long>(var)*3u + dim];
  };

  double gradVel[3][3] = {{0.0}};
  for (unsigned short iDim = 0; iDim < nDim; ++iDim) {
    for (unsigned short jDim = 0; jDim < nDim; ++jDim) {
      const unsigned short var = static_cast<unsigned short>(1 + iDim);
      gradVel[iDim][jDim] = (var < nGradPrim) ? G(var, jDim) : 0.0;
    }
  }

  const double divU = (nDim >= 1 ? gradVel[0][0] : 0.0) +
                      (nDim >= 2 ? gradVel[1][1] : 0.0) +
                      (nDim == 3 ? gradVel[2][2] : 0.0);

  double tau[3][3] = {{0.0}};
  const double pTerm = (2.0/3.0) * (divU * mu);
  for (unsigned short iDim = 0; iDim < nDim; ++iDim) {
    for (unsigned short jDim = 0; jDim < nDim; ++jDim) {
      tau[iDim][jDim] = mu * (gradVel[iDim][jDim] + gradVel[jDim][iDim]);
    }
    tau[iDim][iDim] -= pTerm;
  }

  if (qcr_enabled) {
    double factor = 0.0;
    for (unsigned short iDim = 0; iDim < nDim; ++iDim)
      for (unsigned short jDim = 0; jDim < nDim; ++jDim)
        factor += gradVel[iDim][jDim] * gradVel[iDim][jDim];
    factor = 1.0 / sqrt(fmax(factor, 1.0e-10));

    double tauQCR[3][3] = {{0.0}};
    for (unsigned short iDim = 0; iDim < nDim; ++iDim) {
      for (unsigned short jDim = 0; jDim < nDim; ++jDim) {
        for (unsigned short kDim = 0; kDim < nDim; ++kDim) {
          const double O_ik = (gradVel[iDim][kDim] - gradVel[kDim][iDim]) * factor;
          const double O_jk = (gradVel[jDim][kDim] - gradVel[kDim][jDim]) * factor;
          tauQCR[iDim][jDim] += O_ik * tau[jDim][kDim] + O_jk * tau[iDim][kDim];
        }
      }
    }

    const double c_cr1 = 0.3;
    for (unsigned short iDim = 0; iDim < nDim; ++iDim)
      for (unsigned short jDim = 0; jDim < nDim; ++jDim)
        tau[iDim][jDim] -= c_cr1 * tauQCR[iDim][jDim];
  }

  if (tau_wall && tau_wall[iEdge] > 0.0 && area > 0.0) {
    double proj[3] = {0.0, 0.0, 0.0};
    for (unsigned short iDim = 0; iDim < nDim; ++iDim) {
      proj[iDim] = tau[iDim][0] * unit_nx +
                   (nDim > 1 ? tau[iDim][1] * unit_ny : 0.0) +
                   (nDim > 2 ? tau[iDim][2] * unit_nz : 0.0);
    }
    const double normalProj = proj[0] * unit_nx + proj[1] * unit_ny + proj[2] * unit_nz;
    for (unsigned short iDim = 0; iDim < nDim; ++iDim) {
      const double ucomp = (iDim == 0 ? unit_nx : (iDim == 1 ? unit_ny : unit_nz));
      proj[iDim] -= normalProj * ucomp;
    }
    const double wallShear = sqrt(proj[0]*proj[0] + proj[1]*proj[1] + proj[2]*proj[2]);
    if (wallShear > 0.0) {
      const double scale = tau_wall[iEdge] / wallShear;
      for (unsigned short iDim = 0; iDim < nDim; ++iDim)
        for (unsigned short jDim = 0; jDim < nDim; ++jDim)
          tau[iDim][jDim] *= scale;
    }
  }

  double heat_flux[3] = {0.0, 0.0, 0.0};
  if (kappa && nGradPrim > 0) {
    heat_flux[0] = kap * G(0, 0);
    heat_flux[1] = (nDim > 1) ? kap * G(0, 1) : 0.0;
    heat_flux[2] = (nDim > 2) ? kap * G(0, 2) : 0.0;
  }

  double* flux = edgeFluxes + iEdge * nVar;
  if (nVar >= 1) flux[0] = 0.0;

  for (unsigned short m = 0; m < nDim && (1 + m) < nVar; ++m) {
    double proj = 0.0;
    proj += tau[0][m] * nx;
    proj += (nDim > 1 ? tau[1][m] * ny : 0.0);
    proj += (nDim > 2 ? tau[2][m] * nz : 0.0);
    flux[1 + m] = proj;
  }

  if (nVar >= nDim + 2) {
    const double vel[3] = {u, v, w};
    double proj = 0.0;
    for (unsigned short d = 0; d < nDim; ++d) {
      double tau_dot_v = 0.0;
      for (unsigned short k = 0; k < nDim; ++k) tau_dot_v += tau[d][k] * vel[k];
      const double ncomp = (d == 0 ? nx : (d == 1 ? ny : nz));
      proj += (tau_dot_v + heat_flux[d]) * ncomp;
    }
    flux[nDim + 1] = proj;
  }

  for (unsigned short v = nDim + 2; v < nVar; ++v) flux[v] = 0.0;

  (void)edgeNode0;
  (void)edgeNode1;
}

extern "C" cudaError_t ComputeViscousFluxesGPU(const unsigned long* edgeNode0,
                                               const unsigned long* edgeNode1,
                                               const double* edgeNormal,
                                               const double* edgeArea,
                                               const double* mu_edge,
                                               const double* gradU,
                                               double* edgeFluxes,
                                               unsigned long nEdge,
                                               unsigned short nVar,
                                               cudaStream_t stream) {
  if (!edgeNode0 || !edgeNode1 || !edgeNormal || !edgeArea || !mu_edge || !gradU || !edgeFluxes) {
    return cudaErrorInvalidValue;
  }

  const int block = 256;
  const int grid = static_cast<int>((nEdge + block - 1) / block);
  ViscousFluxKernel<<<grid, block, 0, stream>>>(edgeNode0, edgeNode1,
                                                edgeNormal, edgeArea, mu_edge,
                                                gradU, nEdge, nVar, edgeFluxes);
  return cudaGetLastError();
}

/* Host helper: take flattened host arrays, upload, launch ViscousFluxKernel,
 * and copy edgeFluxes back to host.  This keeps the C++ caller simple while
 * the kernel is still a placeholder. */
extern "C" cudaError_t ComputeViscousFluxesHost(const unsigned long* h_edgeNode0,
                                                const unsigned long* h_edgeNode1,
                                                const double* h_edgeNormal,
                                                const double* h_edgeArea,
                                                const double* h_mu_edge,
                                                const double* h_gradU,
                                                size_t gradU_count, /* number of doubles in h_gradU */
                                                double* h_edgeFluxes,
                                                unsigned long nEdge,
                                                unsigned short nVar,
                                                cudaStream_t stream) {
  if (!h_edgeNode0 || !h_edgeNode1 || !h_edgeNormal || !h_edgeArea || !h_mu_edge || !h_edgeFluxes) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = cudaSuccess;
  const size_t edgeBytes    = nEdge * sizeof(unsigned long);
  const size_t normalBytes  = 3 * nEdge * sizeof(double);
  const size_t areaBytes    = nEdge * sizeof(double);
  const size_t muBytes      = nEdge * sizeof(double);
  const size_t fluxBytes    = static_cast<size_t>(nEdge) * nVar * sizeof(double);
  const size_t gradBytes    = h_gradU && gradU_count ? gradU_count * sizeof(double) : 0;

  unsigned long *d_e0=nullptr, *d_e1=nullptr;
  double *d_normal=nullptr, *d_area=nullptr, *d_mu=nullptr, *d_grad=nullptr, *d_flux=nullptr;

  err = cudaMalloc(&d_e0, edgeBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_e1, edgeBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_normal, normalBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_area, areaBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_mu, muBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_flux, fluxBytes); if (err) goto cleanup;
  if (gradBytes) { err = cudaMalloc(&d_grad, gradBytes); if (err) goto cleanup; }

  err = cudaMemcpyAsync(d_e0, h_edgeNode0, edgeBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_e1, h_edgeNode1, edgeBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_normal, h_edgeNormal, normalBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_area, h_edgeArea, areaBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_mu, h_mu_edge, muBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  if (gradBytes) {
    err = cudaMemcpyAsync(d_grad, h_gradU, gradBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  } else {
    /* allocate tiny zero grad if not provided */
    err = cudaMalloc(&d_grad, sizeof(double)); if (err) goto cleanup;
    double zero = 0.0;
    err = cudaMemcpyAsync(d_grad, &zero, sizeof(double), cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  }

  err = ComputeViscousFluxesGPU(d_e0, d_e1, d_normal, d_area, d_mu, d_grad, d_flux,
                                nEdge, nVar, stream);
  if (err) goto cleanup;

  err = cudaMemcpyAsync(h_edgeFluxes, d_flux, fluxBytes, cudaMemcpyDeviceToHost, stream);
  if (err) goto cleanup;
  err = cudaStreamSynchronize(stream);

cleanup:
  if (d_e0) cudaFree(d_e0);
  if (d_e1) cudaFree(d_e1);
  if (d_normal) cudaFree(d_normal);
  if (d_area) cudaFree(d_area);
  if (d_mu) cudaFree(d_mu);
  if (d_grad) cudaFree(d_grad);
  if (d_flux) cudaFree(d_flux);
  return err;
}

/* Extended GPU launcher: expects device pointers with full primitive and gradient data. */
extern "C" cudaError_t ComputeViscousFluxesGPUFull(const unsigned long* edgeNode0,
                                                   const unsigned long* edgeNode1,
                                                   const double* edgeNormal,
                                                   const double* edgeArea,
                                                   const double* mu_lam,
                                                   const double* mu_turb,
                                                   const double* kappa,
                                                   const double* prim,
                                                   const double* gradPrim,
                                                   const double* tau_wall,
                                                   unsigned long nEdge,
                                                   unsigned short nVar,
                                                   unsigned short nPrim,
                                                   unsigned short nGradPrim,
                                                   unsigned short nDim,
                                                   int qcr_enabled,
                                                   double* edgeFluxes,
                                                   cudaStream_t stream) {
  if (!edgeNode0 || !edgeNode1 || !edgeNormal || !edgeArea || !mu_lam || !prim || !gradPrim || !edgeFluxes) {
    return cudaErrorInvalidValue;
  }
  const int block = 256;
  const int grid  = static_cast<int>((nEdge + block - 1) / block);
  ViscousFluxKernelFull<<<grid, block, 0, stream>>>(edgeNode0, edgeNode1, edgeNormal, edgeArea,
                                                    mu_lam, mu_turb, kappa, prim, gradPrim, tau_wall,
                                                    nEdge, nVar, nPrim, nGradPrim, nDim, qcr_enabled,
                                                    edgeFluxes);
  return cudaGetLastError();
}

/* Host helper for the extended interface: flattens/allocates/copies then calls ComputeViscousFluxesGPUFull. */
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
                                                    cudaStream_t stream) {
  if (!h_edgeNode0 || !h_edgeNode1 || !h_edgeNormal || !h_edgeArea || !h_mu_lam || !h_prim || !h_gradPrim || !h_edgeFluxes) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = cudaSuccess;
  const size_t edgeBytes   = nEdge * sizeof(unsigned long);
  const size_t normalBytes = 3 * nEdge * sizeof(double);
  const size_t areaBytes   = nEdge * sizeof(double);
  const size_t muBytes     = nEdge * sizeof(double);
  const size_t primBytes   = static_cast<size_t>(nEdge) * nPrim * sizeof(double);
  const size_t gradBytes   = gradPrim_count ? gradPrim_count * sizeof(double)
                                           : static_cast<size_t>(nEdge) * nGradPrim * 3u * sizeof(double);
  const size_t fluxBytes   = static_cast<size_t>(nEdge) * nVar * sizeof(double);

  unsigned long *d_e0=nullptr, *d_e1=nullptr;
  double *d_normal=nullptr, *d_area=nullptr, *d_mul=nullptr, *d_mut=nullptr, *d_kap=nullptr;
  double *d_prim=nullptr, *d_grad=nullptr, *d_tau=nullptr, *d_flux=nullptr;

  err = cudaMalloc(&d_e0, edgeBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_e1, edgeBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_normal, normalBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_area, areaBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_mul, muBytes); if (err) goto cleanup;
  if (h_mu_turb) { err = cudaMalloc(&d_mut, muBytes); if (err) goto cleanup; }
  if (h_kappa)   { err = cudaMalloc(&d_kap, muBytes); if (err) goto cleanup; }
  err = cudaMalloc(&d_prim, primBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_grad, gradBytes); if (err) goto cleanup;
  if (h_tau_wall) { err = cudaMalloc(&d_tau, muBytes); if (err) goto cleanup; }
  err = cudaMalloc(&d_flux, fluxBytes); if (err) goto cleanup;

  err = cudaMemcpyAsync(d_e0, h_edgeNode0, edgeBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_e1, h_edgeNode1, edgeBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_normal, h_edgeNormal, normalBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_area, h_edgeArea, areaBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_mul, h_mu_lam, muBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  if (d_mut) { err = cudaMemcpyAsync(d_mut, h_mu_turb, muBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup; }
  if (d_kap) { err = cudaMemcpyAsync(d_kap, h_kappa,   muBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup; }
  err = cudaMemcpyAsync(d_prim, h_prim, primBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_grad, h_gradPrim, gradBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  if (d_tau) { err = cudaMemcpyAsync(d_tau, h_tau_wall, muBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup; }

  err = ComputeViscousFluxesGPUFull(d_e0, d_e1, d_normal, d_area,
                                    d_mul, d_mut, d_kap,
                                    d_prim, d_grad, d_tau,
                                    nEdge, nVar, nPrim, nGradPrim, nDim,
                                    qcr_enabled, d_flux, stream);
  if (err) goto cleanup;

  err = cudaMemcpyAsync(h_edgeFluxes, d_flux, fluxBytes, cudaMemcpyDeviceToHost, stream);
  if (err) goto cleanup;
  err = cudaStreamSynchronize(stream);

cleanup:
  if (d_e0) cudaFree(d_e0);
  if (d_e1) cudaFree(d_e1);
  if (d_normal) cudaFree(d_normal);
  if (d_area) cudaFree(d_area);
  if (d_mul) cudaFree(d_mul);
  if (d_mut) cudaFree(d_mut);
  if (d_kap) cudaFree(d_kap);
  if (d_prim) cudaFree(d_prim);
  if (d_grad) cudaFree(d_grad);
  if (d_tau) cudaFree(d_tau);
  if (d_flux) cudaFree(d_flux);
  return err;
}

#endif

__device__ __forceinline__ void compute_flux_euler(const double* U, double gamma,
                                                   double nx, double ny, double nz,
                                                   unsigned short nDim, unsigned short nVar,
                                                   double* F) {
  const double rho = U[0];
  const double inv_rho = 1.0 / rho;
  const double u = U[1] * inv_rho;
  const double v = U[2] * inv_rho;
  const double w = (nDim == 3) ? U[3] * inv_rho : 0.0;
  const double E = U[nVar - 1] * inv_rho;
  const double kinetic = 0.5 * (u*u + v*v + w*w);
  const double p = (gamma - 1.0) * rho * (E - kinetic);
  const double vn = u * nx + v * ny + w * nz;

  F[0] = rho * vn;
  F[1] = rho * u * vn + p * nx;
  F[2] = rho * v * vn + p * ny;
  if (nDim == 3) {
    F[3] = rho * w * vn + p * nz;
  }
  F[nVar - 1] = (rho * E + p) * vn;
}

__device__ __forceinline__ void build_tangent(const double nx, const double ny, const double nz,
                                              double& t1x, double& t1y, double& t1z,
                                              double& t2x, double& t2y, double& t2z) {
  double ax = 0.0, ay = 0.0, az = 0.0;
  if (fabs(nx) < 0.9) {
    ax = 1.0;
  } else {
    ay = 1.0;
  }
  t1x = ny * az - nz * ay;
  t1y = nz * ax - nx * az;
  t1z = nx * ay - ny * ax;
  const double norm = sqrt(t1x*t1x + t1y*t1y + t1z*t1z);
  const double inv_norm = (norm > 0.0) ? 1.0 / norm : 0.0;
  t1x *= inv_norm; t1y *= inv_norm; t1z *= inv_norm;

  t2x = ny * t1z - nz * t1y;
  t2y = nz * t1x - nx * t1z;
  t2z = nx * t1y - ny * t1x;
}

__device__ __forceinline__ void compute_roe_flux(const double* UL, const double* UR,
                                                 const double nx, const double ny, const double nz,
                                                 unsigned short nDim, unsigned short nVar,
                                                 double gamma, double* flux) {
  double FL[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double FR[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

  compute_flux_euler(UL, gamma, nx, ny, nz, nDim, nVar, FL);
  compute_flux_euler(UR, gamma, nx, ny, nz, nDim, nVar, FR);

  const double nmag = sqrt(nx*nx + ny*ny + nz*nz);
  if (nmag <= 0.0) {
    for (unsigned short v = 0; v < nVar; ++v) flux[v] = 0.0;
    return;
  }
  const double inv_nmag = 1.0 / nmag;
  const double nxh = nx * inv_nmag;
  const double nyh = ny * inv_nmag;
  const double nzh = nz * inv_nmag;

  const double rhoL = UL[0];
  const double rhoR = UR[0];
  const double inv_rhoL = 1.0 / rhoL;
  const double inv_rhoR = 1.0 / rhoR;
  const double uL = UL[1] * inv_rhoL;
  const double vL = UL[2] * inv_rhoL;
  const double wL = (nDim == 3) ? UL[3] * inv_rhoL : 0.0;
  const double uR = UR[1] * inv_rhoR;
  const double vR = UR[2] * inv_rhoR;
  const double wR = (nDim == 3) ? UR[3] * inv_rhoR : 0.0;
  const double EL = UL[nVar - 1] * inv_rhoL;
  const double ER = UR[nVar - 1] * inv_rhoR;
  const double pL = (gamma - 1.0) * rhoL * (EL - 0.5 * (uL*uL + vL*vL + wL*wL));
  const double pR = (gamma - 1.0) * rhoR * (ER - 0.5 * (uR*uR + vR*vR + wR*wR));
  const double HL = (UL[nVar - 1] + pL) * inv_rhoL;
  const double HR = (UR[nVar - 1] + pR) * inv_rhoR;

  const double sqrt_rhoL = sqrt(rhoL);
  const double sqrt_rhoR = sqrt(rhoR);
  const double denom = sqrt_rhoL + sqrt_rhoR;
  const double inv_denom = (denom > 0.0) ? 1.0 / denom : 0.0;

  const double uBar = (sqrt_rhoL * uL + sqrt_rhoR * uR) * inv_denom;
  const double vBar = (sqrt_rhoL * vL + sqrt_rhoR * vR) * inv_denom;
  const double wBar = (sqrt_rhoL * wL + sqrt_rhoR * wR) * inv_denom;
  const double HBar = (sqrt_rhoL * HL + sqrt_rhoR * HR) * inv_denom;
  const double q2Bar = uBar*uBar + vBar*vBar + wBar*wBar;
  double aBar2 = (gamma - 1.0) * (HBar - 0.5 * q2Bar);
  if (aBar2 < 1e-14) aBar2 = 1e-14;
  const double aBar = sqrt(aBar2);
  const double rhoBar = sqrt_rhoL * sqrt_rhoR;
  const double qBar = uBar * nxh + vBar * nyh + wBar * nzh;

  const double drho = rhoR - rhoL;
  const double du = uR - uL;
  const double dv = vR - vL;
  const double dw = wR - wL;
  const double dqn = du * nxh + dv * nyh + dw * nzh;
  const double dp = pR - pL;

  double t1x = 0.0, t1y = 0.0, t1z = 0.0;
  double t2x = 0.0, t2y = 0.0, t2z = 0.0;
  if (nDim == 3) {
    build_tangent(nxh, nyh, nzh, t1x, t1y, t1z, t2x, t2y, t2z);
  } else {
    t1x = -nyh; t1y = nxh; t1z = 0.0;
  }

  const double dUt1 = du * t1x + dv * t1y + dw * t1z;
  const double dUt2 = du * t2x + dv * t2y + dw * t2z;
  const double uBar_t1 = uBar * t1x + vBar * t1y + wBar * t1z;
  const double uBar_t2 = uBar * t2x + vBar * t2y + wBar * t2z;

  const double alpha1 = 0.5 * (dp - rhoBar * aBar * dqn) / aBar2;
  const double alpha5 = 0.5 * (dp + rhoBar * aBar * dqn) / aBar2;
  const double alpha2 = drho - dp / aBar2;
  const double alpha3 = rhoBar * dUt1;
  const double alpha4 = rhoBar * dUt2;

  const double lam1 = fabs(qBar - aBar);
  const double lam2 = fabs(qBar);
  const double lam3 = lam2;
  const double lam4 = lam2;
  const double lam5 = fabs(qBar + aBar);

  double D[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  D[0] += lam1 * alpha1;
  D[1] += lam1 * alpha1 * (uBar - aBar * nxh);
  D[2] += lam1 * alpha1 * (vBar - aBar * nyh);
  D[3] += lam1 * alpha1 * (wBar - aBar * nzh);
  D[4] += lam1 * alpha1 * (HBar - aBar * qBar);

  D[0] += lam2 * alpha2;
  D[1] += lam2 * alpha2 * uBar;
  D[2] += lam2 * alpha2 * vBar;
  D[3] += lam2 * alpha2 * wBar;
  D[4] += lam2 * alpha2 * 0.5 * q2Bar;

  D[1] += lam3 * alpha3 * t1x;
  D[2] += lam3 * alpha3 * t1y;
  D[3] += lam3 * alpha3 * t1z;
  D[4] += lam3 * alpha3 * uBar_t1;

  if (nDim == 3) {
    D[1] += lam4 * alpha4 * t2x;
    D[2] += lam4 * alpha4 * t2y;
    D[3] += lam4 * alpha4 * t2z;
    D[4] += lam4 * alpha4 * uBar_t2;
  }

  D[0] += lam5 * alpha5;
  D[1] += lam5 * alpha5 * (uBar + aBar * nxh);
  D[2] += lam5 * alpha5 * (vBar + aBar * nyh);
  D[3] += lam5 * alpha5 * (wBar + aBar * nzh);
  D[4] += lam5 * alpha5 * (HBar + aBar * qBar);

  flux[0] = 0.5 * (FL[0] + FR[0]) - 0.5 * nmag * D[0];
  flux[1] = 0.5 * (FL[1] + FR[1]) - 0.5 * nmag * D[1];
  flux[2] = 0.5 * (FL[2] + FR[2]) - 0.5 * nmag * D[2];
  flux[3] = 0.5 * (FL[3] + FR[3]) - 0.5 * nmag * D[3];
  flux[nVar - 1] = 0.5 * (FL[nVar - 1] + FR[nVar - 1]) - 0.5 * nmag * D[4];
}

__device__ __forceinline__ void compute_viscous_flux(const double* UL, const double* UR,
                                                     const double* gradU0, const double* gradU1,
                                                     const double nx, const double ny, const double nz,
                                                     double mu, unsigned short nDim,
                                                     unsigned short nVar, double* flux) {
  if (nDim != 3 || nVar != nDim + 2) {
    for (unsigned short v = 0; v < nVar; ++v) flux[v] = 0.0;
    return;
  }

  double grad[3][3];
  for (unsigned short i = 0; i < 3; ++i) {
    for (unsigned short j = 0; j < 3; ++j) {
      grad[i][j] = 0.5 * (gradU0[i * 3 + j] + gradU1[i * 3 + j]);
    }
  }

  const double divU = grad[0][0] + grad[1][1] + grad[2][2];
  double tau[3][3];
  const double two_thirds = 2.0 / 3.0;
  for (unsigned short i = 0; i < 3; ++i) {
    for (unsigned short j = 0; j < 3; ++j) {
      tau[i][j] = mu * (grad[i][j] + grad[j][i]);
    }
    tau[i][i] -= two_thirds * mu * divU;
  }

  const double rhoL = UL[0];
  const double rhoR = UR[0];
  const double uL = UL[1] / rhoL;
  const double vL = UL[2] / rhoL;
  const double wL = UL[3] / rhoL;
  const double uR = UR[1] / rhoR;
  const double vR = UR[2] / rhoR;
  const double wR = UR[3] / rhoR;
  const double u = 0.5 * (uL + uR);
  const double v = 0.5 * (vL + vR);
  const double w = 0.5 * (wL + wR);

  flux[0] = 0.0;
  flux[1] = tau[0][0] * nx + tau[0][1] * ny + tau[0][2] * nz;
  flux[2] = tau[1][0] * nx + tau[1][1] * ny + tau[1][2] * nz;
  flux[3] = tau[2][0] * nx + tau[2][1] * ny + tau[2][2] * nz;
  const double tau_u0 = tau[0][0] * u + tau[0][1] * v + tau[0][2] * w;
  const double tau_u1 = tau[1][0] * u + tau[1][1] * v + tau[1][2] * w;
  const double tau_u2 = tau[2][0] * u + tau[2][1] * v + tau[2][2] * w;
  flux[nVar - 1] = tau_u0 * nx + tau_u1 * ny + tau_u2 * nz;
}

__global__ void RoeNSNodeUpdateKernel(const unsigned long* nodeEdgeOffsets,
                                      const unsigned long* nodeEdges,
                                      const unsigned long* edgeNode0,
                                      const unsigned long* edgeNode1,
                                      const double* edgeNormals,
                                      const double* U,
                                      const double* gradU,
                                      const double* dt,
                                      const double* volume,
                                      const double* resExtra,
                                      const double* resTrunc,
                                      const unsigned long* bndOffsets,
                                      const double* bndNormals,
                                      const unsigned short* bndTypes,
                                      const double* bndParams,
                                      double mu,
                                      double* Uout,
                                      double* residualOut,
                                      unsigned long nPointDomain,
                                      unsigned long nPoint,
                                      unsigned short nVar,
                                      unsigned short nDim,
                                      double gamma) {
  const unsigned long iPoint = blockIdx.x * blockDim.x + threadIdx.x;
  if (iPoint >= nPointDomain || iPoint >= nPoint) return;

  double res[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (auto idx = nodeEdgeOffsets[iPoint]; idx < nodeEdgeOffsets[iPoint + 1]; ++idx) {
    const auto iEdge = nodeEdges[idx];
    const unsigned long i0 = edgeNode0[iEdge];
    const unsigned long i1 = edgeNode1[iEdge];
    const double* UL = U + i0 * nVar;
    const double* UR = U + i1 * nVar;
    const double nx = edgeNormals[3 * iEdge + 0];
    const double ny = edgeNormals[3 * iEdge + 1];
    const double nz = edgeNormals[3 * iEdge + 2];

    double flux_conv[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double flux_visc[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    compute_roe_flux(UL, UR, nx, ny, nz, nDim, nVar, gamma, flux_conv);
    compute_viscous_flux(UL, UR, gradU + i0 * 9, gradU + i1 * 9, nx, ny, nz, mu, nDim, nVar, flux_visc);

    const bool first = (iPoint == i0);
    const double sign = first ? 1.0 : -1.0;
    for (unsigned short v = 0; v < nVar; ++v) {
      res[v] += sign * (flux_conv[v] - flux_visc[v]);
    }
  }

  if (bndOffsets && bndNormals && bndTypes && bndParams) {
    const double* UL = U + iPoint * nVar;
    const double rho_i = UL[0];
    const double inv_rho_i = 1.0 / rho_i;
    const double u_i = UL[1] * inv_rho_i;
    const double v_i = UL[2] * inv_rho_i;
    const double w_i = UL[3] * inv_rho_i;
    const double E_i = UL[nVar - 1] * inv_rho_i;
    const double p_i = (gamma - 1.0) * rho_i * (E_i - 0.5 * (u_i*u_i + v_i*v_i + w_i*w_i));

    for (auto idx = bndOffsets[iPoint]; idx < bndOffsets[iPoint + 1]; ++idx) {
      const double nx = bndNormals[3 * idx + 0];
      const double ny = bndNormals[3 * idx + 1];
      const double nz = bndNormals[3 * idx + 2];
      const double nmag = sqrt(nx*nx + ny*ny + nz*nz);
      if (nmag <= 0.0) continue;
      const double inv_nmag = 1.0 / nmag;
      const double nxh = nx * inv_nmag;
      const double nyh = ny * inv_nmag;
      const double nzh = nz * inv_nmag;

      const unsigned short bc = bndTypes[idx];
      const double* param = bndParams + idx * 5;
      double rho_g = rho_i;
      double u_g = u_i;
      double v_g = v_i;
      double w_g = w_i;
      double p_g = p_i;

      if (bc == static_cast<unsigned short>(GPUFlowBC::FARFIELD) ||
          bc == static_cast<unsigned short>(GPUFlowBC::INLET)) {
        rho_g = param[0];
        u_g = param[1];
        v_g = param[2];
        w_g = param[3];
        p_g = param[4];
      } else if (bc == static_cast<unsigned short>(GPUFlowBC::OUTLET)) {
        p_g = param[4];
      } else if (bc == static_cast<unsigned short>(GPUFlowBC::SYMMETRY) ||
                 bc == static_cast<unsigned short>(GPUFlowBC::EULER_WALL)) {
        const double un = u_i * nxh + v_i * nyh + w_i * nzh;
        u_g = u_i - 2.0 * un * nxh;
        v_g = v_i - 2.0 * un * nyh;
        w_g = w_i - 2.0 * un * nzh;
      }

      double UR[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
      const double vel2_g = u_g*u_g + v_g*v_g + w_g*w_g;
      const double rhoE_g = p_g / (gamma - 1.0) + 0.5 * rho_g * vel2_g;
      UR[0] = rho_g;
      UR[1] = rho_g * u_g;
      UR[2] = rho_g * v_g;
      UR[3] = rho_g * w_g;
      UR[nVar - 1] = rhoE_g;

      double flux_conv[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
      compute_roe_flux(UL, UR, nx, ny, nz, nDim, nVar, gamma, flux_conv);
      for (unsigned short v = 0; v < nVar; ++v) res[v] += flux_conv[v];
    }
  }

  if (resExtra) {
    const double* extra = resExtra + iPoint * nVar;
    for (unsigned short v = 0; v < nVar; ++v) res[v] += extra[v];
  }
  if (resTrunc) {
    const double* trunc = resTrunc + iPoint * nVar;
    for (unsigned short v = 0; v < nVar; ++v) res[v] += trunc[v];
  }

  const double vol = volume ? volume[iPoint] : 0.0;
  const double delta = (vol > 0.0 && dt) ? dt[iPoint] / vol : 0.0;
  const double* Ui = U + iPoint * nVar;
  double* Uo = Uout + iPoint * nVar;
  for (unsigned short v = 0; v < nVar; ++v) {
    Uo[v] = Ui[v] - res[v] * delta;
  }

  if (residualOut) {
    double* out = residualOut + iPoint * nVar;
    for (unsigned short v = 0; v < nVar; ++v) out[v] = res[v];
  }
}

__global__ void RoeNodeUpdateKernel(const unsigned long* nodeEdgeOffsets,
                                    const unsigned long* nodeEdges,
                                    const unsigned long* edgeNode0,
                                    const unsigned long* edgeNode1,
                                    const double* edgeNormals,
                                    const double* U,
                                    const double* dt,
                                    const double* volume,
                                    const double* resExtra,
                                    const double* resTrunc,
                                    double* Uout,
                                    double* residualOut,
                                    unsigned long nPointDomain,
                                    unsigned long nPoint,
                                    unsigned short nVar,
                                    unsigned short nDim,
                                    double gamma) {
  const unsigned long iPoint = blockIdx.x * blockDim.x + threadIdx.x;
  if (iPoint >= nPointDomain || iPoint >= nPoint) return;

  double res[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (auto idx = nodeEdgeOffsets[iPoint]; idx < nodeEdgeOffsets[iPoint + 1]; ++idx) {
    const auto iEdge = nodeEdges[idx];
    const unsigned long i0 = edgeNode0[iEdge];
    const unsigned long i1 = edgeNode1[iEdge];
    const double* UL = U + i0 * nVar;
    const double* UR = U + i1 * nVar;
    const double nx = edgeNormals[3 * iEdge + 0];
    const double ny = edgeNormals[3 * iEdge + 1];
    const double nz = edgeNormals[3 * iEdge + 2];

    const double nmag = sqrt(nx*nx + ny*ny + nz*nz);
    if (nmag <= 0.0) continue;
    const double inv_nmag = 1.0 / nmag;
    const double nxh = nx * inv_nmag;
    const double nyh = ny * inv_nmag;
    const double nzh = nz * inv_nmag;

    double FL[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double FR[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    compute_flux_euler(UL, gamma, nx, ny, nz, nDim, nVar, FL);
    compute_flux_euler(UR, gamma, nx, ny, nz, nDim, nVar, FR);

    const double rhoL = UL[0];
    const double rhoR = UR[0];
    const double inv_rhoL = 1.0 / rhoL;
    const double inv_rhoR = 1.0 / rhoR;
    const double uL = UL[1] * inv_rhoL;
    const double vL = UL[2] * inv_rhoL;
    const double wL = (nDim == 3) ? UL[3] * inv_rhoL : 0.0;
    const double uR = UR[1] * inv_rhoR;
    const double vR = UR[2] * inv_rhoR;
    const double wR = (nDim == 3) ? UR[3] * inv_rhoR : 0.0;
    const double EL = UL[nVar - 1] * inv_rhoL;
    const double ER = UR[nVar - 1] * inv_rhoR;
    const double pL = (gamma - 1.0) * rhoL * (EL - 0.5 * (uL*uL + vL*vL + wL*wL));
    const double pR = (gamma - 1.0) * rhoR * (ER - 0.5 * (uR*uR + vR*vR + wR*wR));
    const double HL = (UL[nVar - 1] + pL) * inv_rhoL;
    const double HR = (UR[nVar - 1] + pR) * inv_rhoR;

    const double sqrt_rhoL = sqrt(rhoL);
    const double sqrt_rhoR = sqrt(rhoR);
    const double denom = sqrt_rhoL + sqrt_rhoR;
    const double inv_denom = (denom > 0.0) ? 1.0 / denom : 0.0;

    const double uBar = (sqrt_rhoL * uL + sqrt_rhoR * uR) * inv_denom;
    const double vBar = (sqrt_rhoL * vL + sqrt_rhoR * vR) * inv_denom;
    const double wBar = (sqrt_rhoL * wL + sqrt_rhoR * wR) * inv_denom;
    const double HBar = (sqrt_rhoL * HL + sqrt_rhoR * HR) * inv_denom;
    const double q2Bar = uBar*uBar + vBar*vBar + wBar*wBar;
    double aBar2 = (gamma - 1.0) * (HBar - 0.5 * q2Bar);
    if (aBar2 < 1e-14) aBar2 = 1e-14;
    const double aBar = sqrt(aBar2);
    const double rhoBar = sqrt_rhoL * sqrt_rhoR;
    const double qBar = uBar * nxh + vBar * nyh + wBar * nzh;

    const double drho = rhoR - rhoL;
    const double du = uR - uL;
    const double dv = vR - vL;
    const double dw = wR - wL;
    const double dqn = du * nxh + dv * nyh + dw * nzh;
    const double dp = pR - pL;

    double t1x = 0.0, t1y = 0.0, t1z = 0.0;
    double t2x = 0.0, t2y = 0.0, t2z = 0.0;
    if (nDim == 3) {
      build_tangent(nxh, nyh, nzh, t1x, t1y, t1z, t2x, t2y, t2z);
    } else {
      t1x = -nyh; t1y = nxh; t1z = 0.0;
    }

    const double dUt1 = du * t1x + dv * t1y + dw * t1z;
    const double dUt2 = du * t2x + dv * t2y + dw * t2z;
    const double uBar_t1 = uBar * t1x + vBar * t1y + wBar * t1z;
    const double uBar_t2 = uBar * t2x + vBar * t2y + wBar * t2z;

    const double alpha1 = 0.5 * (dp - rhoBar * aBar * dqn) / aBar2;
    const double alpha5 = 0.5 * (dp + rhoBar * aBar * dqn) / aBar2;
    const double alpha2 = drho - dp / aBar2;
    const double alpha3 = rhoBar * dUt1;
    const double alpha4 = rhoBar * dUt2;

    const double lam1 = fabs(qBar - aBar);
    const double lam2 = fabs(qBar);
    const double lam3 = lam2;
    const double lam4 = lam2;
    const double lam5 = fabs(qBar + aBar);

    double D[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    D[0] += lam1 * alpha1;
    D[1] += lam1 * alpha1 * (uBar - aBar * nxh);
    D[2] += lam1 * alpha1 * (vBar - aBar * nyh);
    D[3] += lam1 * alpha1 * (wBar - aBar * nzh);
    D[4] += lam1 * alpha1 * (HBar - aBar * qBar);

    D[0] += lam2 * alpha2;
    D[1] += lam2 * alpha2 * uBar;
    D[2] += lam2 * alpha2 * vBar;
    D[3] += lam2 * alpha2 * wBar;
    D[4] += lam2 * alpha2 * 0.5 * q2Bar;

    D[1] += lam3 * alpha3 * t1x;
    D[2] += lam3 * alpha3 * t1y;
    D[3] += lam3 * alpha3 * t1z;
    D[4] += lam3 * alpha3 * uBar_t1;

    if (nDim == 3) {
      D[1] += lam4 * alpha4 * t2x;
      D[2] += lam4 * alpha4 * t2y;
      D[3] += lam4 * alpha4 * t2z;
      D[4] += lam4 * alpha4 * uBar_t2;
    }

    D[0] += lam5 * alpha5;
    D[1] += lam5 * alpha5 * (uBar + aBar * nxh);
    D[2] += lam5 * alpha5 * (vBar + aBar * nyh);
    D[3] += lam5 * alpha5 * (wBar + aBar * nzh);
    D[4] += lam5 * alpha5 * (HBar + aBar * qBar);

    const double flux0 = 0.5 * (FL[0] + FR[0]) - 0.5 * nmag * D[0];
    const double flux1 = 0.5 * (FL[1] + FR[1]) - 0.5 * nmag * D[1];
    const double flux2 = 0.5 * (FL[2] + FR[2]) - 0.5 * nmag * D[2];
    const double fluxW = 0.5 * (FL[3] + FR[3]) - 0.5 * nmag * D[3];
    const double fluxE = 0.5 * (FL[nVar - 1] + FR[nVar - 1]) - 0.5 * nmag * D[4];

    const bool first = (iPoint == i0);
    const double sign = first ? 1.0 : -1.0;
    res[0] += sign * flux0;
    res[1] += sign * flux1;
    res[2] += sign * flux2;
    if (nDim == 3) {
      res[3] += sign * fluxW;
    }
    res[nVar - 1] += sign * fluxE;
  }

  if (resExtra) {
    const double* extra = resExtra + iPoint * nVar;
    for (unsigned short v = 0; v < nVar; ++v) res[v] += extra[v];
  }
  if (resTrunc) {
    const double* trunc = resTrunc + iPoint * nVar;
    for (unsigned short v = 0; v < nVar; ++v) res[v] += trunc[v];
  }

  const double vol = volume ? volume[iPoint] : 0.0;
  const double delta = (vol > 0.0 && dt) ? dt[iPoint] / vol : 0.0;
  const double* Ui = U + iPoint * nVar;
  double* Uo = Uout + iPoint * nVar;
  for (unsigned short v = 0; v < nVar; ++v) {
    Uo[v] = Ui[v] - res[v] * delta;
  }

  if (residualOut) {
    double* out = residualOut + iPoint * nVar;
    for (unsigned short v = 0; v < nVar; ++v) out[v] = res[v];
  }
}

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
                                                 cudaStream_t stream) {
  if (!h_nodeEdgeOffsets || !h_nodeEdges || !h_edgeNode0 || !h_edgeNode1 ||
      !h_edgeNormals || !h_solution || !h_dt || !h_volume || !h_solution_out) {
    return cudaErrorInvalidValue;
  }
  if (nDim < 2 || nDim > 3 || nVar != nDim + 2) {
    return cudaErrorNotSupported;
  }

  unsigned long* d_nodeEdgeOffsets = nullptr;
  unsigned long* d_nodeEdges = nullptr;
  unsigned long* d_edgeNode0 = nullptr;
  unsigned long* d_edgeNode1 = nullptr;
  double* d_edgeNormals = nullptr;
  double* d_solution = nullptr;
  double* d_solution_out = nullptr;
  double* d_dt = nullptr;
  double* d_volume = nullptr;
  double* d_res_extra = nullptr;
  double* d_res_trunc = nullptr;
  double* d_residual_out = nullptr;
  cudaError_t err = cudaSuccess;

  const size_t offsetsBytes = (nPoint + 1) * sizeof(unsigned long);
  const unsigned long edgeListSize = h_nodeEdgeOffsets[nPoint];
  const size_t nodeEdgesBytes = static_cast<size_t>(edgeListSize) * sizeof(unsigned long);
  const size_t edgesBytes = nEdge * sizeof(unsigned long);
  const size_t normalsBytes = nEdge * 3 * sizeof(double);
  const size_t solBytes = static_cast<size_t>(nPoint) * nVar * sizeof(double);
  const size_t dtBytes = static_cast<size_t>(nPoint) * sizeof(double);
  const size_t volBytes = static_cast<size_t>(nPoint) * sizeof(double);
  const size_t resBytes = static_cast<size_t>(nPoint) * nVar * sizeof(double);

  err = cudaMalloc(&d_nodeEdgeOffsets, offsetsBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_nodeEdges, nodeEdgesBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_edgeNode0, edgesBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_edgeNode1, edgesBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_edgeNormals, normalsBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_solution, solBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_solution_out, solBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_dt, dtBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_volume, volBytes); if (err) goto cleanup;
  if (h_residual_extra) { err = cudaMalloc(&d_res_extra, resBytes); if (err) goto cleanup; }
  if (h_residual_trunc) { err = cudaMalloc(&d_res_trunc, resBytes); if (err) goto cleanup; }
  if (h_residual_out) { err = cudaMalloc(&d_residual_out, resBytes); if (err) goto cleanup; }

  err = cudaMemcpyAsync(d_nodeEdgeOffsets, h_nodeEdgeOffsets, offsetsBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_nodeEdges, h_nodeEdges, nodeEdgesBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_edgeNode0, h_edgeNode0, edgesBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_edgeNode1, h_edgeNode1, edgesBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_edgeNormals, h_edgeNormals, normalsBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_solution, h_solution, solBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_dt, h_dt, dtBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_volume, h_volume, volBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  if (h_residual_extra) {
    err = cudaMemcpyAsync(d_res_extra, h_residual_extra, resBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  }
  if (h_residual_trunc) {
    err = cudaMemcpyAsync(d_res_trunc, h_residual_trunc, resBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  }

  {
    const int block = 256;
    const int grid = static_cast<int>((nPointDomain + block - 1) / block);
    RoeNodeUpdateKernel<<<grid, block, 0, stream>>>(d_nodeEdgeOffsets, d_nodeEdges,
                                                    d_edgeNode0, d_edgeNode1, d_edgeNormals,
                                                    d_solution, d_dt, d_volume,
                                                    d_res_extra, d_res_trunc,
                                                    d_solution_out, d_residual_out,
                                                    nPointDomain, nPoint, nVar, nDim, gamma);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
  }

  err = cudaMemcpyAsync(h_solution_out, d_solution_out, solBytes, cudaMemcpyDeviceToHost, stream); if (err) goto cleanup;
  if (h_residual_out) {
    err = cudaMemcpyAsync(h_residual_out, d_residual_out, resBytes, cudaMemcpyDeviceToHost, stream); if (err) goto cleanup;
  }
  err = cudaStreamSynchronize(stream);

cleanup:
  if (d_nodeEdgeOffsets) cudaFree(d_nodeEdgeOffsets);
  if (d_nodeEdges) cudaFree(d_nodeEdges);
  if (d_edgeNode0) cudaFree(d_edgeNode0);
  if (d_edgeNode1) cudaFree(d_edgeNode1);
  if (d_edgeNormals) cudaFree(d_edgeNormals);
  if (d_solution) cudaFree(d_solution);
  if (d_solution_out) cudaFree(d_solution_out);
  if (d_dt) cudaFree(d_dt);
  if (d_volume) cudaFree(d_volume);
  if (d_res_extra) cudaFree(d_res_extra);
  if (d_res_trunc) cudaFree(d_res_trunc);
  if (d_residual_out) cudaFree(d_residual_out);
  return err;
}

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
                                              cudaStream_t stream) {
  if (!h_nodeEdgeOffsets || !h_nodeEdges || !h_edgeNode0 || !h_edgeNode1 ||
      !h_edgeNormals || !h_solution || !h_gradU || !h_dt || !h_volume || !h_solution_out) {
    return cudaErrorInvalidValue;
  }
  if (nDim != 3 || nVar != nDim + 2) {
    return cudaErrorNotSupported;
  }

  unsigned long* d_nodeEdgeOffsets = nullptr;
  unsigned long* d_nodeEdges = nullptr;
  unsigned long* d_edgeNode0 = nullptr;
  unsigned long* d_edgeNode1 = nullptr;
  unsigned long* d_bndOffsets = nullptr;
  double* d_edgeNormals = nullptr;
  double* d_solution = nullptr;
  double* d_solution_out = nullptr;
  double* d_gradU = nullptr;
  double* d_dt = nullptr;
  double* d_volume = nullptr;
  double* d_res_extra = nullptr;
  double* d_res_trunc = nullptr;
  double* d_residual_out = nullptr;
  double* d_bndNormals = nullptr;
  unsigned short* d_bndTypes = nullptr;
  double* d_bndParams = nullptr;
  cudaError_t err = cudaSuccess;

  const size_t offsetsBytes = (nPoint + 1) * sizeof(unsigned long);
  const unsigned long edgeListSize = h_nodeEdgeOffsets[nPoint];
  const size_t nodeEdgesBytes = static_cast<size_t>(edgeListSize) * sizeof(unsigned long);
  const size_t edgesBytes = nEdge * sizeof(unsigned long);
  const size_t normalsBytes = nEdge * 3 * sizeof(double);
  const size_t solBytes = static_cast<size_t>(nPoint) * nVar * sizeof(double);
  const size_t gradBytes = static_cast<size_t>(nPoint) * 9 * sizeof(double);
  const size_t dtBytes = static_cast<size_t>(nPoint) * sizeof(double);
  const size_t volBytes = static_cast<size_t>(nPoint) * sizeof(double);
  const size_t resBytes = static_cast<size_t>(nPoint) * nVar * sizeof(double);
  const size_t bndOffsetsBytes = (nPoint + 1) * sizeof(unsigned long);
  const size_t bndNormalsBytes = static_cast<size_t>(nBnd) * 3 * sizeof(double);
  const size_t bndTypesBytes = static_cast<size_t>(nBnd) * sizeof(unsigned short);
  const size_t bndParamsBytes = static_cast<size_t>(nBnd) * 5 * sizeof(double);

  err = cudaMalloc(&d_nodeEdgeOffsets, offsetsBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_nodeEdges, nodeEdgesBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_edgeNode0, edgesBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_edgeNode1, edgesBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_edgeNormals, normalsBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_solution, solBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_solution_out, solBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_gradU, gradBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_dt, dtBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_volume, volBytes); if (err) goto cleanup;
  if (h_residual_extra) { err = cudaMalloc(&d_res_extra, resBytes); if (err) goto cleanup; }
  if (h_residual_trunc) { err = cudaMalloc(&d_res_trunc, resBytes); if (err) goto cleanup; }
  if (h_residual_out) { err = cudaMalloc(&d_residual_out, resBytes); if (err) goto cleanup; }
  if (h_bndOffsets) { err = cudaMalloc(&d_bndOffsets, bndOffsetsBytes); if (err) goto cleanup; }
  if (nBnd > 0) {
    if (h_bndNormals) { err = cudaMalloc(&d_bndNormals, bndNormalsBytes); if (err) goto cleanup; }
    if (h_bndTypes) { err = cudaMalloc(&d_bndTypes, bndTypesBytes); if (err) goto cleanup; }
    if (h_bndParams) { err = cudaMalloc(&d_bndParams, bndParamsBytes); if (err) goto cleanup; }
  }

  err = cudaMemcpyAsync(d_nodeEdgeOffsets, h_nodeEdgeOffsets, offsetsBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_nodeEdges, h_nodeEdges, nodeEdgesBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_edgeNode0, h_edgeNode0, edgesBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_edgeNode1, h_edgeNode1, edgesBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_edgeNormals, h_edgeNormals, normalsBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_solution, h_solution, solBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_gradU, h_gradU, gradBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_dt, h_dt, dtBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_volume, h_volume, volBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  if (h_residual_extra) {
    err = cudaMemcpyAsync(d_res_extra, h_residual_extra, resBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  }
  if (h_residual_trunc) {
    err = cudaMemcpyAsync(d_res_trunc, h_residual_trunc, resBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  }
  if (h_bndOffsets) {
    err = cudaMemcpyAsync(d_bndOffsets, h_bndOffsets, bndOffsetsBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  }
  if (nBnd > 0) {
    if (h_bndNormals) { err = cudaMemcpyAsync(d_bndNormals, h_bndNormals, bndNormalsBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup; }
    if (h_bndTypes) { err = cudaMemcpyAsync(d_bndTypes, h_bndTypes, bndTypesBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup; }
    if (h_bndParams) { err = cudaMemcpyAsync(d_bndParams, h_bndParams, bndParamsBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup; }
  }

  {
    const int block = 256;
    const int grid = static_cast<int>((nPointDomain + block - 1) / block);
    RoeNSNodeUpdateKernel<<<grid, block, 0, stream>>>(d_nodeEdgeOffsets, d_nodeEdges,
                                                      d_edgeNode0, d_edgeNode1, d_edgeNormals,
                                                      d_solution, d_gradU, d_dt, d_volume,
                                                      d_res_extra, d_res_trunc,
                                                      d_bndOffsets, d_bndNormals, d_bndTypes, d_bndParams,
                                                      mu, d_solution_out, d_residual_out,
                                                      nPointDomain, nPoint, nVar, nDim, gamma);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
  }

  err = cudaMemcpyAsync(h_solution_out, d_solution_out, solBytes, cudaMemcpyDeviceToHost, stream); if (err) goto cleanup;
  if (h_residual_out) {
    err = cudaMemcpyAsync(h_residual_out, d_residual_out, resBytes, cudaMemcpyDeviceToHost, stream); if (err) goto cleanup;
  }
  err = cudaStreamSynchronize(stream);

cleanup:
  if (d_nodeEdgeOffsets) cudaFree(d_nodeEdgeOffsets);
  if (d_nodeEdges) cudaFree(d_nodeEdges);
  if (d_edgeNode0) cudaFree(d_edgeNode0);
  if (d_edgeNode1) cudaFree(d_edgeNode1);
  if (d_bndOffsets) cudaFree(d_bndOffsets);
  if (d_edgeNormals) cudaFree(d_edgeNormals);
  if (d_solution) cudaFree(d_solution);
  if (d_solution_out) cudaFree(d_solution_out);
  if (d_gradU) cudaFree(d_gradU);
  if (d_dt) cudaFree(d_dt);
  if (d_volume) cudaFree(d_volume);
  if (d_res_extra) cudaFree(d_res_extra);
  if (d_res_trunc) cudaFree(d_res_trunc);
  if (d_residual_out) cudaFree(d_residual_out);
  if (d_bndNormals) cudaFree(d_bndNormals);
  if (d_bndTypes) cudaFree(d_bndTypes);
  if (d_bndParams) cudaFree(d_bndParams);
  return err;
}

extern "C" cudaError_t ComputeNSRoeUpdateLoopHost(const unsigned long* h_nodeEdgeOffsets,
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
                                                  unsigned long nIter,
                                                  double* h_solution_out,
                                                  double* h_residual_out,
                                                  cudaStream_t stream) {
  if (!h_nodeEdgeOffsets || !h_nodeEdges || !h_edgeNode0 || !h_edgeNode1 ||
      !h_edgeNormals || !h_solution || !h_gradU || !h_dt || !h_volume || !h_solution_out) {
    return cudaErrorInvalidValue;
  }
  if (nDim != 3 || nVar != nDim + 2 || nIter == 0) {
    return cudaErrorNotSupported;
  }

  unsigned long* d_nodeEdgeOffsets = nullptr;
  unsigned long* d_nodeEdges = nullptr;
  unsigned long* d_edgeNode0 = nullptr;
  unsigned long* d_edgeNode1 = nullptr;
  unsigned long* d_bndOffsets = nullptr;
  double* d_edgeNormals = nullptr;
  double* d_solution = nullptr;
  double* d_solution_out = nullptr;
  double* d_gradU = nullptr;
  double* d_dt = nullptr;
  double* d_volume = nullptr;
  double* d_res_extra = nullptr;
  double* d_res_trunc = nullptr;
  double* d_residual_out = nullptr;
  double* d_bndNormals = nullptr;
  unsigned short* d_bndTypes = nullptr;
  double* d_bndParams = nullptr;
  cudaError_t err = cudaSuccess;

  const size_t offsetsBytes = (nPoint + 1) * sizeof(unsigned long);
  const unsigned long edgeListSize = h_nodeEdgeOffsets[nPoint];
  const size_t nodeEdgesBytes = static_cast<size_t>(edgeListSize) * sizeof(unsigned long);
  const size_t edgesBytes = nEdge * sizeof(unsigned long);
  const size_t normalsBytes = nEdge * 3 * sizeof(double);
  const size_t solBytes = static_cast<size_t>(nPoint) * nVar * sizeof(double);
  const size_t gradBytes = static_cast<size_t>(nPoint) * 9 * sizeof(double);
  const size_t dtBytes = static_cast<size_t>(nPoint) * sizeof(double);
  const size_t volBytes = static_cast<size_t>(nPoint) * sizeof(double);
  const size_t resBytes = static_cast<size_t>(nPoint) * nVar * sizeof(double);
  const size_t bndOffsetsBytes = (nPoint + 1) * sizeof(unsigned long);
  const size_t bndNormalsBytes = static_cast<size_t>(nBnd) * 3 * sizeof(double);
  const size_t bndTypesBytes = static_cast<size_t>(nBnd) * sizeof(unsigned short);
  const size_t bndParamsBytes = static_cast<size_t>(nBnd) * 5 * sizeof(double);

  err = cudaMalloc(&d_nodeEdgeOffsets, offsetsBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_nodeEdges, nodeEdgesBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_edgeNode0, edgesBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_edgeNode1, edgesBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_edgeNormals, normalsBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_solution, solBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_solution_out, solBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_gradU, gradBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_dt, dtBytes); if (err) goto cleanup;
  err = cudaMalloc(&d_volume, volBytes); if (err) goto cleanup;
  if (h_residual_extra) { err = cudaMalloc(&d_res_extra, resBytes); if (err) goto cleanup; }
  if (h_residual_trunc) { err = cudaMalloc(&d_res_trunc, resBytes); if (err) goto cleanup; }
  if (h_residual_out) { err = cudaMalloc(&d_residual_out, resBytes); if (err) goto cleanup; }
  if (h_bndOffsets) { err = cudaMalloc(&d_bndOffsets, bndOffsetsBytes); if (err) goto cleanup; }
  if (nBnd > 0) {
    if (h_bndNormals) { err = cudaMalloc(&d_bndNormals, bndNormalsBytes); if (err) goto cleanup; }
    if (h_bndTypes) { err = cudaMalloc(&d_bndTypes, bndTypesBytes); if (err) goto cleanup; }
    if (h_bndParams) { err = cudaMalloc(&d_bndParams, bndParamsBytes); if (err) goto cleanup; }
  }

  err = cudaMemcpyAsync(d_nodeEdgeOffsets, h_nodeEdgeOffsets, offsetsBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_nodeEdges, h_nodeEdges, nodeEdgesBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_edgeNode0, h_edgeNode0, edgesBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_edgeNode1, h_edgeNode1, edgesBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_edgeNormals, h_edgeNormals, normalsBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_solution, h_solution, solBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_gradU, h_gradU, gradBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_dt, h_dt, dtBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  err = cudaMemcpyAsync(d_volume, h_volume, volBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  if (h_residual_extra) {
    err = cudaMemcpyAsync(d_res_extra, h_residual_extra, resBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  }
  if (h_residual_trunc) {
    err = cudaMemcpyAsync(d_res_trunc, h_residual_trunc, resBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  }
  if (h_bndOffsets) {
    err = cudaMemcpyAsync(d_bndOffsets, h_bndOffsets, bndOffsetsBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup;
  }
  if (nBnd > 0) {
    if (h_bndNormals) { err = cudaMemcpyAsync(d_bndNormals, h_bndNormals, bndNormalsBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup; }
    if (h_bndTypes) { err = cudaMemcpyAsync(d_bndTypes, h_bndTypes, bndTypesBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup; }
    if (h_bndParams) { err = cudaMemcpyAsync(d_bndParams, h_bndParams, bndParamsBytes, cudaMemcpyHostToDevice, stream); if (err) goto cleanup; }
  }

  {
    const int block = 256;
    const int grid = static_cast<int>((nPointDomain + block - 1) / block);
    double* d_in = d_solution;
    double* d_out = d_solution_out;
    double* d_final = d_solution;
    for (unsigned long iter = 0; iter < nIter; ++iter) {
      const bool last = (iter + 1 == nIter);
      RoeNSNodeUpdateKernel<<<grid, block, 0, stream>>>(d_nodeEdgeOffsets, d_nodeEdges,
                                                        d_edgeNode0, d_edgeNode1, d_edgeNormals,
                                                        d_in, d_gradU, d_dt, d_volume,
                                                        d_res_extra, d_res_trunc,
                                                        d_bndOffsets, d_bndNormals, d_bndTypes, d_bndParams,
                                                        mu, d_out, last ? d_residual_out : nullptr,
                                                        nPointDomain, nPoint, nVar, nDim, gamma);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto cleanup;
      d_final = d_out;
      double* tmp = d_in;
      d_in = d_out;
      d_out = tmp;
    }
    d_solution_out = d_final;
  }

  err = cudaMemcpyAsync(h_solution_out, d_solution_out, solBytes, cudaMemcpyDeviceToHost, stream); if (err) goto cleanup;
  if (h_residual_out) {
    err = cudaMemcpyAsync(h_residual_out, d_residual_out, resBytes, cudaMemcpyDeviceToHost, stream); if (err) goto cleanup;
  }
  err = cudaStreamSynchronize(stream);

cleanup:
  if (d_nodeEdgeOffsets) cudaFree(d_nodeEdgeOffsets);
  if (d_nodeEdges) cudaFree(d_nodeEdges);
  if (d_edgeNode0) cudaFree(d_edgeNode0);
  if (d_edgeNode1) cudaFree(d_edgeNode1);
  if (d_bndOffsets) cudaFree(d_bndOffsets);
  if (d_edgeNormals) cudaFree(d_edgeNormals);
  if (d_solution) cudaFree(d_solution);
  if (d_solution_out) cudaFree(d_solution_out);
  if (d_gradU) cudaFree(d_gradU);
  if (d_dt) cudaFree(d_dt);
  if (d_volume) cudaFree(d_volume);
  if (d_res_extra) cudaFree(d_res_extra);
  if (d_res_trunc) cudaFree(d_res_trunc);
  if (d_residual_out) cudaFree(d_residual_out);
  if (d_bndNormals) cudaFree(d_bndNormals);
  if (d_bndTypes) cudaFree(d_bndTypes);
  if (d_bndParams) cudaFree(d_bndParams);
  return err;
}

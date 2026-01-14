// Minimal CUDA example: setZero for AeroCoeffsArray on device buffers.
// This is a stub to illustrate how a GPU kernel could zero the arrays.
// Real integration requires wiring Device pointers and build system updates.

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

struct CEulerSolverGPUState;

extern "C" __global__ void SetZeroKernel(double* CD, double* CL, double* CSF, double* CEff,
                                          double* CFx, double* CFy, double* CFz,
                                          double* CMx, double* CMy, double* CMz,
                                          double* CoPx, double* CoPy, double* CoPz,
                                          double* CT, double* CQ, double* CMerit,
                                          int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    CD[idx] = CL[idx] = CSF[idx] = CEff[idx] = 0.0;
    CFx[idx] = CFy[idx] = CFz[idx] = CMx[idx] = 0.0;
    CMy[idx] = CMz[idx] = CoPx[idx] = CoPy[idx] = 0.0;
    CoPz[idx] = CT[idx] = CQ[idx] = CMerit[idx] = 0.0;
  }
}

// Simple C-callable launcher. Caller owns device memory and stream.
extern "C" void LaunchSetZeroKernel(double* CD, double* CL, double* CSF, double* CEff,
                                    double* CFx, double* CFy, double* CFz,
                                    double* CMx, double* CMy, double* CMz,
                                    double* CoPx, double* CoPy, double* CoPz,
                                    double* CT, double* CQ, double* CMerit,
                                    int n, cudaStream_t stream) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  SetZeroKernel<<<grid, block, 0, stream>>>(CD, CL, CSF, CEff, CFx, CFy, CFz,
                                           CMx, CMy, CMz, CoPx, CoPy, CoPz,
                                           CT, CQ, CMerit, n);
}

// Convenience helper: take host arrays, allocate temporary device buffers, zero them on GPU, copy back.
extern "C" cudaError_t SetZeroHostArrays(double* CD, double* CL, double* CSF, double* CEff,
                                         double* CFx, double* CFy, double* CFz,
                                         double* CMx, double* CMy, double* CMz,
                                         double* CoPx, double* CoPy, double* CoPz,
                                         double* CT, double* CQ, double* CMerit,
                                         int n, cudaStream_t stream) {
  constexpr int kNumArrays = 16;
  double* host[kNumArrays] = {CD, CL, CSF, CEff, CFx, CFy, CFz,
                              CMx, CMy, CMz, CoPx, CoPy, CoPz,
                              CT, CQ, CMerit};
  double* device[kNumArrays] = {};
  const size_t bytes = static_cast<size_t>(n) * sizeof(double);
  cudaError_t err = cudaSuccess;

  // Allocate and copy host -> device (if host pointer is provided)
  for (int i = 0; i < kNumArrays && err == cudaSuccess; ++i) {
    if (!host[i]) continue; // allow null to skip
    err = cudaMalloc(reinterpret_cast<void**>(&device[i]), bytes);
    if (err != cudaSuccess) break;
    err = cudaMemcpyAsync(device[i], host[i], bytes, cudaMemcpyHostToDevice, stream);
  }

  // Launch kernel if allocation/copies succeeded
  if (err == cudaSuccess) {
    LaunchSetZeroKernel(device[0], device[1], device[2], device[3],
                       device[4], device[5], device[6],
                       device[7], device[8], device[9],
                       device[10], device[11], device[12],
                       device[13], device[14], device[15], n, stream);
    err = cudaGetLastError();
  }

  // Copy results back to host (only if earlier steps succeeded)
  for (int i = 0; i < kNumArrays && err == cudaSuccess; ++i) {
    if (host[i] && device[i]) {
      err = cudaMemcpyAsync(host[i], device[i], bytes, cudaMemcpyDeviceToHost, stream);
    }
  }

  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);

  // Cleanup device buffers
  for (int i = 0; i < kNumArrays; ++i) {
    if (device[i]) cudaFree(device[i]);
  }

  return err;
}

/* Stub for future GPU residual assembly. Currently unimplemented and
 * returns cudaErrorNotSupported so the symbol is available at link time. */
extern "C" cudaError_t ComputeResidualGPU(const void* /*geometry*/,
                                          const void* /*solution*/,
                                          const void* /*numerics*/,
                                          cudaStream_t /*stream*/) {
  return cudaErrorNotSupported;
}

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

/* GPU version of SumEdgeFluxes.
 * Inputs (all device pointers):
 *  - nodeEdgeOffsets: CSR offsets, size nPoint+1
 *  - nodeEdges      : edge indices, size nEdge
 *  - edgeNode0/1    : edge endpoints, size nEdge
 *  - edgeFluxes     : edge flux blocks, size nEdge * nVar
 * Output:
 *  - linSysRes      : residual blocks, size nPoint * nVar
 */
__global__ void SumEdgeFluxesKernel(const unsigned long* nodeEdgeOffsets,
                                    const unsigned long* nodeEdges,
                                    const unsigned long* edgeNode0,
                                    const unsigned long* edgeNode1,
                                    const double* edgeFluxes,
                                    double* linSysRes,
                                    unsigned long nPoint,
                                    unsigned short nVar) {
  const unsigned long iPoint = blockIdx.x * blockDim.x + threadIdx.x;
  if (iPoint >= nPoint) return;

  double* res = linSysRes + iPoint * nVar;
  for (unsigned short v = 0; v < nVar; ++v) res[v] = 0.0;

  for (auto idx = nodeEdgeOffsets[iPoint]; idx < nodeEdgeOffsets[iPoint + 1]; ++idx) {
    const auto iEdge = nodeEdges[idx];
    const double* flux = edgeFluxes + iEdge * nVar;
    const bool first = (edgeNode0[iEdge] == iPoint);
    for (unsigned short v = 0; v < nVar; ++v) {
      res[v] += first ? flux[v] : -flux[v];
    }
  }
}

extern "C" cudaError_t SumEdgeFluxesGPU(const unsigned long* nodeEdgeOffsets,
                                        const unsigned long* nodeEdges,
                                        const unsigned long* edgeNode0,
                                        const unsigned long* edgeNode1,
                                        const double* edgeFluxes,
                                        double* linSysRes,
                                        unsigned long nPoint,
                                        unsigned short nVar,
                                        cudaStream_t stream) {
  if (!nodeEdgeOffsets || !nodeEdges || !edgeNode0 || !edgeNode1 || !edgeFluxes || !linSysRes) {
    return cudaErrorInvalidValue;
  }
  const int block = 256;
  const int grid = static_cast<int>((nPoint + block - 1) / block);
  SumEdgeFluxesKernel<<<grid, block, 0, stream>>>(nodeEdgeOffsets, nodeEdges, edgeNode0, edgeNode1,
                                                  edgeFluxes, linSysRes, nPoint, nVar);
  return cudaGetLastError();
}

/* Stub GPU state management for CEulerSolver (no-op until full GPU path exists). */
extern "C" CEulerSolverGPUState* CreateEulerSolverGPU(int /*nZone*/) {
  return nullptr;
}

extern "C" void DestroyEulerSolverGPU(CEulerSolverGPUState* /*state*/) {}

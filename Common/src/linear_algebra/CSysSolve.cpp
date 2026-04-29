/*!
 * \file CSysSolve.cpp
 * \brief Main classes required for solving linear systems of equations
 * \author J. Hicken, F. Palacios, T. Economon, P. Gomes
 * \version 8.3.0 "Harrier"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2025, SU2 Contributors (cf. AUTHORS.md)
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

#include "../../include/linear_algebra/CSysSolve.hpp"
#include "../../include/linear_algebra/CSysSolve_b.hpp"
#include "../../include/parallelization/omp_structure.hpp"
#include "../../include/CConfig.hpp"
#include "../../include/geometry/CGeometry.hpp"
#include "../../include/linear_algebra/CSysMatrix.hpp"
#include "../../include/linear_algebra/CMatrixVectorProduct.hpp"
#include "../../include/linear_algebra/CPreconditioner.hpp"

#include <limits>
#include <sstream>

#ifdef HAVE_AMGX
#include <amgx_c.h>
#endif

/*!
 * \brief Epsilon used in CSysSolve depending on datatype to
 * decide if the linear system is already solved.
 */
namespace {
template <class T>
constexpr T linSolEpsilon() {
  return numeric_limits<passivedouble>::epsilon();
}
template <>
constexpr float linSolEpsilon<float>() {
  return 1e-12;
}
}  // namespace

#ifdef HAVE_AMGX
namespace {
template <class T>
AMGX_Mode AMGXMode();

template <>
AMGX_Mode AMGXMode<su2double>() {
  return AMGX_mode_dDDI;
}

template <>
AMGX_Mode AMGXMode<float>() {
  return AMGX_mode_dFFI;
}

void SilentAMGXPrintCallback(const char*, int) {}

class AMGXRuntime {
 public:
  AMGXRuntime() {
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    AMGX_SAFE_CALL(AMGX_register_print_callback(&SilentAMGXPrintCallback));
    AMGX_SAFE_CALL(AMGX_install_signal_handler());
  }

  ~AMGXRuntime() {
    AMGX_finalize_plugins();
    AMGX_finalize();
  }
};
}  // namespace
#endif

template <class ScalarType>
CSysSolve<ScalarType>::CSysSolve(LINEAR_SOLVER_MODE linear_solver_mode)
    : eps(linSolEpsilon<ScalarType>()),
      lin_sol_mode(linear_solver_mode),
      cg_ready(false),
      bcg_ready(false),
      smooth_ready(false),
      LinSysSol_ptr(nullptr),
      LinSysRes_ptr(nullptr) {}

template <class ScalarType>
CSysSolve<ScalarType>::~CSysSolve() {
  ResetAMGXCache();
}

template <class ScalarType>
void CSysSolve<ScalarType>::ResetAMGXCache() const {
#ifdef HAVE_AMGX
  auto& cache = amgx_cache;
  if (cache.solver) AMGX_SAFE_CALL(AMGX_solver_destroy(reinterpret_cast<AMGX_solver_handle>(cache.solver)));
  if (cache.sol) AMGX_SAFE_CALL(AMGX_vector_destroy(reinterpret_cast<AMGX_vector_handle>(cache.sol)));
  if (cache.rhs) AMGX_SAFE_CALL(AMGX_vector_destroy(reinterpret_cast<AMGX_vector_handle>(cache.rhs)));
  if (cache.matrix) AMGX_SAFE_CALL(AMGX_matrix_destroy(reinterpret_cast<AMGX_matrix_handle>(cache.matrix)));
  if (cache.rsrc) AMGX_SAFE_CALL(AMGX_resources_destroy(reinterpret_cast<AMGX_resources_handle>(cache.rsrc)));
  if (cache.cfg) AMGX_SAFE_CALL(AMGX_config_destroy(reinterpret_cast<AMGX_config_handle>(cache.cfg)));
#endif
  amgx_cache = AMGXCache{};
}

template <class ScalarType>
void CSysSolve<ScalarType>::ApplyGivens(ScalarType s, ScalarType c, ScalarType& h1, ScalarType& h2) const {
  ScalarType temp = c * h1 + s * h2;
  h2 = c * h2 - s * h1;
  h1 = temp;
}

template <class ScalarType>
void CSysSolve<ScalarType>::GenerateGivens(ScalarType& dx, ScalarType& dy, ScalarType& s, ScalarType& c) const {
  if ((dx == 0.0) && (dy == 0.0)) {
    c = 1.0;
    s = 0.0;
  } else if (fabs(dy) > fabs(dx)) {
    ScalarType tmp = dx / dy;
    dx = sqrt(1.0 + tmp * tmp);
    s = Sign(1.0 / dx, dy);
    c = tmp * s;
  } else if (fabs(dy) <= fabs(dx)) {
    ScalarType tmp = dy / dx;
    dy = sqrt(1.0 + tmp * tmp);
    c = Sign(1.0 / dy, dx);
    s = tmp * c;
  } else {
    // dx and/or dy must be invalid
    dx = 0.0;
    dy = 0.0;
    c = 1.0;
    s = 0.0;
  }
  dx = fabs(dx * dy);
  dy = 0.0;
}

template <class ScalarType>
void CSysSolve<ScalarType>::SolveReduced(int n, const su2matrix<ScalarType>& Hsbg, const su2vector<ScalarType>& rhs,
                                         su2vector<ScalarType>& x) const {
  // initialize...
  for (int i = 0; i < n; i++) x[i] = rhs[i];
  // ... and backsolve
  for (int i = n - 1; i >= 0; i--) {
    x[i] /= Hsbg(i, i);
    for (int j = i - 1; j >= 0; j--) {
      x[j] -= Hsbg(j, i) * x[i];
    }
  }
}

template <class ScalarType>
void CSysSolve<ScalarType>::ModGramSchmidt(bool shared_hsbg, int i, su2matrix<ScalarType>& Hsbg,
                                           vector<CSysVector<ScalarType> >& w) const {
  const auto thread = omp_get_thread_num();

  /*--- If Hsbg is shared by multiple threads calling this function, only one
   * thread can write into it. If Hsbg is private, all threads need to write. ---*/

  auto SetHsbg = [&](int row, int col, const ScalarType& value) {
    if (!shared_hsbg || thread == 0) {
      Hsbg(row, col) = value;
    }
  };

  /*--- Parameter for reorthonormalization ---*/

  const ScalarType reorth = 0.98;

  /*--- Get the norm of the vector being orthogonalized, and find the
  threshold for re-orthogonalization ---*/

  ScalarType nrm = w[i + 1].squaredNorm();
  ScalarType thr = nrm * reorth;

  /*--- The norm of w[i+1] < 0.0 or w[i+1] = NaN ---*/

  if ((nrm <= 0.0) || (nrm != nrm)) {
    /*--- nrm is the result of a dot product, communications are implicitly handled. ---*/
    SU2_MPI::Error("FGMRES orthogonalization failed, linear solver diverged.", CURRENT_FUNCTION);
  }

  /*--- Begin main Gram-Schmidt loop ---*/

  for (int k = 0; k < i + 1; k++) {
    ScalarType prod = w[i + 1].dot(w[k]);
    ScalarType h_ki = prod;
    w[i + 1] -= prod * w[k];

    /*--- Check if reorthogonalization is necessary ---*/

    if (prod * prod > thr) {
      prod = w[i + 1].dot(w[k]);
      h_ki += prod;
      w[i + 1] -= prod * w[k];
    }
    SetHsbg(k, i, h_ki);

    /*--- Update the norm and check its size ---*/

    nrm -= pow(h_ki, 2);
    nrm = max<ScalarType>(nrm, 0.0);
    thr = nrm * reorth;
  }

  /*--- Test the resulting vector ---*/

  nrm = w[i + 1].norm();
  SetHsbg(i + 1, i, nrm);

  /*--- Scale the resulting vector ---*/

  w[i + 1] /= nrm;
}

template <class ScalarType>
void CSysSolve<ScalarType>::WriteHeader(const string& solver, ScalarType restol, ScalarType resinit) const {
  cout << "\n# " << solver << " residual history\n";
  cout << "# Residual tolerance target = " << restol << "\n";
  cout << "# Initial residual norm     = " << resinit << endl;
}

template <class ScalarType>
void CSysSolve<ScalarType>::WriteHistory(unsigned long iter, ScalarType res) const {
  cout << "     " << iter << "     " << res << endl;
}

template <class ScalarType>
void CSysSolve<ScalarType>::WriteFinalResidual(const string& solver, unsigned long iter, ScalarType res) const {
  cout << "# " << solver << " final (true) residual:\n";
  cout << "# Iteration = " << iter << ": |res|/|res0| = " << res << ".\n" << endl;
}

template <class ScalarType>
void CSysSolve<ScalarType>::WriteWarning(ScalarType res_calc, ScalarType res_true, ScalarType tol) const {
  cout << "# WARNING:\n";
  cout << "# true residual norm and calculated residual norm do not agree.\n";
  cout << "# true_res = " << res_true << ", calc_res = " << res_calc << ", tol = " << tol * 10 << ".\n";
  cout << "# true_res - calc_res = " << res_true - res_calc << endl;
}

template <class ScalarType>
unsigned long CSysSolve<ScalarType>::CG_LinSolver(const CSysVector<ScalarType>& b, CSysVector<ScalarType>& x,
                                                  const CMatrixVectorProduct<ScalarType>& mat_vec,
                                                  const CPreconditioner<ScalarType>& precond, ScalarType tol,
                                                  unsigned long m, ScalarType& residual, bool monitoring,
                                                  const CConfig* config) const {
  const bool masterRank = (SU2_MPI::GetRank() == MASTER_NODE);
  ScalarType norm_r = 0.0, norm0 = 0.0;
  unsigned long i = 0;

  /*--- Check the subspace size ---*/

  if (m < 1) {
    SU2_MPI::Error("Number of linear solver iterations must be greater than 0.", CURRENT_FUNCTION);
  }

  /*--- Allocate if not allocated yet, only one thread can
   *    do this since the working vectors are shared. ---*/

  if (!cg_ready) {
    BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS {
      auto nVar = b.GetNVar();
      auto nBlk = b.GetNBlk();
      auto nBlkDomain = b.GetNBlkDomain();

      A_x.Initialize(nBlk, nBlkDomain, nVar, nullptr);
      r.Initialize(nBlk, nBlkDomain, nVar, nullptr);
      z.Initialize(nBlk, nBlkDomain, nVar, nullptr);
      p.Initialize(nBlk, nBlkDomain, nVar, nullptr);

      cg_ready = true;
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS
  }

  /*--- Calculate the initial residual, compute norm, and check if system is already solved ---*/

  if (!xIsZero) {
    mat_vec(x, A_x);
    r = b - A_x;
  } else {
    r = b;
  }

  /*--- Only compute the residuals in full communication mode. ---*/

  if (config->GetComm_Level() == COMM_FULL) {
    norm_r = r.norm();
    norm0 = b.norm();

    /*--- Set the norm to the initial initial residual value ---*/

    if (tol_type == LinearToleranceType::RELATIVE) norm0 = norm_r;

    if ((norm_r < tol * norm0) || (norm_r < eps)) {
      if (masterRank && (lin_sol_mode != LINEAR_SOLVER_MODE::MESH_DEFORM)) {
        SU2_OMP_MASTER
        cout << "CSysSolve::ConjugateGradient(): system solved by initial guess." << endl;
        END_SU2_OMP_MASTER
      }
      return 0;
    }

    /*--- Output header information including initial residual ---*/

    if (monitoring && masterRank) {
      SU2_OMP_MASTER {
        WriteHeader("CG", tol, norm_r);
        WriteHistory(i, norm_r / norm0);
      }
      END_SU2_OMP_MASTER
    }
  }

  precond(r, z);
  p = z;
  ScalarType r_dot_z = r.dot(z);

  /*---  Loop over all search directions ---*/

  for (i = 0; i < m; i++) {
    /*--- Apply matrix to p to build Krylov subspace ---*/

    mat_vec(p, A_x);

    /*--- Calculate step-length alpha ---*/

    ScalarType alpha = r_dot_z / A_x.dot(p);

    /*--- Update solution and residual: ---*/

    x += alpha * p;
    r -= alpha * A_x;

    /*--- Only compute the residuals in full communication mode. ---*/

    if (config->GetComm_Level() == COMM_FULL) {
      /*--- Check if solution has converged, else output the relative residual if necessary ---*/

      norm_r = r.norm();
      if (norm_r < tol * norm0) break;
      if (((monitoring) && (masterRank)) && ((i + 1) % monitorFreq == 0)) {
        SU2_OMP_MASTER
        WriteHistory(i + 1, norm_r / norm0);
        END_SU2_OMP_MASTER
      }
    }

    precond(r, z);

    /*--- Calculate Gram-Schmidt coefficient, beta = (r_{i+1}, z_{i+1}) / (r_{i}, z_{i}) ---*/

    ScalarType beta = r_dot_z;
    r_dot_z = r.dot(z);
    beta = r_dot_z / beta;

    /*--- Gram-Schmidt orthogonalization. ---*/

    p = beta * p + z;
  }

  /*--- Recalculate final residual (this should be optional) ---*/

  if ((monitoring) && (config->GetComm_Level() == COMM_FULL)) {
    if (masterRank) {
      SU2_OMP_MASTER
      WriteFinalResidual("CG", i, norm_r / norm0);
      END_SU2_OMP_MASTER
    }

    if (recomputeRes) {
      mat_vec(x, A_x);
      r = b - A_x;
      ScalarType true_res = r.norm();

      if (fabs(true_res - norm_r) > tol * 10.0) {
        if (masterRank) {
          SU2_OMP_MASTER
          WriteWarning(norm_r, true_res, tol);
          END_SU2_OMP_MASTER
        }
      }
    }
  }

  residual = norm_r / norm0;
  return i;
}

template <class ScalarType>
unsigned long CSysSolve<ScalarType>::FGMRES_LinSolver(const CSysVector<ScalarType>& b, CSysVector<ScalarType>& x,
                                                      const CMatrixVectorProduct<ScalarType>& mat_vec,
                                                      const CPreconditioner<ScalarType>& precond, ScalarType tol,
                                                      unsigned long m, ScalarType& residual, bool monitoring,
                                                      const CConfig* config) const {
  const bool masterRank = (SU2_MPI::GetRank() == MASTER_NODE);
  const bool flexible = !precond.IsIdentity();
  /*--- If we call the solver outside of a parallel region, but the number of threads allows,
   * we still want to parallelize some of the expensive operations. ---*/
  const bool nestedParallel = !omp_in_parallel() && omp_get_max_threads() > 1;

  /*---  Check the subspace size ---*/

  if (m < 1) {
    SU2_MPI::Error("Number of linear solver iterations must be greater than 0.", CURRENT_FUNCTION);
  }

  if (m > 5000) {
    SU2_MPI::Error("FGMRES subspace is too large.", CURRENT_FUNCTION);
  }

  /*--- Allocate if not allocated yet ---*/

  if (W.size() <= m || (flexible && Z.size() <= m)) {
    BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS {
      W.resize(m + 1);
      for (auto& w : W) w.Initialize(x.GetNBlk(), x.GetNBlkDomain(), x.GetNVar(), nullptr);
      if (flexible) {
        Z.resize(m + 1);
        for (auto& z : Z) z.Initialize(x.GetNBlk(), x.GetNBlkDomain(), x.GetNVar(), nullptr);
      }
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS
  }

  /*--- Define various arrays. In parallel, each thread of each rank has and works
   on its own thread, since calculations on these arrays are based on dot products
   (reduced across all threads and ranks) all threads do the same computations. ---*/

  su2vector<ScalarType> g(m + 1), sn(m + 1), cs(m + 1), y(m);
  g = ScalarType(0);
  sn = ScalarType(0);
  cs = ScalarType(0);
  y = ScalarType(0);
  su2matrix<ScalarType> H(m + 1, m);
  H = ScalarType(0);

  /*--- Calculate the norm of the rhs vector. ---*/

  ScalarType norm0 = b.norm();

  /*--- Calculate the initial residual (actually the negative residual) and compute its norm. ---*/

  if (!xIsZero) {
    mat_vec(x, W[0]);
    W[0] -= b;
  } else {
    W[0] = -b;
  }

  ScalarType beta = W[0].norm();

  /*--- Set the norm to the initial initial residual value ---*/

  if (tol_type == LinearToleranceType::RELATIVE) norm0 = beta;

  if ((beta < tol * norm0) || (beta < eps)) {
    /*--- System is already solved ---*/

    if (masterRank) {
      SU2_OMP_MASTER
      cout << "CSysSolve::FGMRES(): system solved by initial guess." << endl;
      END_SU2_OMP_MASTER
    }
    residual = beta;
    return 0;
  }

  /*--- Normalize residual to get w_{0} (the negative sign is because w[0]
        holds the negative residual, as mentioned above). ---*/

  W[0] /= -beta;

  /*--- Initialize the RHS of the reduced system ---*/

  g[0] = beta;

  /*--- Output header information including initial residual ---*/

  unsigned long i = 0;
  if ((monitoring) && (masterRank)) {
    SU2_OMP_MASTER {
      WriteHeader("FGMRES", tol, beta);
      WriteHistory(i, beta / norm0);
    }
    END_SU2_OMP_MASTER
  }

  /*---  Loop over all search directions ---*/

  for (i = 0; i < m; i++) {
    /*---  Check if solution has converged ---*/

    if (beta < tol * norm0) break;

    if (flexible) {
      /*---  Precondition the CSysVector w[i] and store result in z[i] ---*/

      precond(W[i], Z[i]);

      /*---  Add to Krylov subspace ---*/

      mat_vec(Z[i], W[i + 1]);
    } else {
      mat_vec(W[i], W[i + 1]);
    }

    /*---  Modified Gram-Schmidt orthogonalization ---*/

    if (nestedParallel) {
      /*--- "omp parallel if" does not work well here ---*/
      SU2_OMP_PARALLEL
      ModGramSchmidt(true, i, H, W);
      END_SU2_OMP_PARALLEL
    } else {
      ModGramSchmidt(false, i, H, W);
    }

    /*---  Apply old Givens rotations to new column of the Hessenberg matrix then generate the
     new Givens rotation matrix and apply it to the last two elements of H[:][i] and g ---*/

    for (unsigned long k = 0; k < i; k++) ApplyGivens(sn[k], cs[k], H[k][i], H[k + 1][i]);
    GenerateGivens(H[i][i], H[i + 1][i], sn[i], cs[i]);
    ApplyGivens(sn[i], cs[i], g[i], g[i + 1]);

    /*---  Set L2 norm of residual and check if solution has converged ---*/

    beta = fabs(g[i + 1]);

    /*---  Output the relative residual if necessary ---*/

    if ((((monitoring) && (masterRank)) && ((i + 1) % monitorFreq == 0))) {
      SU2_OMP_MASTER
      WriteHistory(i + 1, beta / norm0);
      END_SU2_OMP_MASTER
    }
  }

  /*---  Solve the least-squares system and update solution ---*/

  SolveReduced(i, H, g, y);

  const auto& basis = flexible ? Z : W;

  if (nestedParallel) {
    SU2_OMP_PARALLEL
    for (unsigned long k = 0; k < i; k++) x += y[k] * basis[k];
    END_SU2_OMP_PARALLEL
  } else {
    for (unsigned long k = 0; k < i; k++) x += y[k] * basis[k];
  }

  /*---  Recalculate final (neg.) residual (this should be optional) ---*/

  if ((monitoring) && (config->GetComm_Level() == COMM_FULL)) {
    if (masterRank) {
      SU2_OMP_MASTER
      WriteFinalResidual("FGMRES", i, beta / norm0);
      END_SU2_OMP_MASTER
    }

    if (recomputeRes) {
      mat_vec(x, W[0]);
      W[0] -= b;
      ScalarType res = W[0].norm();

      if (fabs(res - beta) > tol * 10) {
        if (masterRank) {
          SU2_OMP_MASTER
          WriteWarning(beta, res, tol);
          END_SU2_OMP_MASTER
        }
      }
    }
  }

  residual = beta / norm0;
  return i;
}

template <class ScalarType>
unsigned long CSysSolve<ScalarType>::RFGMRES_LinSolver(const CSysVector<ScalarType>& b, CSysVector<ScalarType>& x,
                                                       const CMatrixVectorProduct<ScalarType>& mat_vec,
                                                       const CPreconditioner<ScalarType>& precond, ScalarType tol,
                                                       unsigned long MaxIter, ScalarType& residual, bool monitoring,
                                                       const CConfig* config) {
  const auto restartIter = config->GetLinear_Solver_Restart_Frequency();

  SU2_OMP_MASTER {
    xIsZero = false;
    tol_type = LinearToleranceType::ABSOLUTE;
  }
  END_SU2_OMP_MASTER

  const ScalarType norm0 = b.norm();  // <- Has a barrier

  for (auto totalIter = 0ul; totalIter < MaxIter;) {
    /*--- Enforce a hard limit on total number of iterations ---*/
    auto iterLimit = min(restartIter, MaxIter - totalIter);
    auto iter = FGMRES_LinSolver(b, x, mat_vec, precond, tol, iterLimit, residual, monitoring, config);
    totalIter += iter;
    if (residual <= tol * norm0 || iter < iterLimit) return totalIter;
  }
  return 0;
}

template <class ScalarType>
unsigned long CSysSolve<ScalarType>::BCGSTAB_LinSolver(const CSysVector<ScalarType>& b, CSysVector<ScalarType>& x,
                                                       const CMatrixVectorProduct<ScalarType>& mat_vec,
                                                       const CPreconditioner<ScalarType>& precond, ScalarType tol,
                                                       unsigned long m, ScalarType& residual, bool monitoring,
                                                       const CConfig* config) const {
  const bool masterRank = (SU2_MPI::GetRank() == MASTER_NODE);
  ScalarType norm_r = 0.0, norm0 = 0.0;
  unsigned long i = 0;

  /*--- Check the subspace size ---*/

  if (m < 1) {
    SU2_MPI::Error("Number of linear solver iterations must be greater than 0.", CURRENT_FUNCTION);
  }

  /*--- Allocate if not allocated yet ---*/

  if (!bcg_ready) {
    BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS {
      auto nVar = b.GetNVar();
      auto nBlk = b.GetNBlk();
      auto nBlkDomain = b.GetNBlkDomain();

      A_x.Initialize(nBlk, nBlkDomain, nVar, nullptr);
      r_0.Initialize(nBlk, nBlkDomain, nVar, nullptr);
      r.Initialize(nBlk, nBlkDomain, nVar, nullptr);
      p.Initialize(nBlk, nBlkDomain, nVar, nullptr);
      v.Initialize(nBlk, nBlkDomain, nVar, nullptr);
      z.Initialize(nBlk, nBlkDomain, nVar, nullptr);

      bcg_ready = true;
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS
  }

  /*--- Calculate the initial residual, compute norm, and check if system is already solved ---*/

  if (!xIsZero) {
    mat_vec(x, A_x);
    r = b - A_x;
  } else {
    r = b;
  }

  /*--- Only compute the residuals in full communication mode. ---*/
  //mpi check
  if (config->GetComm_Level() == COMM_FULL) {
    norm_r = r.norm();
    norm0 = b.norm();

    /*--- Set the norm to the initial initial residual value ---*/
    // tolerence check
    if (tol_type == LinearToleranceType::RELATIVE) norm0 = norm_r;

    if ((norm_r < tol * norm0) || (norm_r < eps)) {
      if (masterRank) {
        SU2_OMP_MASTER
        cout << "CSysSolve::BCGSTAB(): system solved by initial guess." << endl;
        END_SU2_OMP_MASTER
      }
      return 0;
    }

    /*--- Output header information including initial residual ---*/

    if ((monitoring) && (masterRank)) {
      SU2_OMP_MASTER {
        WriteHeader("BCGSTAB", tol, norm_r);
        WriteHistory(i, norm_r / norm0);
      }
      END_SU2_OMP_MASTER
    }
  }

  /*--- Initialization ---*/

  ScalarType alpha = 1.0, omega = 1.0, rho = 1.0, rho_prime = 1.0;
  p = ScalarType(0.0);
  v = ScalarType(0.0);
  r_0 = r;

  /*--- Loop over all search directions ---*/

  for (i = 0; i < m; i++) {
    /*--- Compute rho_prime ---*/

    rho_prime = rho;

    /*--- Compute rho_i ---*/

    rho = r.dot(r_0);

    /*--- Compute beta ---*/

    ScalarType beta = (rho / rho_prime) * (alpha / omega);

    /*--- Update p ---*/

    p = beta * (p - omega * v) + r;

    /*--- Preconditioning step ---*/

    precond(p, z);
    mat_vec(z, v);

    /*--- Calculate step-length alpha ---*/

    ScalarType r_0_v = r_0.dot(v);
    alpha = rho / r_0_v;

    /*--- Update solution and residual ---*/

    x += alpha * z;
    r -= alpha * v;

    /*--- Preconditioning step ---*/

    precond(r, z);
    mat_vec(z, A_x);

    /*--- Calculate step-length omega, avoid division by 0. ---*/

    omega = A_x.squaredNorm();
    if (omega == ScalarType(0)) break;
    omega = A_x.dot(r) / omega;

    /*--- Update solution and residual ---*/

    x += omega * z;
    r -= omega * A_x;

    /*--- Only compute the residuals in full communication mode. ---*/

    if (config->GetComm_Level() == COMM_FULL) {
      /*--- Check if solution has converged, else output the relative residual if necessary ---*/

      norm_r = r.norm();
      if (norm_r < tol * norm0) break;
      if (((monitoring) && (masterRank)) && ((i + 1) % monitorFreq == 0)) {
        SU2_OMP_MASTER
        WriteHistory(i + 1, norm_r / norm0);
        END_SU2_OMP_MASTER
      }
    }
  }

  /*--- Recalculate final residual (this should be optional) ---*/

  if ((monitoring) && (config->GetComm_Level() == COMM_FULL)) {
    if (masterRank) {
      SU2_OMP_MASTER
      WriteFinalResidual("BCGSTAB", i, norm_r / norm0);
      END_SU2_OMP_MASTER
    }

    if (recomputeRes) {
      mat_vec(x, A_x);
      r = b - A_x;
      ScalarType true_res = r.norm();

      if ((fabs(true_res - norm_r) > tol * 10.0) && (masterRank)) {
        SU2_OMP_MASTER
        WriteWarning(norm_r, true_res, tol);
        END_SU2_OMP_MASTER
      }
    }
  }

  residual = norm_r / norm0;
  return i;
}

template <class ScalarType>
unsigned long CSysSolve<ScalarType>::BCGSTAB_LinSolver_GPU(const CSysVector<ScalarType>& b, CSysVector<ScalarType>& x,
                                                           CSysMatrix<ScalarType>& Jacobian, ScalarType tol,
                                                           unsigned long m, ScalarType& residual, bool monitoring,
                                                           CGeometry* geometry, const CConfig* config) const {
#ifdef HAVE_CUDA
  const bool masterRank = true;
  ScalarType norm_r = 0.0, norm0 = 0.0;
  unsigned long i = 0;

  if (m < 1) {
    SU2_MPI::Error("Number of linear solver iterations must be greater than 0.", CURRENT_FUNCTION);
  }

  if (!bcg_ready) {
    auto nVar = b.GetNVar();
    auto nBlk = b.GetNBlk();
    auto nBlkDomain = b.GetNBlkDomain();

    A_x.Initialize(nBlk, nBlkDomain, nVar, nullptr);
    r_0.Initialize(nBlk, nBlkDomain, nVar, nullptr);
    r.Initialize(nBlk, nBlkDomain, nVar, nullptr);
    p.Initialize(nBlk, nBlkDomain, nVar, nullptr);
    v.Initialize(nBlk, nBlkDomain, nVar, nullptr);
    z.Initialize(nBlk, nBlkDomain, nVar, nullptr);

    bcg_ready = true;
  }

  b.HtDTransfer();
  x.HtDTransfer();
  Jacobian.HtDTransfer();
  Jacobian.JacobiPreconditionerHtDTransfer();

  if (!xIsZero) {
    Jacobian.GPUMatrixVectorProduct(x, A_x, geometry, config, false, false);
    r.GPUCopyFrom(b);
    r.GPUAXPY(ScalarType(-1), A_x);
  } else {
    r.GPUCopyFrom(b);
  }

  norm_r = sqrt(r.GPUDot(r));
  norm0 = sqrt(b.GPUDot(b));

  if (tol_type == LinearToleranceType::RELATIVE) norm0 = norm_r;

  ScalarType alpha = 1.0, omega = 1.0, rho = 1.0, rho_prime = 1.0;
  p.GPUSetVal(0.0);
  v.GPUSetVal(0.0);
  r_0.GPUCopyFrom(r);

  for (i = 0; i < m; i++) {
    rho_prime = rho;
    rho = r.GPUDot(r_0);
    if (fabs(rho) < eps) break;

    ScalarType beta = (rho / rho_prime) * (alpha / omega);
    p.GPUUpdateP(v, r, beta, omega);

    Jacobian.ComputeJacobiPreconditionerGPU(p, z, false);
    Jacobian.GPUMatrixVectorProduct(z, v, geometry, config, false, false);

    ScalarType r_0_v = r_0.GPUDot(v);
    if (fabs(r_0_v) < eps) break;
    alpha = rho / r_0_v;

    x.GPUAXPY(alpha, z);
    r.GPUAXPY(-alpha, v);

    Jacobian.ComputeJacobiPreconditionerGPU(r, z, false);
    Jacobian.GPUMatrixVectorProduct(z, A_x, geometry, config, false, false);

    ScalarType A_x_sq = A_x.GPUDot(A_x);
    if (A_x_sq == ScalarType(0)) break;
    omega = A_x.GPUDot(r) / A_x_sq;

    x.GPUAXPY(omega, z);
    r.GPUAXPY(-omega, A_x);

    norm_r = sqrt(r.GPUDot(r));
    if (norm_r < tol * norm0) break;
  }
  residual = norm_r / norm0;
  x.DtHTransfer();
  return i;
#else
  (void)b;
  (void)x;
  (void)Jacobian;
  (void)tol;
  (void)m;
  (void)residual;
  (void)monitoring;
  (void)geometry;
  (void)config;
  SU2_MPI::Error("BCGSTAB_LinSolver_GPU requires CUDA support.", CURRENT_FUNCTION);
  return 0;
#endif
}

template <class ScalarType>
unsigned long CSysSolve<ScalarType>::AMGX_LinSolver(const CSysVector<ScalarType>& b, CSysVector<ScalarType>& x,
                                                    CSysMatrix<ScalarType>& Jacobian, ScalarType tol, unsigned long m,
                                                    bool monitoring, CGeometry* geometry,
                                                    const CConfig* config) const {
#ifdef HAVE_AMGX
  if (m < 1) {
    SU2_MPI::Error("Number of linear solver iterations must be greater than 0.", CURRENT_FUNCTION);
  } // Require at least one AmgX main-solver iteration.

  if (SU2_MPI::GetSize() != 1) {
    SU2_MPI::Error("AMGX_LinSolver currently supports single-rank execution only.", CURRENT_FUNCTION);
  } // This AmgX wrapper currently supports only single-rank runs.

  if (Jacobian.GetNVar() != Jacobian.GetNEqn()) {
    SU2_MPI::Error("AMGX_LinSolver requires square block matrices.", CURRENT_FUNCTION);
  } // Require square matrix blocks for the AmgX linear system.

  const auto n = static_cast<int>(Jacobian.GetNPointDomain());
  const auto nnz = static_cast<int>(Jacobian.GetNNZ());
  const auto block_dimx = static_cast<int>(Jacobian.GetNVar());
  const auto block_dimy = static_cast<int>(Jacobian.GetNEqn());
  std::vector<int> row_ptr(static_cast<size_t>(n) + 1);
  std::vector<int> col_ind(static_cast<size_t>(nnz));
  for (int i = 0; i < n + 1; ++i) row_ptr[i] = static_cast<int>(Jacobian.GetRowPtr()[i]);
  for (int i = 0; i < nnz; ++i) col_ind[i] = static_cast<int>(Jacobian.GetColInd()[i]);
  const auto* values = Jacobian.GetMatrixValues();
  const auto* rhs_ptr = b.begin();

  const auto mode = AMGXMode<ScalarType>();

  std::ostringstream cfg_stream;
  cfg_stream << "config_version=2, "
             << "block_format=ROW_MAJOR, "
             << "solver(main)=FGMRES, "
             << "main:preconditioner(amg)=AMG, "
             << "main:gmres_n_restart=10, "
             << "main:max_iters=" << m << ", "
             << "main:tolerance=" << tol << ", "
             << "main:convergence=RELATIVE_INI, "
             << "main:norm=L2, "
             << "main:use_scalar_norm=1, "
             << "main:monitor_residual=1, "
             << "main:print_solve_stats=0, "
             << "amg:algorithm=AGGREGATION, "
             << "amg:max_iters=1, "
             << "amg:presweeps=0, "
             << "amg:postsweeps=3, "
             << "amg:selector=SIZE_2, "
             << "amg:cycle=V, "
             << "amg:smoother=MULTICOLOR_DILU, "
             << "amg:error_scaling=0, "
             << "amg:max_levels=50, "
             << "amg:coarseAgenerator=LOW_DEG, "
             << "amg:matrix_coloring_scheme=PARALLEL_GREEDY, "
             << "amg:max_uncolored_percentage=0.05, "
             << "amg:relaxation_factor=0.75, "
             << "amg:coarse_solver=DENSE_LU_SOLVER, "
             << "amg:min_coarse_rows=32";

  static const AMGXRuntime amgx_runtime;
  (void)amgx_runtime;

  auto& cache = amgx_cache;
  const bool cache_matches = cache.cfg && cache.n == n && cache.nnz == nnz && cache.block_dimx == block_dimx &&
                             cache.block_dimy == block_dimy && cache.max_iters == m && cache.tol == tol;

  if (!cache_matches) {
    ResetAMGXCache();

    AMGX_config_handle cfg_handle;
    AMGX_resources_handle rsrc_handle;
    AMGX_matrix_handle A_handle;
    AMGX_vector_handle b_handle, x_handle;
    AMGX_solver_handle solver_handle;

    AMGX_SAFE_CALL(AMGX_config_create(&cfg_handle, cfg_stream.str().c_str()));
    AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc_handle, cfg_handle));
    AMGX_SAFE_CALL(AMGX_matrix_create(&A_handle, rsrc_handle, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&b_handle, rsrc_handle, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&x_handle, rsrc_handle, mode));
    AMGX_SAFE_CALL(AMGX_solver_create(&solver_handle, rsrc_handle, mode, cfg_handle));

    cache.cfg = cfg_handle;
    cache.rsrc = rsrc_handle;
    cache.matrix = A_handle;
    cache.rhs = b_handle;
    cache.sol = x_handle;
    cache.solver = solver_handle;
    cache.n = n;
    cache.nnz = nnz;
    cache.block_dimx = block_dimx;
    cache.block_dimy = block_dimy;
    cache.max_iters = m;
    cache.tol = tol;
  }

  auto A_handle = reinterpret_cast<AMGX_matrix_handle>(cache.matrix);
  auto b_handle = reinterpret_cast<AMGX_vector_handle>(cache.rhs);
  auto x_handle = reinterpret_cast<AMGX_vector_handle>(cache.sol);
  auto solver_handle = reinterpret_cast<AMGX_solver_handle>(cache.solver);

  if (!cache.matrix_uploaded) {
    AMGX_SAFE_CALL(AMGX_matrix_upload_all(A_handle, n, nnz, block_dimx, block_dimy, row_ptr.data(), col_ind.data(),
                                          values, nullptr));
    AMGX_SAFE_CALL(AMGX_solver_setup(solver_handle, A_handle));
    cache.matrix_uploaded = true;
  } else {
    AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A_handle, n, nnz, values, nullptr));
    AMGX_SAFE_CALL(AMGX_solver_resetup(solver_handle, A_handle));
  }
  AMGX_SAFE_CALL(AMGX_vector_upload(b_handle, n, block_dimy, rhs_ptr));
  AMGX_SAFE_CALL(AMGX_vector_upload(x_handle, n, block_dimx, &x[0]));

  AMGX_SAFE_CALL(AMGX_solver_solve(solver_handle, b_handle, x_handle));
  AMGX_SAFE_CALL(AMGX_vector_download(x_handle, &x[0]));

  AMGX_SOLVE_STATUS status = AMGX_SOLVE_FAILED;
  AMGX_SAFE_CALL(AMGX_solver_get_status(solver_handle, &status));
  int num_iters = 0;
  AMGX_SAFE_CALL(AMGX_solver_get_iterations_number(solver_handle, &num_iters));

  return (status == AMGX_SOLVE_SUCCESS) ? static_cast<unsigned long>(num_iters) : 0ul;
#else
  (void)b;
  (void)x;
  (void)Jacobian;
  (void)tol;
  (void)m;
  (void)monitoring;
  (void)geometry;
  (void)config;
  SU2_MPI::Error("AMGX_LinSolver requires a build with HAVE_AMGX enabled.", CURRENT_FUNCTION);
  return 0;
#endif
}


template <class ScalarType>
unsigned long CSysSolve<ScalarType>::Smoother_LinSolver(const CSysVector<ScalarType>& b, CSysVector<ScalarType>& x,
                                                        const CMatrixVectorProduct<ScalarType>& mat_vec,
                                                        const CPreconditioner<ScalarType>& precond, ScalarType tol,
                                                        unsigned long m, ScalarType& residual, bool monitoring,
                                                        const CConfig* config) const {
  const bool masterRank = (SU2_MPI::GetRank() == MASTER_NODE);
  const bool fix_iter_mode = tol < eps;
  ScalarType norm_r = 0.0, norm0 = 0.0;
  unsigned long i = 0;

  /*--- Relaxation factor, see comments inside the loop over the smoothing iterations. ---*/
  const ScalarType omega = SU2_TYPE::GetValue(config->GetLinear_Solver_Smoother_Relaxation());

  if (m < 1) {
    SU2_MPI::Error("Number of linear solver iterations must be greater than 0.", CURRENT_FUNCTION);
  }

  /*--- Allocate vectors for residual (r), solution increment (z), and matrix-vector
   product (A_x), for the latter two this is done only on the first call to the method. ---*/

  if (!smooth_ready) {
    BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS {
      auto nVar = b.GetNVar();
      auto nBlk = b.GetNBlk();
      auto nBlkDomain = b.GetNBlkDomain();

      A_x.Initialize(nBlk, nBlkDomain, nVar, nullptr);
      r.Initialize(nBlk, nBlkDomain, nVar, nullptr);
      z.Initialize(nBlk, nBlkDomain, nVar, nullptr);

      smooth_ready = true;
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS
  }

  /*--- Compute the initial residual and check if the system is already solved (if in COMM_FULL mode). ---*/

  if (!xIsZero) {
    mat_vec(x, A_x);
    r = b - A_x;
  } else {
    r = b;
  }

  /*--- Only compute the residuals in full communication mode. ---*/

  if (config->GetComm_Level() == COMM_FULL) {
    norm_r = r.norm();
    norm0 = b.norm();

    /*--- Set the norm to the initial initial residual value ---*/

    if (tol_type == LinearToleranceType::RELATIVE) norm0 = norm_r;

    if ((norm_r < tol * norm0) || (norm_r < eps)) {
      if (masterRank) {
        SU2_OMP_MASTER
        cout << "CSysSolve::Smoother_LinSolver(): system solved by initial guess." << endl;
        END_SU2_OMP_MASTER
      }
      return 0;
    }

    /*--- Output header information including initial residual. ---*/

    if ((monitoring) && (masterRank)) {
      SU2_OMP_MASTER {
        WriteHeader("Smoother", tol, norm_r);
        WriteHistory(i, norm_r / norm0);
      }
      END_SU2_OMP_MASTER
    }
  }

  /*--- Smoothing Iterations ---*/

  for (i = 0; i < m; i++) {
    /*--- Compute the solution increment (z), or "search" direction, by applying
     the preconditioner to the residual, i.e. z = M^{-1} * r. ---*/

    precond(r, z);

    /*--- The increment will be added to the solution (with relaxation omega)
     to update the residual, needed to compute the next increment, we get the
     product of the matrix with the direction (into A_x) and subtract from the
     current residual, the system is linear so this saves some computation
     compared to re-evaluating r = b-A*x. ---*/

    mat_vec(z, A_x);

    /*--- Update solution and residual with relaxation omega. Mathematically this
     is a modified Richardson iteration for the left-preconditioned system
     M^{-1}(b-A*x) which converges if ||I-w*M^{-1}*A|| < 1. Combining this method
     with a Gauss-Seidel preconditioner and w>1 is NOT equivalent to SOR. ---*/

    x += omega * z;
    r -= omega * A_x;

    /*--- Only compute the residuals in full communication mode. ---*/
    /*--- Check if solution has converged, else output the relative residual if necessary. ---*/

    if (!fix_iter_mode && config->GetComm_Level() == COMM_FULL) {
      norm_r = r.norm();
      if (norm_r < tol * norm0) break;
      if (((monitoring) && (masterRank)) && ((i + 1) % monitorFreq == 0)) {
        SU2_OMP_MASTER
        WriteHistory(i + 1, norm_r / norm0);
        END_SU2_OMP_MASTER
      }
    }
  }

  if (fix_iter_mode) norm_r = r.norm();

  if ((monitoring) && (masterRank) && (config->GetComm_Level() == COMM_FULL)) {
    SU2_OMP_MASTER
    WriteFinalResidual("Smoother", i, norm_r / norm0);
    END_SU2_OMP_MASTER
  }

  residual = norm_r / norm0;
  return i;
}

template <class ScalarType>
unsigned long CSysSolve<ScalarType>::Solve(CSysMatrix<ScalarType>& Jacobian, const CSysVector<su2double>& LinSysRes,
                                           CSysVector<su2double>& LinSysSol, CGeometry* geometry,
                                           const CConfig* config) {
  /*---
   A word about the templated types. It is assumed that the residual and solution vectors are always of su2doubles,
   meaning that they are active in the discrete adjoint. The same assumption is made in SetExternalSolve.
   When the Jacobian is passive (and therefore not compatible with the vectors) we go through the "HandleTemporaries"
   mechanisms. Note that CG, BCGSTAB, and FGMRES, all expect the vector to be compatible with the Product and
   Preconditioner (and therefore with the Matrix). Likewise for Solve_b (which is used by CSysSolve_b).
   There are no provisions here for active Matrix and passive Vectors as that makes no sense since we only handle the
   derivatives of the residual in CSysSolve_b.
  ---*/

  unsigned short KindSolver, KindPrecond;
  unsigned long MaxIter;
  ScalarType SolverTol;
  bool ScreenOutput;

  switch (lin_sol_mode) {
    /*--- Mesh Deformation mode ---*/
    case LINEAR_SOLVER_MODE::MESH_DEFORM: {
      KindSolver = config->GetKind_Deform_Linear_Solver();
      KindPrecond = config->GetKind_Deform_Linear_Solver_Prec();
      MaxIter = config->GetDeform_Linear_Solver_Iter();
      SolverTol = SU2_TYPE::GetValue(config->GetDeform_Linear_Solver_Error());
      ScreenOutput = config->GetDeform_Output();
      break;
    }

    /*--- Gradient Smoothing mode ---*/
    case LINEAR_SOLVER_MODE::GRADIENT_MODE: {
      KindSolver = config->GetKind_Grad_Linear_Solver();
      KindPrecond = config->GetKind_Grad_Linear_Solver_Prec();
      MaxIter = config->GetGrad_Linear_Solver_Iter();
      SolverTol = SU2_TYPE::GetValue(config->GetGrad_Linear_Solver_Error());
      ScreenOutput = true;
      break;
    }

    /*--- Normal mode
     * assumes that 'lin_sol_mode==LINEAR_SOLVER_MODE::STANDARD', but does not enforce it to avoid compiler warning.
     * ---*/
    default: {
      KindSolver = config->GetKind_Linear_Solver();
      KindPrecond = config->GetKind_Linear_Solver_Prec();
      MaxIter = config->GetLinear_Solver_Iter();
      SolverTol = SU2_TYPE::GetValue(config->GetLinear_Solver_Error());
      ScreenOutput = false;
      break;
    }
  }

  /*--- Stop the recording for the linear solver ---*/
  bool TapeActive = NO;

  if (config->GetDiscrete_Adjoint()) {
#ifdef CODI_REVERSE_TYPE

    TapeActive = AD::TapeActive();

    /*--- Declare external function inputs, outputs, and data ---*/
    BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS {
      AD::SetExtFuncIn(&LinSysRes[0], LinSysRes.GetLocSize());
      AD::SetExtFuncOut(&LinSysSol[0], LinSysSol.GetLocSize());
      AD::FuncHelper.addUserData(&LinSysRes);
      AD::FuncHelper.addUserData(&LinSysSol);
      AD::FuncHelper.addUserData(&Jacobian);
      AD::FuncHelper.addUserData(geometry);
      AD::FuncHelper.addUserData(config);
      AD::FuncHelper.addUserData(this);
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS
#endif
  }

  unsigned long IterLinSol = 0;

  /*--- Declaration of the external function ---*/
  auto externalFunction = [&]() {
    /*--- Create matrix-vector product, preconditioner, and solve the linear system ---*/

    HandleTemporariesIn(LinSysRes, LinSysSol);

    const auto kindPrec = static_cast<ENUM_LINEAR_SOLVER_PREC>(KindPrecond);
    auto mat_vec = CSysMatrixVectorProduct<ScalarType>(Jacobian, geometry, config);
    std::unique_ptr<CPreconditioner<ScalarType>> precond;

    if (KindSolver != AMGX) {
      precond.reset(CPreconditioner<ScalarType>::Create(kindPrec, Jacobian, geometry, config));
      precond->Build();
    }

    /*--- Solve system. ---*/

    ScalarType residual = Residual;
    bool updateResidual = true;

    switch (KindSolver) {
      case BCGSTAB:
        if (config->GetCUDA() && kindPrec == JACOBI && SU2_MPI::GetSize() == 1) {
          IterLinSol = BCGSTAB_LinSolver_GPU(*LinSysRes_ptr, *LinSysSol_ptr, Jacobian, SolverTol, MaxIter, residual,
                                             ScreenOutput, geometry, config);
        } else {
          IterLinSol = BCGSTAB_LinSolver(*LinSysRes_ptr, *LinSysSol_ptr, mat_vec, *precond, SolverTol, MaxIter,
                                         residual, ScreenOutput, config);
        }
        break;
      case AMGX:
        IterLinSol = AMGX_LinSolver(*LinSysRes_ptr, *LinSysSol_ptr, Jacobian, SolverTol, MaxIter, ScreenOutput,
                           geometry, config);
        updateResidual = false;
        break;
      case FGMRES:
        IterLinSol = FGMRES_LinSolver(*LinSysRes_ptr, *LinSysSol_ptr, mat_vec, *precond, SolverTol, MaxIter, residual,
                                      ScreenOutput, config);
        break;
      case RESTARTED_FGMRES:
        IterLinSol = RFGMRES_LinSolver(*LinSysRes_ptr, *LinSysSol_ptr, mat_vec, *precond, SolverTol, MaxIter, residual,
                                       ScreenOutput, config);
        break;
      case CONJUGATE_GRADIENT:
        IterLinSol = CG_LinSolver(*LinSysRes_ptr, *LinSysSol_ptr, mat_vec, *precond, SolverTol, MaxIter, residual,
                                  ScreenOutput, config);
        break;
      case SMOOTHER:
        IterLinSol = Smoother_LinSolver(*LinSysRes_ptr, *LinSysSol_ptr, mat_vec, *precond, SolverTol, MaxIter, residual,
                                        ScreenOutput, config);
        break;
      case PASTIX_LDLT:
      case PASTIX_LU:
        Jacobian.BuildPastixPreconditioner(geometry, config, KindSolver);
        Jacobian.ComputePastixPreconditioner(*LinSysRes_ptr, *LinSysSol_ptr, geometry, config);
        IterLinSol = 1;
        residual = 1e-20;
        break;
      default:
        SU2_MPI::Error("Unknown type of linear solver.", CURRENT_FUNCTION);
    }

    SU2_OMP_MASTER {
      if (updateResidual) Residual = residual;
      Iterations = IterLinSol;
    }
    END_SU2_OMP_MASTER

    HandleTemporariesOut(LinSysSol);

    if (TapeActive) {
      /*--- To keep the behavior of SU2_DOT, but not strictly required since jacobian is symmetric(?). ---*/
      const bool RequiresTranspose =
          ((lin_sol_mode != LINEAR_SOLVER_MODE::MESH_DEFORM) || (config->GetKind_SU2() == SU2_COMPONENT::SU2_DOT));

      if (lin_sol_mode == LINEAR_SOLVER_MODE::MESH_DEFORM)
        KindPrecond = config->GetKind_Deform_Linear_Solver_Prec();
      else if (lin_sol_mode == LINEAR_SOLVER_MODE::GRADIENT_MODE)
        KindPrecond = config->GetKind_Grad_Linear_Solver_Prec();
      else
        KindPrecond = config->GetKind_DiscAdj_Linear_Prec();

      /*--- Build preconditioner for the transposed Jacobian ---*/

      if (RequiresTranspose) Jacobian.TransposeInPlace();

      switch (KindPrecond) {
        case ILU:
          if (RequiresTranspose) Jacobian.BuildILUPreconditioner();
          break;
        case JACOBI:
        case LINELET:
          if (RequiresTranspose) Jacobian.BuildJacobiPreconditioner();
          break;
        case LU_SGS:
          /*--- Nothing to build. ---*/
          break;
        case PASTIX_ILU:
        case PASTIX_LU_P:
        case PASTIX_LDLT_P:
          /*--- It was already built. ---*/
          break;
        default:
          SU2_MPI::Error("The specified preconditioner is not yet implemented for the discrete adjoint method.",
                         CURRENT_FUNCTION);
          break;
      }
    }
  }; /*--- Finish declaration of the external function ---*/

#ifdef CODI_REVERSE_TYPE
  /*--- Call the external function with appropriate AD handling ---*/
  AD::FuncHelper.callPrimalFuncWithADType(externalFunction);

  AD::FuncHelper.addToTape(CSysSolve_b<ScalarType>::Solve_b);
#else
  /*--- Without reverse AD, call the external function directly ---*/
  externalFunction();
#endif

  return IterLinSol;
}

template <class ScalarType>
unsigned long CSysSolve<ScalarType>::Solve_b(CSysMatrix<ScalarType>& Jacobian, const CSysVector<su2double>& LinSysRes,
                                             CSysVector<su2double>& LinSysSol, CGeometry* geometry,
                                             const CConfig* config, const bool directCall) {
  unsigned short KindSolver, KindPrecond;
  unsigned long MaxIter, IterLinSol = 0;
  ScalarType SolverTol;
  bool ScreenOutput;

  switch (lin_sol_mode) {
    /*--- Mesh Deformation mode ---*/
    case LINEAR_SOLVER_MODE::MESH_DEFORM: {
      KindSolver = config->GetKind_Deform_Linear_Solver();
      KindPrecond = config->GetKind_Deform_Linear_Solver_Prec();
      MaxIter = config->GetDeform_Linear_Solver_Iter();
      SolverTol = SU2_TYPE::GetValue(config->GetDeform_Linear_Solver_Error());
      ScreenOutput = config->GetDeform_Output();
      break;
    }

    /*--- Gradient Smoothing mode ---*/
    case LINEAR_SOLVER_MODE::GRADIENT_MODE: {
      KindSolver = config->GetKind_Grad_Linear_Solver();
      KindPrecond = config->GetKind_Grad_Linear_Solver_Prec();
      MaxIter = config->GetGrad_Linear_Solver_Iter();
      SolverTol = SU2_TYPE::GetValue(config->GetGrad_Linear_Solver_Error());
      ScreenOutput = true;
      break;
    }

    /*--- Normal mode
     * assumes that 'lin_sol_mode==LINEAR_SOLVER_MODE::STANDARD', but does not enforce it to avoid compiler warning.
     * ---*/
    default: {
      KindSolver = config->GetKind_Linear_Solver();
      KindPrecond = config->GetKind_Linear_Solver_Prec();
      MaxIter = config->GetLinear_Solver_Iter();
      SolverTol = SU2_TYPE::GetValue(config->GetLinear_Solver_Error());
      ScreenOutput = false;
      break;
    }
  }

  /*--- Set up preconditioner and matrix-vector product ---*/

  const auto kindPrec = static_cast<ENUM_LINEAR_SOLVER_PREC>(KindPrecond);
  std::unique_ptr<CPreconditioner<ScalarType>> precond;

  /*--- If there was no call to solve first the preconditioner needs to be built here. ---*/
  if (KindSolver != AMGX) {
    precond.reset(CPreconditioner<ScalarType>::Create(kindPrec, Jacobian, geometry, config));
  }

  if (directCall && precond) {
    Jacobian.TransposeInPlace();
    precond->Build();
  }

  auto mat_vec = CSysMatrixVectorProduct<ScalarType>(Jacobian, geometry, config);

  /*--- Solve the system ---*/

  /*--- Local variable to prevent all threads from writing to a shared location (this->Residual). ---*/
  ScalarType residual = 0.0;

  HandleTemporariesIn(LinSysRes, LinSysSol);

  switch (KindSolver) {
    case FGMRES:
      IterLinSol = FGMRES_LinSolver(*LinSysRes_ptr, *LinSysSol_ptr, mat_vec, *precond, SolverTol, MaxIter, residual,
                                    ScreenOutput, config);
      break;
    case RESTARTED_FGMRES:
      IterLinSol = RFGMRES_LinSolver(*LinSysRes_ptr, *LinSysSol_ptr, mat_vec, *precond, SolverTol, MaxIter, residual,
                                     ScreenOutput, config);
      break;
    case BCGSTAB:
      IterLinSol = BCGSTAB_LinSolver(*LinSysRes_ptr, *LinSysSol_ptr, mat_vec, *precond, SolverTol, MaxIter, residual,
                                     ScreenOutput, config);
      break;
    case AMGX:
      SU2_MPI::Error("AMGX linear solver is not implemented in Solve_b.", CURRENT_FUNCTION);
      break;
    case CONJUGATE_GRADIENT:
      IterLinSol = CG_LinSolver(*LinSysRes_ptr, *LinSysSol_ptr, mat_vec, *precond, SolverTol, MaxIter, residual,
                                ScreenOutput, config);
      break;
    case SMOOTHER:
      IterLinSol = Smoother_LinSolver(*LinSysRes_ptr, *LinSysSol_ptr, mat_vec, *precond, SolverTol, MaxIter, residual,
                                      ScreenOutput, config);
      break;
    case PASTIX_LDLT:
    case PASTIX_LU:
      if (directCall) Jacobian.BuildPastixPreconditioner(geometry, config, KindSolver);
      Jacobian.ComputePastixPreconditioner(*LinSysRes_ptr, *LinSysSol_ptr, geometry, config);
      IterLinSol = 1;
      residual = 1e-20;
      break;
    default:
      SU2_MPI::Error("Unknown type of linear solver.", CURRENT_FUNCTION);
      break;
  }

  HandleTemporariesOut(LinSysSol);

  SU2_OMP_MASTER {
    Residual = residual;
    Iterations = IterLinSol;
  }
  END_SU2_OMP_MASTER

  return IterLinSol;
}

/*--- Explicit instantiations ---*/

#ifdef CODI_FORWARD_TYPE
template class CSysSolve<su2double>;
#else
template class CSysSolve<su2mixedfloat>;
#ifdef USE_MIXED_PRECISION
template class CSysSolve<passivedouble>;
#endif
#endif

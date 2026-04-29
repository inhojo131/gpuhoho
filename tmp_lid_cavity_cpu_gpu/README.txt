2D lid-driven cavity timing cases for CPU/GPU/AmgX comparison.

Files:
- lid_cavity_cpu.cfg
- lid_cavity_gpu.cfg
- lid_cavity_amgx.cfg
- lid_cavity_*_cfl1.cfg
- lid_cavity_*_cfl1_iter40000.cfg
- lid_cavity_*_cfl5_iter8000.cfg

Design choices:
- Same physics and mesh for all cases
- CPU/GPU native cases use BCGSTAB + JACOBI
- AmgX case uses the AMGX linear solver wrapper
- Same multigrid setting: MGLEVEL=0
- Single-rank execution recommended for all cases, and required for the current GPU BCGSTAB and AmgX paths
- RECTANGLE mesh is used so no external mesh file is needed

Build:
- From the repository root
- ninja -C build

Run CPU:
- cd tmp_lid_cavity_cpu_gpu
- /usr/bin/time -f "CPU elapsed: %E" ../build/SU2_CFD/src/SU2_CFD lid_cavity_cpu.cfg 2>&1 | tee cpu_run.log

Run GPU:
- cd tmp_lid_cavity_cpu_gpu
- /usr/bin/time -f "GPU BCGSTAB elapsed: %E" ../build/SU2_CFD/src/SU2_CFD lid_cavity_gpu.cfg 2>&1 | tee gpu_bcgstab_run.log

Run AmgX:
- cd tmp_lid_cavity_cpu_gpu
- /usr/bin/time -f "AMGX elapsed: %E" ../build/SU2_CFD/src/SU2_CFD lid_cavity_amgx.cfg 2>&1 | tee amgx_run.log

Useful comparisons:
- Wall clock reported by /usr/bin/time
- WALL_TIME in the solver screen output
- history_cpu.csv vs history_gpu.csv vs history_amgx.csv for convergence trend

Notes:
- The optimized native GPU BCGSTAB path keeps the Jacobian and Jacobi inverse on the device for the duration of each linear solve.
- The AmgX wrapper requires block_format=ROW_MAJOR for SU2 block matrices and uses monitored residuals for robust convergence.
- If you want a quicker smoke test, reduce ITER from 2000 to 200 or 500.

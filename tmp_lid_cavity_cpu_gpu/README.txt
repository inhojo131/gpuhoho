2D lid-driven cavity timing cases for CPU/GPU comparison.

Files:
- lid_cavity_cpu.cfg
- lid_cavity_gpu.cfg

Design choices:
- Same physics and mesh for both cases
- Same linear solver: BCGSTAB + JACOBI
- Same multigrid setting: MGLEVEL=0
- Single-rank execution recommended for both, and required for the current GPU BCGSTAB path
- RECTANGLE mesh is used so no external mesh file is needed

Build:
- From /mnt/data2/ihj0303/gpuhoho
- ninja -C build

Run CPU:
- cd /mnt/data2/ihj0303/gpuhoho/tmp_lid_cavity_cpu_gpu
- /usr/bin/time -f "CPU elapsed: %E" ../build/SU2_CFD/src/SU2_CFD lid_cavity_cpu.cfg | tee cpu_run.log

Run GPU:
- cd /mnt/data2/ihj0303/gpuhoho/tmp_lid_cavity_cpu_gpu
- /usr/bin/time -f "GPU elapsed: %E" ../build/SU2_CFD/src/SU2_CFD lid_cavity_gpu.cfg | tee gpu_run.log

Useful comparisons:
- Wall clock reported by /usr/bin/time
- WALL_TIME in the solver screen output
- history_cpu.csv vs history_gpu.csv for convergence trend

Notes:
- The current GPU linear-solver path is intended for single-rank use.
- If you want a quicker smoke test, reduce ITER from 2000 to 200 or 500.

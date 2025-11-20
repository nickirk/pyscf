# Hybrid Parallelism & KPAR Complexity in KPWSCF

The user asked about handling `np > nk` (more MPI ranks than k-points) and implementing a scheme similar to VASP's `KPAR`. This document explains the current limitations and the complexity of implementing such a scheme.

## Current Parallelization (Pure K-point MPI)

Currently, `KPWSCF` uses **pure K-point parallelization**:
-   **Distribution**: The `nk` k-points are distributed as evenly as possible among `size` MPI ranks.
-   **Efficiency**: This is highly efficient when `nk >= size` because k-points are largely independent (except for charge density reduction and exchange broadcasting).
-   **Limitation (`np > nk`)**: If `size > nk`, some ranks (specifically `size - nk` ranks) are assigned **0 k-points**.
    -   These ranks sit idle during most computations (`_block_davidson`).
    -   They *must* still participate in global collectives (`allreduce`, `Bcast`) to prevent deadlocks (which we just fixed).
    -   **Result**: Wasted computational resources. Speedup saturates at `np = nk`.

## Hybrid MPI (K-point + Band/Grid) / "KPAR"

To utilize `np > nk`, we need to parallelize *within* a single k-point calculation. This is often done via **Band** or **Grid** parallelization.

### VASP's KPAR Approach
VASP's `KPAR` parameter divides the `MPI_COMM_WORLD` into groups:
-   **K-point Groups**: Ranks are divided into `KPAR` groups. Each group handles a subset of k-points.
-   **Band/Grid Groups**: Within each K-point group, multiple ranks work together on the *same* k-point.

### Implementation Complexity

Implementing this in `KPWSCF` (PySCF) is non-trivial:

1.  **Communicator Splitting**:
    -   We need to split `MPI_COMM_WORLD` into sub-communicators:
        -   `comm_k`: For communication between k-point groups (e.g., density reduction).
        -   `comm_band`: For communication within a k-point group (e.g., FFTs, orthogonalization).
    -   **Code Impact**: Every `comm.allreduce` or `comm.Bcast` currently in the code assumes `MPI_COMM_WORLD`. We would need to pass specific communicators to every method.

2.  **Distributed FFTs (Grid Parallelism)**:
    -   If multiple ranks work on one k-point, the 3D grid `(mx, my, mz)` must be distributed (e.g., slab decomposition).
    -   **Complexity**: Requires **parallel 3D FFTs** (MPI-aware FFTW).
    -   Current `pyfftw` usage is serial (within a rank). Switching to MPI-FFT requires significant changes to data layout and transposes.

3.  **Distributed Linear Algebra (Band Parallelism)**:
    -   Davidson diagonalization involves matrices of size `(nband, nband)`.
    -   Orthogonalization (`QR`, `eigh`) would need to be distributed (e.g., ScaLAPACK) if `nband` is large, or replicated if small.
    -   **Complexity**: Managing distributed arrays and using parallel linear algebra libraries.

4.  **Memory Management**:
    -   Data distribution changes completely. Instead of `psi_g[local_k, nband, ngrids]`, it might be `psi_g[local_k, local_nband, ngrids]` or `psi_g[local_k, nband, local_ngrids]`.

## Alternative: MPI + OpenMP (Threading)

A simpler approach to utilize more cores when `np > nk` is **Hybrid MPI + OpenMP**:
-   **MPI**: Set `np = nk` (or a divisor of `nk`).
-   **OpenMP**: Use multiple threads per MPI rank (`OMP_NUM_THREADS > 1`).

### Pros:
-   **No Code Changes**: PySCF and NumPy/SciPy already release the GIL and use OpenMP/MKL for heavy operations (einsum, dot, FFTs).
-   **Memory Efficient**: Threads share memory within a rank.

### Cons:
-   **Python GIL**: Python code (loops, logic) runs on 1 thread. Only C-extensions (NumPy, FFTW) parallelize.
-   **FFT Scaling**: 3D FFT scaling with threads is good but not perfect.
-   **Amdahl's Law**: Serial Python parts become the bottleneck.

### Recommendation
For now, the best way to run `KPWSCF` when you have many cores is:
1.  Set `MPI ranks (np)` to be equal to (or a factor of) `nk`.
2.  Set `OMP_NUM_THREADS` such that `np * OMP_NUM_THREADS = Total Cores`.

Example: 64 cores, 8 k-points.
-   Run with `mpirun -n 8` and `export OMP_NUM_THREADS=8`.

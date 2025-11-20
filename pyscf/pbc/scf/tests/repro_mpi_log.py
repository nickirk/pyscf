import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.scf import kpwscf
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_test():
    cell = gto.Cell()
    cell.atom = 'He 0 0 0'
    cell.a = np.eye(3) * 3.0
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [10, 10, 10]
    cell.verbose = 4
    cell.build()

    # Use fewer k-points than ranks (if np > 2)
    # Let's use 2 k-points
    kpts = cell.make_kpts([2, 1, 1])
    
    if rank == 0:
        print(f"Running with np={size}, nk={len(kpts)}")
        cell.verbose = 4
    else:
        cell.verbose = 0
        
    cell.build()

    mf = kpwscf.KPWSCF(cell, kpts=kpts)
    # mf.verbose is set in __init__, let's see if it sticks
    
    mf.kernel(max_cycle=2)

if __name__ == '__main__':
    run_test()

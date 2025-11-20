#!/usr/bin/env python
# Copyright 2025 The PySCF Developers.
# 
# Profiling script for KPWSCF with C2 molecule in a box
# Mesh: 24x24x24, K-mesh: 2x2x1

import sys
import os
import numpy as np
import cProfile
import pstats
from io import StringIO

from pyscf import lib
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.lib.kpts import make_kpts

# Import kpwscf from local directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kpwscf import KPWSCF


def make_c2_cell():
    """Create C2 molecule in a box with specified mesh."""
    cell = pbcgto.Cell()
    cell.atom = '''
    H 0.0 0.0 0.0
    O 1.2425 0.0 0.0
    H 2.485 0.0 0.0
    '''
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    # Box size: 10 Bohr in each direction to minimize interaction between images
    cell.a = np.eye(3) * 4.0
    cell.mesh = [24, 24, 24]
    cell.verbose = 4
    cell.build()
    return cell


def profile_kpwscf():
    """Profile KPWSCF with C2 molecule."""
    print("=" * 80)
    print("Profiling KPWSCF for C2 molecule")
    print("=" * 80)
    print("System configuration:")
    print("  Molecule: C2 (1.2425 Bohr bond length)")
    print("  Box size: 10x10x10 Bohr")
    print("  FFT mesh: 24x24x24")
    print("  K-mesh: 2x2x1")
    print("=" * 80)
    
    # Create cell
    cell = make_c2_cell()
    print(f"Cell built successfully")
    print(f"  Number of electrons: {cell.nelectron}")
    print(f"  Volume: {cell.vol:.2f} Bohr^3")
    
    # Create k-points mesh
    # Use cell.make_kpts for k-point generation
    kpts = cell.make_kpts([2, 2, 1])
    print(f"  Number of k-points: {len(kpts)}")
    print(f"  K-points:\n{kpts}")
    
    # Profile the initialization
    print("\n" + "=" * 80)
    print("Profiling KPWSCF initialization...")
    print("=" * 80)
    
    profiler_init = cProfile.Profile()
    profiler_init.enable()
    
    mf = KPWSCF(cell, kpts=kpts, nband=10)
    
    profiler_init.disable()
    
    # Print initialization profile
    s = StringIO()
    ps = pstats.Stats(profiler_init, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print("\nInitialization Profile (top 30 functions by cumulative time):")
    print(s.getvalue())
    
    # Profile the SCF calculation
    print("\n" + "=" * 80)
    print("Profiling KPWSCF SCF calculation...")
    print("=" * 80)
    
    profiler_scf = cProfile.Profile()
    profiler_scf.enable()
    
    try:
        mf.kernel(max_cycle=5)
    except Exception as e:
        print(f"Error during SCF: {e}")
        import traceback
        traceback.print_exc()
    
    profiler_scf.disable()
    
    # Print SCF profile
    s = StringIO()
    ps = pstats.Stats(profiler_scf, stream=s).sort_stats('cumulative')
    ps.print_stats(50)
    print("\nSCF Profile (top 50 functions by cumulative time):")
    print(s.getvalue())
    
    # Also print by total time
    s = StringIO()
    ps = pstats.Stats(profiler_scf, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print("\nSCF Profile (top 30 functions by total time):")
    print(s.getvalue())
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    if hasattr(mf, 'converged') and mf.converged:
        print(f"  SCF converged: {mf.converged}")
        print(f"  Total energy: {mf.e_tot:.8f} Ha")
        print(f"  Number of iterations: {getattr(mf, 'niter', 'N/A')}")
    else:
        print("  SCF did not converge or encountered an error")
    
    # Save detailed stats to file
    with open('/Users/keliao/Work/project/pyscf/profile_kpwscf_c2_init.stats', 'w') as f:
        ps = pstats.Stats(profiler_init, stream=f).sort_stats('cumulative')
        ps.print_stats()
    print("\nDetailed initialization profile saved to: profile_kpwscf_c2_init.stats")
    
    with open('/Users/keliao/Work/project/pyscf/profile_kpwscf_c2_scf.stats', 'w') as f:
        ps = pstats.Stats(profiler_scf, stream=f).sort_stats('cumulative')
        ps.print_stats()
    print("Detailed SCF profile saved to: profile_kpwscf_c2_scf.stats")
    
    return mf


if __name__ == '__main__':
    profile_kpwscf()

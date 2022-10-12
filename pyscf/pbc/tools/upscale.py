#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



"""
This module upscales wavefunction obtained on a sparse k-mesh to one that is
defined on a denser k-mesh, using info from the former.

"""

import warnings
import copy
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.gto import ATM_SLOTS, BAS_SLOTS, ATOM_OF, PTR_COORD
from pyscf.pbc.lib.kpts_helper import get_kconserv, get_kconserv3
from pyscf.pbc import scf, mp, df, cc
from pyscf.pbc.mp.kmp2 import padding_k_idx, _add_padding
from pyscf.pbc.mp import kmp2
from pyscf import __config__
from pyscf.lib.parameters import LARGE_DENOM


def set_up_method(cell, k_mesh):
    """
    Function to set up calculation.

    Args:
        cell: cell object
        k_mesh: list of integers of size 3

    Returns:
        kmf: scf object
        kmp: MP2 object
    """
    kpts = cell.make_kpts(k_mesh, with_gamma_point=True)
    kmf = scf.KRHF(cell, kpts, exxdiv='ewald')
    gdf = df.GDF(cell, kpts).build()
    kmf.with_df = gdf
    kmf.kernel()
    kmp = mp.KMP2(kmf)

    return kmf, kmp

def upscale(t1, t2, kmp_s, kmp_d, nn_table, n_shell=1):
    """
    Args:
        t1: np array of size [nkpts_s, nocc, nvir]
        t2: np array of size [nkpts_s, nkpts_s, nkpts_s, nocc, nocc, nvir, nvir]
        kmp_s: mp2 object on a sparse k-mesh
        kmp_s: mp2 object on a dense k-mesh
        nn_table: 2D np array. [nkpts_d, nkpts_s],
                  number of dense k-points x number of sparse k-points
        n_shell: integer. Determine how far should it neighbours go.

    Returns:
        t2_d: The upscaled T2 amplitudes. The same size as t2_d
    """
    # loop over each k-point in k_d

    kconserv_s = kmp_s.khelper.kconserv
    kconserv_d = kmp_d.khelper.kconserv

    nkpts_d = len(kmp_d.kpts)
    nkpts_s = len(kmp_s.kpts)
    nmo = kmp_s.nmo
    nocc = kmp_s.nocc
    nvir = nmo - nocc

    with_df_ints = kmp_d.with_df_ints and isinstance(kmp_d._scf.with_df, df.GDF)

    mem_avail = kmp_d.max_memory - lib.current_memory()[0]
    mem_usage = ((nkpts_d + nkpts_s) * (nocc * nvir)**2) * 16 / 1e6
    if with_df_ints:
        mydf = kmp_d._scf.with_df
        if mydf.auxcell is None:
            # Calculate naux based on precomputed GDF integrals
            naux = mydf.get_naoaux()
        else:
            naux = mydf.auxcell.nao_nr()

        mem_usage += (nkpts_d**2 * naux * nocc * nvir) * 16 / 1e6

    mem_usage += (nkpts_d**3 * (nocc * nvir)**2) * 16 / 1e6
    mem_usage += (nkpts_s**3 * (nocc * nvir)**2) * 16 / 1e6

    if mem_usage > mem_avail:
        raise MemoryError('Insufficient memory! MP2 memory usage %d MB (currently available %d MB)'
                          % (mem_usage, mem_avail))

    t1_d = np.zeros((nkpts_d, nocc, nvir),
                    dtype=kmp_s.t2.dtype)
    t2_d = np.zeros((nkpts_d, nkpts_d, nkpts_d, nocc, nocc, nvir, nvir),
                    dtype=kmp_s.t2.dtype)

    # Build 3-index DF tensor Lov
    if with_df_ints:
        Lov = kmp2._init_mp_df_eris(kmp_d)

    oovv_ij = np.zeros((nkpts_d,nocc,nocc,nvir,nvir), dtype=kmp_d.mo_coeff[0].dtype)
    emp2_us = 0.

    for ki in range(nkpts_d):
        ki_nn = nn_table[ki, :].argsort()[:n_shell]
        tot_weight = np.zeros([nocc, nvir])
        for ki_s in ki_nn:
            weight = get_singles_denom(kmp_d, ki) / get_singles_denom(kmp_s, ki_s)
            weight = np.abs(weight)
            t1_d[ki] += t1[ki_s] * weight
            tot_weight += weight

        if tot_weight.all() != 0.:
            t1_d[ki] /= tot_weight
            t1_d[ki] *= nkpts_s / nkpts_d
            #t1_d[ki] *= np.sign(t1[ki])

    for ki in range(nkpts_d):
        print("ki =", ki, ", nkpts = ", nkpts_d)
        ki_nn = nn_table[ki, :].argsort()[:n_shell]
        for kj in range(nkpts_d):
            kj_nn = nn_table[kj, :].argsort()[:n_shell]
            for ka in range(nkpts_d):
                kb = kconserv_d[ki,ka,kj]

                if with_df_ints:
                    oovv_ij[ka] = (1./nkpts_d) * lib.einsum("Lia,Ljb->iajb", Lov[ki, ka], Lov[kj, kb]).transpose(0,2,1,3)
                else:
                    orbo_i = kmp_d.mo_coeff[ki][:,:nocc]
                    orbo_j = kmp_d.mo_coeff[kj][:,:nocc]
                    orbv_a = kmp_d.mo_coeff[ka][:,nocc:]
                    orbv_b = kmp_d.mo_coeff[kb][:,nocc:]
                    oovv_ij[ka] = kmp2.fao2mo((orbo_i,orbv_a,orbo_j,orbv_b),
                                         (kmp_d.kpts[ki],kmp_d.kpts[ka],kmp_d.kpts[kj],kmp_d.kpts[kb]),
                                         compact=False).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts_d

            for ka in range(nkpts_d):
                kb = kconserv_d[ki,ka,kj]
                ka_nn = nn_table[ka, :].argsort()[:n_shell]
                kb_nn = nn_table[kb, :].argsort()[:n_shell]
                tot_weight = np.zeros([nocc, nocc, nvir, nvir])

                for ki_s in ki_nn:
                    for ka_s in ka_nn:
                        for kj_s in kj_nn:
                            #kb_s = kconserv_s[ki_s, ka_s, kj_s]
                                
                            #if kb_s in kb_nn:
                            #weight = get_doubles_denom(kmp_s, [ki_s, kj_s, ka_s, kb_s])
                            #weight /= get_doubles_denom(kmp_d, [ki, kj, ka, kb])
                            weight = kmp_d.t2[ki, kj, ka]/kmp_s.t2[ki_s, kj_s, ka_s]
                            weight = np.abs(weight)
                            #t2_d[ki, kj, ka] += t2[ki_s, kj_s, ka_s] * weight
                            t2_d[ki, kj, ka] += np.abs(t2[ki_s, kj_s, ka_s]) * weight
                            tot_weight += weight
                if tot_weight.all() != 0.:
                    t2_d[ki, kj, ka] /= tot_weight
                    t2_d[ki, kj, ka] *= nkpts_s / nkpts_d
                    t2_d[ki, kj, ka] *= np.sign(kmp_d.t2[ki, kj, ka])
                    
                else:
                    raise RuntimeWarning("Total weight cannot be 0!")
                
                woovv = 2 * oovv_ij[ka] - oovv_ij[kb].transpose(0, 1, 3, 2)
                t2_tmp = lib.einsum("ia, jb-> ijab", t1_d[ki], t1_d[kj])
                t2_tmp += t2_d[ki, kj, ka]
                emp2_us += lib.einsum("ijab, ijab", t2_tmp, woovv).real



    emp2_us /= nkpts_d

    return emp2_us, t1_d, t2_d

def get_singles_denom(kmp, ki):
    """
    Calculate the denominator in for singles amplitudes.

    Args:
        kmp: KMP2 object.
        ki: Integer. ki

    Returns:
        eia: np array [nocc, nvir], at the specified k-points
    """
    nocc = kmp.nocc
    nvir = kmp.nmo - kmp.nocc
    

    mo_e_o = kmp.mo_energy[ki][:nocc]
    mo_e_v = kmp.mo_energy[ki][nocc:]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(kmp, kind="split")
    # Remove zero/padded elements from denominator
    eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=kmp.mo_energy[0].dtype)
    n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ki])
    eia[n0_ovp_ia] = (mo_e_o[:,None] - mo_e_v)[n0_ovp_ia]

    return eia

def get_doubles_denom(kmp, kpts):
    """
    Calculate the denominator in MP2 amplitudes.

    Args:
        kmp: KMP2 object.
        kpts: list of 4 integers. [ki, kj, ka, kb], ki, kj... are the indices
              of the k-points.
        ijab: list of 4 integers. [i, j, a, b]  are the indices
              of the orbitals.

    Returns:
        eijab: np array [nocc, nocc, nvir, nvir], at the specified k-points
        and i, j, a, b indices.
    """
    nocc = kmp.nocc
    nvir = kmp.nmo - kmp.nocc
    if len(kpts) != 4:
        raise ValueError
    ki, kj, ka, kb = kpts[:]

    mo_e_o = [kmp.mo_energy[k][:nocc] for k in range(kmp.nkpts)]
    mo_e_v = [kmp.mo_energy[k][nocc:] for k in range(kmp.nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(kmp, kind="split")
    # Remove zero/padded elements from denominator
    eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=kmp.mo_energy[0].dtype)
    n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
    eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

    ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=kmp.mo_energy[0].dtype)
    n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
    ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]

    eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)

    return eijab


def get_nn_dist(kpts_s, kpts_d, n_shell=1):
    """
    Function to find for each k-point in kpts_d the nearest k-points to it on kpts_s.

    Args:
        kpts_s: np.ndarray, size [nkpts_s, 3]
        kpts_d: np.ndarray, size [nkpts_d, 3]
        n_shell: integer. Number of shells of near neighbours to include

    Returns:
         dist_ds: np.ndarray, size [nkpts_d, nkpts_s], distance between two
         k-points in the dense and sparse k-meshes.
    """
    dist_ds = lib.direct_sum("ni+mi ->nmi", kpts_d, -kpts_s)
    dist_ds = np.sqrt(lib.einsum("nmi, nmi -> nm", dist_ds, dist_ds))

    return dist_ds


if __name__ == '__main__':
    from pyscf.pbc import gto, mp
    from pyscf.pbc import scf

    '''
    Example calculation on H2 chain
    '''
    cell = gto.Cell()
    cell.pseudo = 'gth-pade'
    cell.basis = 'gth-dzv'
    cell.ke_cutoff = 50
    cell.atom='''
        H 2.00   2.00   1.20
        H 2.00   2.00   2.60
        '''
    cell.a = '''
        4.0   0.0   0.0
        0.0   4.0   0.0
        0.0   0.0   4.0
        '''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()

    # need a function to set up the systems.
    nks_mf_s = [2, 2, 3]
    kmf_s, kmp_s = set_up_method(cell, nks_mf_s)

    nks_mf_d = [2, 2, 3]
    kmf_d, kmp_d = set_up_method(cell, nks_mf_d)

    #ehf_s = kmf_s.kernel()
    #ehf_d = kmf_d.kernel()
    dist_nm = get_nn_dist(kmf_s.kpts, kmf_d.kpts)

    emp2_s, t2_s = kmp_s.kernel()
    emp2_d, t2_d = kmp_d.kernel()

    #emp2_us, t2_us = upscale(t2_s, kmp_s, kmp_d, dist_nm, 1)
    #abs_diff = np.abs(t2_d) - np.abs(t2_us)
    #diff_norm = np.linalg.norm(abs_diff)
    #diff_e = emp2_us - emp2_d
    #print("Upscaled MP2 e = ", emp2_us)


    # set up CCSD 
    e_us = 0.
    mycc = cc.KCCSD(kmf_s)
    ecc, t1, t2 = mycc.kernel()
    e_us, t1_us, t2_us = upscale(t1, t2, kmp_s, kmp_d, dist_nm, 1)
    print(e_us, emp2_s, emp2_d)


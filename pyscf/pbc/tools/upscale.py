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

def get_approx_singles(kmp_d, t2):
    """Approximate t1 based on the upscaled t2 amplitudes.

    Args:
        kmp_d (KMP2 object): MP2 object defined on the dense k-mesh.
        t2 (np ndarray): T2 amplitudes (the upscaled or any other T2 amplitudes) defined on the dense k-mesh.

    Returns:
        t1: The approximated T1 amplitudes.
    """
    nocc = kmp_d.nocc
    nvir = kmp_d.nmo - nocc
    nkpts = kmp_d.nkpts

    kconserv = kmp_d.khelper.kconserv

    t1 = np.zeros([nkpts, nocc, nvir], dtype=t2.dtype)

    ooov_ij = np.array([nkpts, nocc, nocc, nocc, nvir])
    # need V_ooov for the first step approximation.
    if kmp_d.with_df_ints:
        Loo = _init_upscale_df_eris(kmp_d)
        Lov = kmp2._init_mp_df_eris(kmp_d)
    else:
        raise NotImplementedError    

    for ki in range(nkpts):
        for kj in range(nkpts):
            for kk in range(nkpts):
                ka = kconserv(ki, kk, kj)
                ooov_ij[ka] = (1./nkpts) * lib.einsum("Lij,Lka->ijka", Loo[ki, kj], Lov[kk, ka]).transpose(0,2,1,3)
    
            for ka in range(nkpts):
                ke = kconserv[ki, ka, kj]
                t1[ka] += lib.einsum("klic, ackl -> ai", ooov_ij[ke], t2[ki, kj, ke])

    for ka in range(nkpts):
        ki = ka
        for km in range(nkpts):
            for kn in range(nkpts):
                ke = kconserv[km, ki, kn]
                #t1new[ka] += -0.5 * einsum('imef,maef->ia', t2[ki, km, ke], eris.ovvv[km, ka, ke])
                # FIXME: the ooov_ij need transpose and verification
                t1[ka] += -0.5 * lib.einsum('mnae,nmei->ia', t2[km, kn, ka], ooov_ij[kn, km, ke])

    return t1

def upscale(kcc_s, kmp_s, kmp_d, nn_table, n_shell=1, **kwargs):
    """
    Args:
        kcc_s (CCSD object): defined on a sparse k-mesh. Its t1 and t2 will be upscaled.
        kmp_s (MP2 object): defined on a sparse k-mesh.
        kmp_s (MP2 object): defined on a dense k-mesh.
        nn_table: 2D np array. [nkpts_d, nkpts_s],
                  number of dense k-points x number of sparse k-points
        n_shell: integer. Determine how far should its neighbours go.
    
    kwargs:
        t1_phase (np array): dtyep=complex128. Provide the phase structure for the upscaled t1, 
            should have the same size as t1.
        t2_phase (np array): dtyep=complex128. Provide the phase structure for the upscaled t2, 
            should have the same size as t2.

    Returns:
        t1_us: The upscaled T1 amplitudes. [nkpts_d, nocc, nvir]
        t2_us: The upscaled T2 amplitudes. The same size as kmp_d.t2
    """
    # loop over each k-point in k_d

    log = logger.Logger(kmp_d.stdout, kmp_d.verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
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

    t1_us = np.zeros((nkpts_d, nocc, nvir),
                    dtype=kmp_s.t2.dtype)
    t2_us = np.zeros((nkpts_d, nkpts_d, nkpts_d, nocc, nocc, nvir, nvir),
                    dtype=kmp_s.t2.dtype)

    if "t1_phase" in kwargs.keys():
        t1_phase = kwargs["t1_phase"]

    if "t2_phase" in kwargs.keys():
        t2_phase = kwargs["t2_phase"]
    else:
        t2_phase = kmp_d.t2/np.abs(kmp_d.t2) 

    for ki in range(nkpts_d):
        ki_nn = nn_table[ki, :].argsort()[:n_shell]
        #tot_weight = np.zeros([nocc, nvir])
        for ki_s in ki_nn:
            weight = get_singles_denom(kmp_d, ki) / get_singles_denom(kmp_s, ki_s)
            weight = np.abs(weight)
            t1_us[ki] += kcc_s.t1[ki_s] * weight
            #tot_weight += weight

    #    if tot_weight.all() != 0.:
    #        t1_us[ki] /= tot_weight
    #        t1_us[ki] *= np.sqrt(nkpts_s / nkpts_d)
    #        #t1_d[ki] *= np.sign(t1[ki])

    for ki in range(nkpts_d):
        log.info("Finished %d of total %d kpts ", ki+1, nkpts_d)
        ki_nn = nn_table[ki, :].argsort()[:n_shell]
        for kj in range(nkpts_d):
            kj_nn = nn_table[kj, :].argsort()[:n_shell]
            for ka in range(nkpts_d):
                ka_nn = nn_table[ka, :].argsort()[:n_shell]

                for ki_s in ki_nn:
                    for ka_s in ka_nn:
                        for kj_s in kj_nn:
                            weight = kmp_d.t2[ki, kj, ka]/kmp_s.t2[ki_s, kj_s, ka_s]
                            weight = np.abs(weight)
                            t2_us[ki, kj, ka] += np.abs(kcc_s.t2[ki_s, kj_s, ka_s]) * weight
                            t2_us[ki, kj, ka] *= t2_phase[ki, kj, ka]
                


    log.timer("upscale", *cput0)

    return t1_us, t2_us


def get_energy(t1, t2, kmf):
    """calculate correlation energy based on provided t1 and t2 amplitudes, kmf provides 

    Args:
        t1 (np array): size [nkpts, nocc, nvir]
        t2 (np array): size [nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]
        kmf (scf object): mean-field object to provide eri or df for contraction with t1 and t2 or construct 
            Coulomb integrals
    Returns:
        e_corr (float): correlation energy
    """

    # need kmp2 object for DF
    mykmp2 = mp.KMP2(kmf)
    nmo = mykmp2.nmo
    nocc = mykmp2.nocc
    nvir = nmo - nocc
    nkpts = mykmp2.nkpts

    with_df_ints = isinstance(kmf.with_df, df.GDF)

    mem_avail = mykmp2.max_memory - lib.current_memory()[0]
    mem_usage = (nkpts * (nocc * nvir)**2) * 16 / 1e6
    if with_df_ints:
        mydf = kmf.with_df
        if mydf.auxcell is None:
            # Calculate naux based on precomputed GDF integrals
            naux = mydf.get_naoaux()
        else:
            naux = mydf.auxcell.nao_nr()

    # Build 3-index DF tensor Lov
    if with_df_ints:
        Lov = kmp2._init_mp_df_eris(mykmp2)

    oovv_ij = np.zeros((nkpts,nocc,nocc,nvir,nvir), dtype=kmf.mo_coeff[0].dtype)
    e_corr = 0.
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = mykmp2.khelper.kconserv[ki,ka,kj]

                if with_df_ints:
                    oovv_ij[ka] = (1./nkpts) * lib.einsum("Lia,Ljb->iajb", Lov[ki, ka], Lov[kj, kb]).transpose(0,2,1,3)
                else:
                    orbo_i = mykmp2.mo_coeff[ki][:,:nocc]
                    orbo_j = mykmp2.mo_coeff[kj][:,:nocc]
                    orbv_a = mykmp2.mo_coeff[ka][:,nocc:]
                    orbv_b = mykmp2.mo_coeff[kb][:,nocc:]
                    oovv_ij[ka] = kmp2.fao2mo((orbo_i,orbv_a,orbo_j,orbv_b),
                                         (mykmp2.kpts[ki],mykmp2.kpts[ka],mykmp2.kpts[kj],mykmp2.kpts[kb]),
                                         compact=False).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts
            for ka in range(nkpts):
                kb = mykmp2.khelper.kconserv[ki,ka,kj]
                woovv = 2 * oovv_ij[ka] - oovv_ij[kb].transpose(0, 1, 3, 2)
                t2_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=t2[0,0,0,0].dtype)
                t2_tmp = lib.einsum("ia, jb-> ijab", t1[ka], t1[ka])
                t2_tmp += t2[ki, kj, ka]
                e_corr += lib.einsum("ijab, ijab", t2_tmp.conj(), woovv).real
    e_corr /= nkpts

    return e_corr

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
    

    mo_e_o = [kmp.mo_energy[ki][:nocc]]
    mo_e_v = [kmp.mo_energy[ki][nocc:]]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(kmp, kind="split")
    # Remove zero/padded elements from denominator
    eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=kmp.mo_energy[0].dtype)
    n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ki])
    eia[n0_ovp_ia] = (mo_e_o[0][:,None] - mo_e_v[0])[n0_ovp_ia]

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

def coarse_grain(mp_d, mp_s, ns=3):
    kpts_d = mp_d.kpts[::ns]
    kpts_s = mp_s.kpts
    assert np.allclose(kpts_d, kpts_s)
    nkpts_s = mp_s.nkpts
    nmo = mp_d.nmo
    nocc = mp_d.nocc
    nvir = nmo - nocc

    mp_s.mo_energy = mp_d.mo_energy[::ns]
    mp_s.mo_coeff = mp_d.mo_coeff[::ns]


    with_df_ints = isinstance(mp_d._scf.with_df, df.GDF)

    if with_df_ints:
        mydf = mp_d._scf.with_df
        if mydf.auxcell is None:
            # Calculate naux based on precomputed GDF integrals
            naux = mydf.get_naoaux()
        else:
            naux = mydf.auxcell.nao_nr()

    # Build 3-index DF tensor Lov
    if with_df_ints:
        Lov = kmp2._init_mp_df_eris(mp_d)[::ns, ::ns]

    t2 = np.zeros((nkpts_s, nkpts_s, nkpts_s,nocc,nocc,nvir,nvir), dtype=mp_d.mo_coeff[0].dtype)
    oovv_ij = np.zeros((nkpts_s,nocc,nocc,nvir,nvir), dtype=mp_d.mo_coeff[0].dtype)
    e_corr = 0.
    for ki in range(nkpts_s):
        for kj in range(nkpts_s):
            for ka in range(nkpts_s):
                kb = mp_s.khelper.kconserv[ki,ka,kj]

                if with_df_ints:
                    oovv_ij[ka] = (1./nkpts_s) * lib.einsum("Lia,Ljb->iajb", Lov[ki, ka], Lov[kj, kb]).transpose(0,2,1,3)
                else:
                    orbo_i = mp_s.mo_coeff[ki][:,:nocc]
                    orbo_j = mp_s.mo_coeff[kj][:,:nocc]
                    orbv_a = mp_s.mo_coeff[ka][:,nocc:]
                    orbv_b = mp_s.mo_coeff[kb][:,nocc:]
                    oovv_ij[ka] = kmp2.fao2mo((orbo_i,orbv_a,orbo_j,orbv_b),
                                         (mp_s.kpts[ki],mp_s.kpts[ka],mp_s.kpts[kj],mp_s.kpts[kb]),
                                         compact=False).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts_s
            for ka in range(nkpts_s):
                kb = mp_s.khelper.kconserv[ki,ka,kj]
                eijab = get_doubles_denom(mp_s, [ki, kj, ka, kb])
                t2_ijab = np.conj(oovv_ij[ka]/eijab)
                woovv = 2 * oovv_ij[ka] - oovv_ij[kb].transpose(0, 1, 3, 2)
                t2[ki, kj, ka] = t2_ijab
                e_corr += lib.einsum("ijab, ijab", t2_ijab, woovv).real
    e_corr /= nkpts_s
    return e_corr, t2

def _init_upscale_df_eris(mp):
    """Compute 3-center electron repulsion integrals, i.e. (L|oo),
    where `L` denotes DF auxiliary basis functions and `o` and `v` occupied and virtual
    canonical crystalline orbitals. Note that `o` and `v` contain kpt indices `ko` and `kv`,
    and the third kpt index `kL` is determined by the conservation of momentum.

    Arguments:
        mp (KMP2) -- A KMP2 instance

    Returns:
        Lov (numpy.ndarray) -- 3-center DF ints, with shape (nkpts, nkpts, naux, nocc, nvir)
    """
    from pyscf.ao2mo import _ao2mo
    from pyscf.pbc.lib.kpts_helper import gamma_point

    log = logger.Logger(mp.stdout, mp.verbose)

    if mp._scf.with_df._cderi is None:
        mp._scf.with_df.build()

    cell = mp._scf.cell
    if cell.dimension == 2:
        # 2D ERIs are not positive definite. The 3-index tensors are stored in
        # two part. One corresponds to the positive part and one corresponds
        # to the negative part. The negative part is not considered in the
        # DF-driven CCSD implementation.
        raise NotImplementedError

    nocc = mp.nocc
    nmo = mp.nmo
    nao = cell.nao_nr()

    mo_coeff = _add_padding(mp, mp.mo_coeff, mp.mo_energy)[0]
    kpts = mp.kpts
    nkpts = len(kpts)
    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *mo_coeff)
    Loo = np.empty((nkpts, nkpts), dtype=object)

    cput0 = (logger.process_clock(), logger.perf_counter())

    bra_start = 0
    bra_end = nocc
    ket_start = nmo
    ket_end = ket_start + nocc
    with df.CDERIArray(mp._scf.with_df._cderi) as cderi_array:
        tao = []
        ao_loc = None
        for ki in range(nkpts):
            for kj in range(nkpts):
                Lpq_ao = cderi_array[ki,kj]

                mo = np.hstack((mo_coeff[ki], mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order='F')
                if dtype == np.double:
                    out = _ao2mo.nr_e2(Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), aosym='s2')
                else:
                    #Note: Lpq.shape[0] != naux if linear dependency is found in auxbasis
                    if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                        Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
                    out = _ao2mo.r_e2(Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), tao, ao_loc)
                Loo[ki, kj] = out.reshape(-1, nocc, nocc)

    log.timer_debug1("transforming Loo integrals", *cput0)

    return Loo




if __name__ == '__main__':
    from pyscf.pbc import gto, mp
    from pyscf.pbc import scf

    '''
    Example calculation on H2 chain
    '''
    cell = gto.Cell()
    cell.pseudo = 'gth-pade'
    cell.basis = 'sto6g'
    cell.ke_cutoff = 50
    cell.atom='''
        H 2.00   2.00   1.20
        H 2.00   2.00   2.60
        H 2.00   1.20   2.00
        H 2.00   2.60   2.00
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
    #nks_mf_s = [3, 3, 3]
    nks_mf_s = [1, 1, 2]
    kmf_s, kmp_s = set_up_method(cell, nks_mf_s)

    nks_mf_d = [1, 1, 4]
    #nks_mf_d = [3, 3, 3]
    kmf_d, kmp_d = set_up_method(cell, nks_mf_d)

    #ehf_s = kmf_s.kernel()
    #ehf_d = kmf_d.kernel()
    dist_nm = get_nn_dist(kmf_s.kpts, kmf_d.kpts)

    emp2_s, t2_s = kmp_s.kernel()
    emp2_d, t2_d = kmp_d.kernel()


    # set up CCSD 
    e_us = 0.
    mycc = cc.KCCSD(kmf_s)
    mycc.max_cycle = 50
    ecc, t1_cc_s, t2_cc_s = mycc.kernel()
    mycc_d = cc.KCCSD(kmf_d)
    mycc_d.max_cycle = 0
    ecc, t1_mp2_d, t2_mp2_d = mycc_d.kernel()
    mycc_d.max_cycle = 50
    ecc, t1_cc_d, t2_cc_d = mycc_d.kernel()
    t1_us, t2_us = upscale(mycc, kmp_s, kmp_d, dist_nm, 1, t2_phase=t2_cc_d/np.abs(t2_cc_d))
    e_us = get_energy(t1_cc_d, t2_us, kmf_d)
    abs_diff_abs_sum = np.abs(t2_us) - np.abs(t2_cc_d)
    abs_diff_abs_sum = np.einsum("xyzijab ->", np.abs(abs_diff_abs_sum))
    diff_abs_sum = t2_us - t2_cc_d
    diff_abs_sum = np.einsum("xyzijab ->", np.abs(diff_abs_sum))
    print("t2 abs_diff_sum = ", abs_diff_abs_sum)
    print("t2 diff_sum = ", diff_abs_sum)
    print(e_us, emp2_s, emp2_d, ecc)
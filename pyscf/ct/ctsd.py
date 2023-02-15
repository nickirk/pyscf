#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
#
# Author: Ke Liao <ke.liao.whu@gmail.com>
#

"""
Canonical Transformation module.
This module consists of:
1), the construction of the CT Hamiltonian in the active space 
(with external space downfolded by the similarity transformation), which
can be solved by any quantum chemistry methods;
"""
import numpy as np
from scipy.optimize import newton_krylov

from pyscf import lib
from pyscf.lib import logger, diis
from pyscf import ao2mo, gto, scf
from functools import reduce
from pyscf.cc.ccsd import _ChemistsERIs


def tensor_analysis(t):
    if len(t.shape) == 2:
        logger.info(self, "***** order 2 tensor")
        logger.info(self, " sum = ", np.sum(t))
        logger.info(self, " abs sum = ", np.sum(np.abs(t)))
        logger.info(self, " nonsymmetriness = ", np.sum(t - t.T))
        logger.info(self, " nondiagonalness = ", np.sum(np.abs(t)) - np.sum(np.abs(t.diagonal())))
    elif len(t.shape) == 4:
        logger.info(self, "***** order 4 tensor")
        logger.info(self, " sum = ", np.sum(t))
        logger.info(self, " abs sum = ", np.sum(np.abs(t)))
        logger.info(self, " nonsymmetriness 0123 - 1032= ", np.sum(np.abs(t - t.transpose((1, 0, 3, 2)).copy())))
        logger.info(self, " nonsymmetriness 0123 - 3210= ", np.sum(np.abs(t - t.transpose((3, 2, 1, 0)).copy())))
        logger.info(self, " nonsymmetriness 0123 - 2301= ", np.sum(np.abs(t - t.transpose((2, 3, 0, 1)).copy())))
        logger.info(self, " nonsymmetriness 0123 - 1023= ", np.sum(np.abs(t - t.transpose((1, 0, 2, 3)).copy())))
        logger.info(self, " nonsymmetriness 0123 - 0132= ", np.sum(np.abs(t - t.transpose((0, 1, 3, 2)).copy())))

def symmetrize(t):
    """This function symmetrizes tensor t of dim 2 or 4.

    Args:
        t (np ndarray): 2 or 4 dimension np ndarray to be symmetrized.
    
    Returns:
        t_sym (np ndarray): symmetrized tensor of the same size as the input tensor.

    """
    t_sym = t.copy()
    if t.ndim == 2:
        t_sym += t.T.copy()
        t_sym /= 2.
    elif t.ndim == 4:
        t_sym += t.transpose((1, 0, 3, 2)).copy()
        t_sym += t.transpose((2, 3, 0, 1)).copy()
        t_sym += t.transpose((3, 2, 1, 0)).copy()
        t_sym *= 1./4.
    else:
        raise ValueError("Incorrect length of tensor: "+str(len(t))+"! Only 2 or 4 is allowed.")

    return t_sym

    
def get_d3_slice_direct(dm1=None, dm2=None, slices=None):
    """
    Function to compute a block of the 3-RDM, according to the provided
    slices *without exchange of indices*.

    Args:
        dm1: 2D array. 1-RDM
        dm2: 4D array. 2-RDM
        slices: integers specifying the starting and stop positions
                for each of the six indices of D^{p1p2p3}_{q1q2q3}.
                Example input:
                [0, 100, 0, 100, 0, 50, 0, 100, 0, 100, 0, 50]
                If set to None, get the full d3 tensor.
    Returns:
        d3: 6D array.
    """
    if slices is None:
        nmo = dm1.shape[0]
        slices = [0, nmo, 0, nmo, 0, nmo, 0, nmo, 0, nmo, 0, nmo]
    if len(slices) != 12:
        raise RuntimeError("Length of slices is not 12!")
    p1_lo, p2_lo, p3_lo, q1_lo, q2_lo, q3_lo = slices[::2]
    p1_up, p2_up, p3_up, q1_up, q2_up, q3_up = slices[1::2]

    d3 = np.zeros(np.asarray(slices[1::2])-np.asarray(slices[::2]))

    dm1_tmp = dm1[p1_lo:p1_up, q1_lo:q1_up]
    dm2_tmp = dm2[p2_lo:p2_up, p3_lo:p3_up, q2_lo:q2_up, q3_lo:q3_up]
    d3 += lib.einsum("mn, pqrs -> mpqnrs", dm1_tmp, dm2_tmp)

    dm1_tmp = dm1[p1_lo:p1_up, q2_lo:q2_up]
    dm2_tmp = dm2[p2_lo:p2_up, p3_lo:p3_up, q1_lo:q1_up, q3_lo:q3_up]
    d3 -= 1 / 2 * lib.einsum("mn, pqrs -> mpqrns", dm1_tmp, dm2_tmp)

    dm1_tmp = dm1[p1_lo:p1_up, q3_lo:q3_up]
    dm2_tmp = dm2[p2_lo:p2_up, p3_lo:p3_up, q2_lo:q2_up, q1_lo:q1_up]
    d3 -= 1 / 2 * lib.einsum("mn, pqrs -> mpqsrn", dm1_tmp, dm2_tmp)

    return d3


def get_d3_slice(dm1=None, dm2=None, slices=None):
    p1_lo, p2_lo, p3_lo, q1_lo, q2_lo, q3_lo = slices[::2]
    p1_up, p2_up, p3_up, q1_up, q2_up, q3_up = slices[1::2]

    d3 = get_d3_slice_direct(dm1, dm2, slices)

    slices = [p2_lo, p2_up, p3_lo, p3_up, p1_lo, p1_up,
              q2_lo, q2_up, q3_lo, q3_up, q1_lo, q1_up]

    d3 += np.transpose(get_d3_slice_direct(dm1, dm2, slices),
                       (2, 0, 1, 5, 3, 4))

    slices = [p3_lo, p3_up, p1_lo, p1_up, p2_lo, p2_up,
              q3_lo, q3_up, q1_lo, q1_up, q2_lo, q2_up]

    d3 += np.transpose(get_d3_slice_direct(dm1, dm2, slices),
                       (1, 2, 0, 4, 5, 3))
    return d3

def find_inds_bound(ind, t_nmo, nmo):
    ind_bounds = []
    for p in ind:
        if p in "xy":
            ind_bounds.append([t_nmo, nmo])
        elif p in "mnuv":
            ind_bounds.append([0, nmo])
        elif p in "pqrs":
            ind_bounds.append([0, t_nmo])
        else:
            raise ValueError("Index %s not recongnized", p)
    return ind_bounds

def part_dms(dm1_inds, dm2_inds, dm1, dm2, t_nmo, nmo):
    bd = find_inds_bound(dm1_inds, t_nmo, nmo)
    dm1_part = dm1[bd[0][0]:bd[0][1], bd[1][0]:bd[1][1]].copy()
    bd = find_inds_bound(dm2_inds, t_nmo, nmo)
    dm2_part = dm2[bd[0][0]:bd[0][1], bd[1][0]:bd[1][1], bd[2][0]:bd[2][1], bd[3][0]:bd[3][1]].copy()

    return dm1_part, dm2_part

def contr_dm3(o2_inds, o3_inds, v, dm1, dm2, t_nmo, nmo):
    p1, p2, p3, p4, p5, p6 = o3_inds[:]

    dm1_inds = p1+p4
    dm2_inds = p2+p3+p5+p6
    dm1_part, dm2_part = part_dms(dm1_inds, dm2_inds, dm1, dm2, t_nmo, nmo)
    r = lib.einsum(o2_inds+", "+dm1_inds+", "+dm2_inds+" -> xypq", v, dm1_part, dm2_part)

    dm1_inds = p1+p5
    dm2_inds = p2+p3+p4+p6
    dm1_part, dm2_part = part_dms(dm1_inds, dm2_inds, dm1, dm2, t_nmo, nmo)
    r -= 0.5 * lib.einsum(o2_inds+", "+dm1_inds+", "+dm2_inds+" -> xypq", v, dm1_part, dm2_part)

    dm1_inds = p1+p6
    dm2_inds = p2+p3+p5+p4
    dm1_part, dm2_part = part_dms(dm1_inds, dm2_inds, dm1, dm2, t_nmo, nmo)
    r -= 0.5 * lib.einsum(o2_inds+", "+dm1_inds+", "+dm2_inds+" -> xypq", v, dm1_part, dm2_part)

    dm1_inds = p2+p5
    dm2_inds = p3+p1+p6+p4
    dm1_part, dm2_part = part_dms(dm1_inds, dm2_inds, dm1, dm2, t_nmo, nmo)
    r += lib.einsum(o2_inds+", "+dm1_inds+", "+dm2_inds+"-> xypq", v, dm1_part, dm2_part)

    dm1_inds = p2+p6
    dm2_inds = p3+p1+p5+p4
    dm1_part, dm2_part = part_dms(dm1_inds, dm2_inds, dm1, dm2, t_nmo, nmo)
    r -= 0.5 * lib.einsum(o2_inds+", "+dm1_inds+", "+dm2_inds+" -> xypq", v, dm1_part, dm2_part)

    dm1_inds = p2+p4
    dm2_inds = p3+p1+p6+p5
    dm1_part, dm2_part = part_dms(dm1_inds, dm2_inds, dm1, dm2, t_nmo, nmo)
    r -= 0.5 * lib.einsum(o2_inds+", "+dm1_inds+", "+dm2_inds+"-> xypq", v, dm1_part, dm2_part)

    dm1_inds = p3+p6
    dm2_inds = p1+p2+p4+p5
    dm1_part, dm2_part = part_dms(dm1_inds, dm2_inds, dm1, dm2, t_nmo, nmo)
    r += lib.einsum(o2_inds+", "+dm1_inds+", "+dm2_inds+" -> xypq", v, dm1_part, dm2_part)

    dm1_inds = p3+p4
    dm2_inds = p1+p2+p6+p5
    dm1_part, dm2_part = part_dms(dm1_inds, dm2_inds, dm1, dm2, t_nmo, nmo)
    r -= 0.5 * lib.einsum(o2_inds+", "+dm1_inds+", "+dm2_inds+" -> xypq", v, dm1_part, dm2_part)

    dm1_inds = p3+p5
    dm2_inds = p1+p2+p4+p6
    dm1_part, dm2_part = part_dms(dm1_inds, dm2_inds, dm1, dm2, t_nmo, nmo)
    r -= 0.5 * lib.einsum(o2_inds+", "+dm1_inds+", "+dm2_inds+" -> xypq", v, dm1_part, dm2_part)

    return r


def get_d_zero(dm1=None, dm2=None):
    """
    function to construct \mathring{D}^{p1p2}{q1q2} (Eq 18) in
    Ref: Phys. Chem. Chem. Phys., 2012, 14, 7809–7820

    Args:
        dm1: 2D array. 1-RDM
        dm2: 4D array. 2-RDM

    Returns:
        d0: 4D array. Modified 2-RDM
    """

    d0 = -dm2
    d0 += 4. / 3 * lib.einsum("mn, uv -> munv", dm1, dm1)
    d0 -= 4. / 3 * 1. / 2 * lib.einsum("mn, uv -> muvn", dm1, dm1)

    return d0


def get_d_bar(dm1=None, dm2=None):
    """
    function to construct \mathring{D}^{p1p2}{q1q2} (Eq 19) in
    Ref: Phys. Chem. Chem. Phys., 2012, 14, 7809–7820

    Args:
        dm1: 2D array. 1-RDM in mo basis
        dm2: 4D array. 2-RDM in mo basis

    Returns:
        d_bar: 4D array. Modified 2-RDM
    """

    d_bar = -dm2
    d_bar += 2. * lib.einsum("mn, uv -> munv", dm1, dm1)
    d_bar -= lib.einsum("mv, un -> munv", dm1, dm1)

    return d_bar

def get_d_hat(dm1=None, dm2=None):
    """
    function to construct \mathring{\hat{D}}^{p1p2}{q1q2} (Eq 21) in
    Ref: Phys. Chem. Chem. Phys., 2012, 14, 7809–7820

    Args:
        dm1: 2D array. 1-RDM in mo basis
        dm2: 4D array. 2-RDM in mo basis

    Returns:
        d_bar: 4D array. Modified 2-RDM
    """

    d_hat = dm2.copy()
    d_hat -= 2./3. * lib.einsum("mn, uv -> munv", dm1, dm1)
    d_hat += 2./3. * 1./2 * lib.einsum("mv, un -> munv", dm1, dm1)

    return d_hat


class CTSD(lib.StreamObject):
    """
    Canonical transformation class.

    Attributes:
       t1: the singles amplitudes for CT. It can be supplied by the user or 
           computed internally by MP2 or F12.
       t2: the doubles amplitudes for CT. It can be supplied by the user or 
           computed internally by MP2 or F12.
       t_basis: target basis set. Indices p, q, r, s for general orbitals,
                i,j... for occupied and a, b...for virtual.
       e_basis: external basis set. Indices, x, y, z

    Saved results:
        ct_0
        ct_o1: transformed 1-body integrals.
        ct_o2: transformed 2-body integrals.
    """

    def __init__(self, mf,
                 a_nmo=None, mo_coeff=None, mo_occ=None,
                 dm1=None, dm2=None, h_core=None, fock=None, eri=None):
        """
        mf: mean field object.
        """
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ

        self.mol = mf.mol
        self.mf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        # total number of orbitals
        self.nmo = len(mf.mo_energy)
        # core orbitals are doubly occupied, indices i,j,k,l...
        # currently only closed shell systems are supported
        self.c_nmo = int(np.sum(mf.mo_occ) / 2)
        # active orbitals are those which are fractionally or low-lying
        # virtuals, a,b,c,d...
        if a_nmo is None:
            a_nmo = int(self.c_nmo * 1.5)
        self.a_nmo = a_nmo
        # target orbitals are the combination of core and active orbitals, 
        # indices p,q,r,s
        self.t_nmo = a_nmo + self.c_nmo
        # external orbitals are empty virtuals excluding the ones that are 
        # included in active space, x,y,z
        self.e_nmo = self.nmo - self.t_nmo
        #self.incore_complete = self.incore_complete or self.mol.incore_anyway

        self.ct_0 = None
        self.ct_o1 = None
        self.ct_o2 = None
        if eri is None:
            eri = self.ao2mo()
        self.eri = eri
        self.h_core = h_core
        self.fock = fock
        # iterative solution 
        self.max_cycle = 50
        # max number of bch expansion
        self.build_hbar_use_bch = True
        self.n_bch = 50
        self.conv_tol_e = 1e-7
        self.conv_tol_normt = 1e-6
        self.diis_space = 8
        self.get_res_counter = 0
        self.gs_only = False
        ##################################################
        # don't modify the following attributes, they are not input options
        self.mo_coeff = mo_coeff
        self._mo_energy = None

        self.mo_occ = mo_occ
        self.emp2 = None
        self.e_hf = None
        self.e_corr = None
        self._t2s = np.zeros([self.e_nmo, self.e_nmo, self.t_nmo, self.t_nmo])
        self._t1s = np.zeros([self.e_nmo, self.t_nmo])
        self.t1, self.t2 = self.part_amps()
        self.is_amp_init = False
        self._nocc = None
        self._nmo = None
        self._amps_algo = None
        self.chkfile = mf.chkfile
        self.callback = None


        # need 1-RDM and 2-RDM in mo representation.
        if dm1 is None:
            dm1 = np.diag(self.mf.mo_occ)
        if dm2 is None:
            dm2 = (np.einsum('ij, kl -> ikjl', dm1, dm1) - np.einsum(
                'il, kj -> ikjl', dm1, dm1) / 2.)
        self.dm1 = dm1
        self.dm2 = dm2

        keys = set(('amps_algo'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('CTSD nocc = %s, a_nmo = %s, e_nmo = %s, t_nmo = %s', self.nocc, self.a_nmo, self.e_nmo, self.t_nmo)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('build_hbar_use_bch = %s', self.build_hbar_use_bch)
        log.info('gs_only = %s', self.gs_only)
        if self.build_hbar_use_bch:
            log.info('n_bch = %d', self.n_bch)
        else:
            log.info('amps_algo = %s', self.amps_algo)
        log.info('conv_tol_e = %g', self.conv_tol_e)
        log.info('conv_tol_normt = %s', self.conv_tol_normt)
        log.info('diis_space = %d', self.diis_space)
        #log.info('diis_file = %s', self.diis_file)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def build_hbar(self, bch=False, cutoff=1e-8, n_max=10, **kwargs):
        """
        Main function which calls other functions to calculate all the parts of
        the CT Hamiltonian and assemble them.

        pseudo-code:
        1. Get the original Hamiltonian integrals H and \bar{H} <- H;
        2. Get the first commutator [H, T]_{1,2} and add its contribution to
        \bar{H};
        3. Get the second commutator and add its contribution to \bar{H}.
        There are two options to calculate the second commutator,
            i) [[H, T]_{1,2}, T]_{1,2} or
            ii) [[F, T]_{1,2}, T]_{1,2} as in Watson and Chan paper.
        The default will be the latter.

        Args:


        Returns:
            ct_0: constant term in \bar{H}, float
            ct_o1: singles term in \bar{H}, ndarray
            ct_o2: doubles term in \bar{H}, ndarray
        """

        if self.eri is None:
            self.eri = self.ao2mo()
        if bch is not None:
            self.build_hbar_use_bch = bch
        if not bch:
            self.dump_flags()
        # initialize the amplitudes
        if "amps_algo" in kwargs:
            self.amps_algo = kwargs["amps_algo"]
        if "t1" in kwargs:
            t1 = kwargs["t1"] 
        else:
            t1 = None
        if "t2" in kwargs:
            t2 = kwargs["t2"] 
        else:
            t2 = None
        self.t1, self.t2 = self.init_amps(t1, t2)


        # we need the one-body part of the original Hamiltonian h_mn
        if self.h_core is None:
            h_mn = self.mf.get_hcore()
            h_mn = self.ao2mo(h_mn)
        else:
            h_mn = self.h_core
        
        if self.build_hbar_use_bch:
            logger.info(self, "    using BCH expansion up to max %s terms", n_max)
            h_norm = np.inf
            # bch construction of H bar, \bar{H} = H + \sum_n 1/n! [[[...]]]
            ct_0 = 0.
            ct_o1 = h_mn.copy()
            ct_o2 = self.eri.copy()
            commutator = (ct_0, ct_o1, ct_o2)
            n = 1
            while h_norm > cutoff and n < n_max:
                ct_0_tmp, ct_o1_tmp, ct_o2_tmp = self.commute(*commutator)
                fact_n_inv = 1/np.math.factorial(n) 
                h_norm = np.linalg.norm(ct_o1_tmp) + np.linalg.norm(ct_o2_tmp) + ct_0_tmp
                ct_0 += ct_0_tmp * fact_n_inv
                ct_o1 += ct_o1_tmp * fact_n_inv
                ct_o2 += ct_o2_tmp * fact_n_inv
                commutator = (ct_0_tmp, ct_o1_tmp, ct_o2_tmp)
                n += 1
            logger.info(self, "    used in total %d terms in BCH expansion", n)
            self.ct_0 = ct_0
            self.ct_o1 = ct_o1
            self.ct_o2 = ct_o2
        else:
            # Hyleraas approximation to H bar
            logger.info(self, "    using Hylleraas functional to approximate H_bar")
            ct_0, ct_o1, ct_o2 = self.commute(o1=h_mn, o2=self.eri)

            ct_o1 += h_mn
            ct_o2 += self.eri

            # The second commutator 1/2*[[F, T], T]
            # First construct F, the Fock matrix, as an approximation to H
            # Notice fock has to be in mo basis

            if self.fock is None:
                fock_mn = self.mf.get_fock()
                fock_mn = self.ao2mo(fock_mn)
            else:
                fock_mn = self.fock

            # The following [o1, t] gives rise to 1- and 2-body terms
            c0_f, c1_f, c2_f = self.commute(*self.commute(o1=fock_mn))


            # final step might need to do some transformation on the eris so that
            # other solvers can use it directly as usual integrals.
            self.ct_0 = ct_0 + c0_f/2.
            self.ct_o1 = ct_o1 + c1_f/2.
            self.ct_o2 = ct_o2 + c2_f/2.

        return self.ct_0, self.ct_o1, self.ct_o2
    
    def solve(self, method="diis", max_cycle=100, step=0.1, n_bch=None, gs_only=True):
        '''
         This function solves the generalized Brillouin condition self-consistently
        '''

        # sanity check if the H bar is constructed.
        #if self.ct_0 is None or self.ct_o1 is None or self.ct_o2 is None:
        #    raise  ValueError("CT integrals not evaluated!")
        
        # TODO: iterative 
        self.get_mp2_amps()
        self.collect_amps()
        t_init = np.zeros(self.e_nmo*self.t_nmo + self.e_nmo**2 * self.t_nmo**2)
        self.is_amp_init = True
        e_ct = 0.
        if n_bch is not None:
            self.n_bch = n_bch
        if max_cycle is not None:
            self.max_cycle = max_cycle
        self.gs_only = gs_only

        self.dump_flags()

        if method == "newton_krylov":
            ts = newton_krylov(self.get_residual, t_init, maxiter=self.max_cycle)
            self._t1s, self._t2s = self.vec_to_amps(ts)
            e_ct = self.get_e_tot()
            e_corr = self.get_e_corr()
            mo_e = self.mo_energy
            logger.info(self, "Total # residual builds = %s, E_corr(CTSD) = %.15g", self.get_res_counter, e_corr)
            logger.debug1(self, "    mo_energy = %s", mo_e)

        elif method == "diis":
            de = np.inf
            e_old = 0.
            i = 0
            mydiis = diis.DIIS()
            mydiis.space = self.diis_space 
            dt_norm = np.inf
            for i in range(self.max_cycle):
                if np.abs(de) > self.conv_tol_e and dt_norm > self.conv_tol_normt:
                    dt = self.get_residual(t_init)
                    dt_norm = np.linalg.norm(dt)
                    dt1, dt2 = self.vec_to_amps(dt)
                    mo_e = self.mo_energy
                    e_xp = -(mo_e[self.t_nmo:, None] - mo_e[None, :self.t_nmo])
                    e_xypq = lib.einsum("xi, yj -> xyij", e_xp, e_xp)
                    dt1 /= e_xp
                    dt2 /= e_xypq
                    self._t1s -= dt1 * step
                    self._t2s -= dt2 * step
                    t_init = self.amps_to_vec(self._t1s, self._t2s)
                    t_init = mydiis.update(t_init)
                    e_ct = self.get_e_tot()
                    de = e_ct - e_old
                    e_old = e_ct
                    e_corr = self.get_e_corr()
                    logger.info(self, "cycle = %s, E_corr(CTSD) = %.15g, dE = %.7e, |dt| = %.7e", i, e_corr, de, dt_norm)
                    logger.debug1(self, "    mo_energy = %s", mo_e)
        else:
            raise NotImplementedError


        logger.info(self, "E_tot(CTSD) = %s", e_ct)
        return e_ct
    
    def get_residual(self, t_ravel):
        self._t1s, self._t2s = self.vec_to_amps(t_ravel)
        self.build_hbar(bch=True, n_max=self.n_bch)
        r1 = self.get_singles_residual()
        # put active-external block to 0
        r2 = self.get_doubles_residual()
        if self.gs_only:
            r1[:, self.c_nmo:self.t_nmo] = 0.
            r2[:, :, self.c_nmo:self.t_nmo, self.c_nmo:self.t_nmo] = 0.
            r2[:, :, self.c_nmo:self.t_nmo, :self.c_nmo] = 0.
            r2[:, :, :self.c_nmo, self.c_nmo:self.t_nmo] = 0.
        r = self.amps_to_vec(r1, r2)
        e_corr = self.get_e_corr()
        dt_norm = np.linalg.norm(r)
        self.get_res_counter += 1
        v_xypq = self.ct_o2[self.t_nmo:, self.t_nmo:, :self.t_nmo, :self.t_nmo]
        v_xyab = self.ct_o2[self.t_nmo:, self.t_nmo:, self.c_nmo:self.t_nmo, self.c_nmo:self.t_nmo]
        v_xyai = self.ct_o2[self.t_nmo:, self.t_nmo:, self.c_nmo:self.t_nmo, :self.c_nmo]
        v_xyij = self.ct_o2[self.t_nmo:, self.t_nmo:, :self.c_nmo, :self.c_nmo]
        fock_xp = self.get_fock()[self.t_nmo:, :self.t_nmo]
        fock_xa = self.get_fock()[self.t_nmo:, self.c_nmo:self.t_nmo]
        fock_xi = self.get_fock()[self.t_nmo:, :self.c_nmo]
        logger.debug1(self, "# calls to res = %s, E_corr(CTSD) = %.15g, |dt| = %.7e",  self.get_res_counter, e_corr, dt_norm)
        logger.debug1(self, "    |t1| = %.15g, |t2| = %.15g", np.linalg.norm(self._t1s), np.linalg.norm(self._t2s))
        logger.debug1(self, "    |r1| = %.15g, |r2| = %.15g", np.linalg.norm(r1), np.linalg.norm(r2))
        logger.debug1(self, "    |f_xp| = %.15g, |v_xypq| = %.15g", np.linalg.norm(fock_xp), np.linalg.norm(v_xypq))
        logger.debug1(self, "    |f_xa| = %.15g, |v_xyab| = %.15g, |v_xyai| = %.15g", np.linalg.norm(fock_xa), np.linalg.norm(v_xyab), np.linalg.norm(v_xyai))
        logger.debug1(self, "    |f_xi| = %.15g, |v_xyij| = %.15g", np.linalg.norm(fock_xi), np.linalg.norm(v_xyij))

        return r
    
    def amps_to_vec(self, t1, t2):
        t_nmo = self.t_nmo
        e_nmo = self.e_nmo
        t1_size = t_nmo * e_nmo
        size = self.t_nmo * self.e_nmo  + (self.t_nmo * self.e_nmo)**2
        vec = np.zeros(size)

        vec[:t1_size] = t1.ravel().copy()
        vec[t1_size:] = t2.ravel().copy()

        return vec

    def vec_to_amps(self, vec):
        t1 = vec[:self.t_nmo*self.e_nmo].reshape([self.e_nmo, self.t_nmo]).copy()
        t2 = vec[self.t_nmo*self.e_nmo:].reshape([self.e_nmo, self.e_nmo, self.t_nmo, self.t_nmo]).copy()
        
        return t1, t2

    
    def get_singles_residual(self):

        dm1 = self.dm1
        dm2 = self.dm2
        t_nmo = self.t_nmo
        r1 = np.zeros(self._t1s.shape)
        r1 += np.einsum("nx, np -> xp", self.ct_o1[:, t_nmo:], dm1[:, :t_nmo])
        r1 -= np.einsum("np, nx -> xp", self.ct_o1[:, :t_nmo], dm1[:, t_nmo:])
        v_mnxu = self.ct_o2[:, :, self.t_nmo:, :]
        r1 += np.einsum("mnxu, mnpu -> xp", v_mnxu, dm2[:, :, :t_nmo, :])
        v_mnpu = self.ct_o2[:, :, :self.t_nmo, :]
        r1 -= np.einsum("mnpu, mnxu -> xp", v_mnpu, dm2[:, :, t_nmo:, :])
        r1 *= 2.

        return r1

    def get_doubles_residual(self):
        
        t_nmo = self.t_nmo
        nmo = self.nmo
        
        dm2 = self.dm2
        dm1 = self.dm1

        r2 = np.zeros(self._t2s.shape)
        r2_bar = np.zeros(self._t2s.shape)
        r2_bar += 2. * np.einsum("nx, nypq -> xypq", self.ct_o1[:, t_nmo:], dm2[:, t_nmo:, :t_nmo, :t_nmo])
        r2_bar -= 2. * np.einsum("np, nqxy -> xypq", self.ct_o1[:, :t_nmo], dm2[:, :t_nmo, t_nmo:, t_nmo:])
        r2 = 0.5 * r2_bar.copy() 
        r2 += 0.5 * r2_bar.transpose((1, 0, 3, 2))

        v_mnxy = self.ct_o2[:, :, t_nmo:, t_nmo:]
        v_mnpq = self.ct_o2[:, :, :t_nmo, :t_nmo]

        r2_bar = np.einsum("mnxy, mnpq -> xypq", v_mnxy, dm2[:, :, :t_nmo, :t_nmo])
        r2_bar -= np.einsum("mnpq, mnxy -> xypq", v_mnpq, dm2[:, :, t_nmo:, t_nmo:])





        v_mnxu = self.ct_o2[:, :, t_nmo:, :]
        dm2_hat = get_d_hat(self.dm1, self.dm2)
        r2_bar += 2. * contr_dm3("mnxu", "mnypuq", v_mnxu, self.dm1, dm2_hat, self.t_nmo, self.nmo)
        v_mnpu = self.ct_o2[:, :, :t_nmo, :]
        r2_bar -= 2. * contr_dm3("mnpu", "mnqxuy", v_mnpu, self.dm1, dm2_hat, self.t_nmo, self.nmo)
        # slices = [0, nmo, 0, nmo, t_nmo, nmo,
        #           0, t_nmo, 0, nmo, 0, t_nmo]
        # d3 = get_d3_slice(dm1, dm2, slices)
        # v_mnxu = self.ct_o2[:, :, t_nmo:, :]
        # r2_bar += 2. * np.einsum("mnxu, mnypuq -> xypq", v_mnxu, d3)

        # slices = [0, nmo, 0, nmo, 0, t_nmo,
        #           t_nmo, nmo,  0, nmo, t_nmo, nmo]
        # d3 = get_d3_slice(dm1, dm2, slices)
        # v_mnpu = self.ct_o2[:, :, :t_nmo, :]
        # r2_bar -= 2. * np.einsum("mnpu, mnqxuy -> xypq", v_mnpu, d3)

        r2 += r2_bar * 0.5
        r2 += r2_bar.transpose((1, 0, 3, 2)) * 0.5

        return r2
    

    def init_amps(self, t1=None, t2=None):
        """
        Get the transformaiton amplitudes.

        Args:
            algo: string. Method to compute amps.

        Returns:

        """
        # self.part_amps()
        if (t1 is None or t2 is None) and not self.is_amp_init:
            self.t1, self.t2 = self.get_amps()
            logger.info(self, "using "+self._amps_algo+" amps...")
            # assign t1 and t2 partitions to _t1s and _t2s big arrays.
            # whenever self.t1 and self.t2 are updated, _t1s and _t2s should also be
            # updated. FIXME: this is not the ideal way to do things, because it is
            # error prone and takes additional memory. (But it is convenient to implement
            # things this way.)
            self.collect_amps()
            self.is_amp_init = True
        elif t1 is not None and t2 is not None:
            logger.debug2(self, "    using user provided amplitudes...")
            self._t1s, self._t2s = t1, t2
            self.t1, self.t2 = self.part_amps()
            self.is_amp_init = True
        elif self.is_amp_init:
            logger.debug2(self, "    amplitudes already initialised by user.")
            self.part_amps()
            self.collect_amps()
            self.is_amp_init = True
        
        return self.t1, self.t2

    def get_amps(self):
        if self._amps_algo == "mp2" or self._amps_algo is None:
            self._amps_algo = 'mp2'
            return self.get_mp2_amps()
        elif self._amps_algo == "reg_mp2":
            return self.get_reg_mp2_amps()
        elif self._amps_algo == "shift_mp2":
            return self.get_shifted_mp2_amps()
        elif self._amps_algo == "zero":
            return self.get_zero_amps()
        else:
            raise NotImplementedError

    def get_zero_amps(self):
        # for test purpose
        self.t1["xi"] = np.zeros([self.e_nmo, self.c_nmo])
        self.t1["xa"] = np.zeros([self.e_nmo, self.a_nmo])

        self.t2["xyij"] = np.zeros([self.e_nmo, self.e_nmo, self.c_nmo,
                                    self.c_nmo])
        self.t2["xyab"] = np.zeros([self.e_nmo, self.e_nmo, self.a_nmo,
                                    self.a_nmo])
        self.t2["xyai"] = np.zeros([self.e_nmo, self.e_nmo, self.a_nmo,
                                    self.c_nmo])
        return self.t1, self.t2

    def get_mp2_amps(self, fock=None, eri=None, with_active=False):

        if fock is None:
            fock = self.mf.get_fock()
            fock = self.ao2mo(fock)
        
        if eri is None:
            eri = self.eri
        

        mo_e = fock.diagonal()
        mo_e_i = mo_e[:self.c_nmo]
        # if regularization is needed, one can do it here.
        mo_e_a = mo_e[self.c_nmo:self.t_nmo]
        mo_e_x = mo_e[self.t_nmo:]

        # occ to external
        e_xi = -(mo_e_x[:, None] - mo_e_i[None, :])

        self.t1["xi"] = fock[self.t_nmo:, :self.c_nmo].copy()
        self.t2["xyij"] = self.eri[self.t_nmo:, self.t_nmo:, :self.c_nmo,
                               :self.c_nmo].copy()

        self.t1["xi"] /= e_xi

        self.t2["xyij"] /= lib.direct_sum("xi+yj -> xyij", e_xi, e_xi)

        if with_active:
            e_xa = -(mo_e_x[:, None] - mo_e_a[None, :])
            self.t1["xa"] = fock[self.t_nmo:, self.c_nmo:self.t_nmo].copy()

            self.t2["xyab"] = eri[self.t_nmo:, self.t_nmo:,
                            self.c_nmo:self.t_nmo:, self.c_nmo:self.t_nmo].copy()
            self.t2["xyai"] = eri[self.t_nmo:, self.t_nmo:,
                              self.c_nmo:self.t_nmo, :self.c_nmo].copy()
            self.t1["xa"] /= e_xa
            self.t2["xyab"] /= lib.direct_sum("xa+yb -> xyab", e_xa, e_xa)
            self.t2["xyai"] /= lib.direct_sum("xa+yi -> xyai", e_xa, e_xi)

        return self.t1, self.t2

    def get_shifted_mp2_amps(self, epsilon=None):
        logger.info(self, "using shifted mp2 amps.")
        logger.info(self, "epsilon = ", epsilon)

        if self.fock is None:
            fock_mn = self.mf.get_fock()
            fock_mn = self.ao2mo(fock_mn)
        else:
            fock_mn = self.fock
        mo_energy = fock_mn.diagonal()
        if epsilon is None:
            epsilon = mo_energy[self.nocc]+0.1

        mo_e_i = mo_energy[:self.c_nmo]
        # if regularization is needed, one can do it here.
        mo_e_a = mo_energy[self.c_nmo:self.t_nmo]
        mo_e_x = mo_energy[self.t_nmo:]

        e_xi = -(mo_e_x[:, None] - mo_e_i[None, :])
        #e_xa = -(mo_e_x[:, None] - epsilon)
        shift = -epsilon * np.ones(self.a_nmo)
        e_xa = -(mo_e_x[:, None] - shift[None, :])

        self.t1["xi"] = fock_mn[self.t_nmo:, :self.c_nmo].copy()
        self.t1["xa"] = fock_mn[self.t_nmo:, self.c_nmo:self.t_nmo].copy()

        self.t2["xyij"] = self.eri[self.t_nmo:, self.t_nmo:, :self.c_nmo,
                               :self.c_nmo].copy()
        self.t2["xyab"] = self.eri[self.t_nmo:, self.t_nmo:,
                        self.c_nmo:self.t_nmo:, self.c_nmo:self.t_nmo].copy()
        self.t2["xyai"] = self.eri[self.t_nmo:, self.t_nmo:,
                          self.c_nmo:self.t_nmo, :self.c_nmo].copy()


        self.t1["xi"] /= e_xi
        self.t1["xa"] /= e_xa
        delta_xyab = lib.direct_sum("xa+yb -> xyab", e_xa, e_xa)
        delta_xyij = lib.direct_sum("xi+yj -> xyij", e_xi, e_xi)
        delta_xyai = lib.direct_sum("xa+yi -> xyai", e_xa, e_xi)

        self.t2["xyab"] *= 1. / delta_xyab
        self.t2["xyij"] *= 1. / delta_xyij
        self.t2["xyai"] *= 1. / delta_xyai

        return self.t1, self.t2

    def get_reg_mp2_amps(self):

        if self.fock is None:
            fock_mn = self.mf.get_fock()
            fock_mn = self.ao2mo(fock_mn)
        else:
            fock_mn = self.fock
        mo_energy = fock_mn.diagonal()

        mo_e_i = mo_energy[:self.c_nmo]
        # if regularization is needed, one can do it here.
        mo_e_a = mo_energy[self.c_nmo:self.t_nmo]
        mo_e_x = mo_energy[self.t_nmo:]

        e_xi = -(mo_e_x[:, None] - mo_e_i[None, :])
        e_xa = -(mo_e_x[:, None] - mo_e_a[None, :])

        self.t1["xi"] = fock_mn[self.t_nmo:, :self.c_nmo].copy()
        self.t1["xa"] = fock_mn[self.t_nmo:, self.c_nmo:self.t_nmo].copy()

        self.t2["xyab"] = self.eri[self.t_nmo:, self.t_nmo:,
                        self.c_nmo:self.t_nmo:, self.c_nmo:self.t_nmo].copy()
        self.t2["xyij"] = self.eri[self.t_nmo:, self.t_nmo:, :self.c_nmo,
                               :self.c_nmo].copy()
        self.t2["xyai"] = self.eri[self.t_nmo:, self.t_nmo:,
                          self.c_nmo:self.t_nmo, :self.c_nmo].copy()

        self.t1["xi"] /= e_xi
        self.t1["xa"] /= e_xa
        delta_xyab = lib.direct_sum("xa+yb -> xyab", e_xa, e_xa)
        delta_xyij = lib.direct_sum("xi+yj -> xyij", e_xi, e_xi)
        delta_xyai = lib.direct_sum("xa+yi -> xyai", e_xa, e_xi)
        sigma = 0.1

        self.t2["xyab"] *= 1. / delta_xyab * (1. - np.exp(-sigma * delta_xyab**2)) 
        self.t2["xyij"] *= 1. / delta_xyij * (1. - np.exp(-sigma * delta_xyij**2)) 
        self.t2["xyai"] *= 1. / delta_xyai * (1. - np.exp(-sigma * delta_xyai**2)) 

        return self.t1, self.t2

    def get_f12_amps(self):
        raise NotImplementedError

    def commute(self, o0=0., o1=None, o2=None):
        c1 = None
        c2 = None
        c0 = o0

        if o1 is not None:
            c1, c2 = self.commute_o1_t(o1)
        if o2 is not None:
            c0, c1_prime, c2_prime, c2_dprime = self.commute_o2_t(o2)
            c1 += c1_prime
            #c2 += c2_prime 
            c2 += c2_prime + c2_dprime
        
        if c1 is not None:
            c1 = symmetrize(c1)
        if c2 is not None:
            c2 = symmetrize(c2)

        return c0, c1, c2

    def commute_o1_t(self, o1):
        """
        This function assembles the two parts of the commutator
        [o1, T]=[o1, t1]+[o1, t2] together.

        Args:
            o1_mn: 2D array of size [nmo, nmo]. The one-body integral to be
            transformed

        Returns:
            ct_o1: np ndarray, [nmo, nmo]
            ct_o2: np ndarray, [nmo, nmo, nmo, nmo]
        """
        # Note get_c1 already symmetrize the transformed integral.
        ct_o1 = self.get_c1(o1)

        ct_o2 = self.get_c2(o1)

        return ct_o1, ct_o2
    
    def commute_o2_t(self, o2):
        """
        This function assembles the two parts of the commutator
        [o2, T]=[o2, t1]+[o1, t2] together.

        Args:
            o2: 4D array of size [nmo, nmo, nmo, nmo]. The two-body integral to be
            transformed

        Returns:
            ct_o1: np ndarray, [nmo, nmo]
            ct_o2: np ndarray, [nmo, nmo, nmo, nmo]
        """
        c2_prime = self.get_c2_prime(o2)

        c0 = self.get_c0(o2)
        c1_prime = self.get_c1_prime(o2)
        c2_dprime = self.get_c2_dprime(o2)


        return c0, c1_prime, c2_prime, c2_dprime

    def get_c1(self, o1=None, t1=None):
        """
        This function calculates the commutator between o1_mn and O_px.

        Args:
            o_mn: a 1-body (2 indices) tensor defined on the parent space

        Returns:
            c_mn: transformed integral, defined on the parent space
        """
        if o1 is None:
            o1 = self.mf.get_hcore()
            o1 = self.ao2mo(o1)

        t1 = self._t1s
        o1_mx = o1[:, self.t_nmo:]
        o1_mp = o1[:, :self.t_nmo]
        # equ (35) in Ref: Phys. Chem. Chem. Phys., 2012, 14, 7809–7820
        c1_mn = np.zeros([self.nmo, self.nmo])
        # only the following terms are relevant to the target space
        c1_mn[:, :self.t_nmo] = 2. * lib.einsum("mx, xp -> mp", o1_mx, t1)

        # connections between target and external space
        c1_mn[:, self.t_nmo:] -= 2. * lib.einsum("mp, xp -> mx", o1_mp, t1)

        c1_mn = symmetrize(c1_mn)

        return c1_mn

    def get_c2(self, o_mn=None, t2=None):
        """
        This function computes the [\hat{h}_1, \hat{T}_2]_{1,2}, equ (37) 
        and (38) in Ref: Phys. Chem. Chem. Phys., 2012, 14, 7809–7820

        Args:
            o_mn: Rank 2 tensor of size [nao, nao], defined on the parent
                  basis set

        Returns:
            ct_eris: Transformed rank 4 tensor defined on the parent basis set.
        """
        #o_xa = o_mn[self.t_nmo:, self.c_nmo:self.t_nmo]
        #o_xi = o_mn[self.t_nmo:, :self.c_nmo]
        if t2 is None:
            t2 = self._t2s
        if o_mn is None:
            o_mn = self.mf.get_fock()
            o_mn = self.ao2mo(o_mn)
        o_mx = o_mn[:, self.t_nmo:]
        o_mp = o_mn[:, :self.t_nmo]
        c2 = np.zeros(self.eri.shape)
        c2[:, self.t_nmo:, :self.t_nmo, :self.t_nmo] = 4.*lib.einsum(
            "mx, xypq -> mypq", o_mx, t2
            )
        c2[:, :self.t_nmo, self.t_nmo:, self.t_nmo:] += -4.*lib.einsum(
            "mp, xypq -> mqxy", o_mp, t2
        )

        c2 = symmetrize(c2)

        return c2

    def get_c2_prime(self, o2=None):
        """
        This function computes the CT contirubtion [o2, t1].
        Only contribution to the target space is implemented. Connections
        between target and external space, or that within external space
        are currently not implemented.

        Args:
            o2: rank 4 tensor (ndarray). Specify which 2-body operator to
                commute with the T1 operator. If not supplied, the default is
                the Coulomb eri.
        Returns:
            c2_prime: rank 4 tensor. 
        
        """
        #TODO: generalize it to the who space. When singles amp is 0, c2_prime
        # is also 0. Not used for now.
        if o2 is None:
            o2 = self.eri
        t1 = self._t1s
        c2_prime = np.zeros(o2.shape)
        c2_prime[:, :, :self.t_nmo, :] = 4. * np.einsum("mnxu, xp -> mnpu", o2[:, :, self.t_nmo:, :], t1)
        c2_prime[:, :, self.t_nmo:,:] -= 4. * np.einsum("mnpu, xp -> mnxu", o2[:, :, :self.t_nmo, :], t1)
        c2_prime = symmetrize(c2_prime)
        return c2_prime

    def get_c0(self, o2=None, dm1=None, dm2=None):
        """
        Calculate the constant contribution to energy from CT.
        Args:
            o2: 4D ndarray. 2-body operator to be transformed.
        Returns:
            c0: float.

        """
        if o2 is None:
            o2 = self.eri

        if dm1 is None:
            dm1 = self.dm1
        if dm2 is None:
            dm2 = self.dm2

        # construct Do^{p1p2e2}_{a1q2a2}
        #dm2 = get_d_zero(dm1, dm2)


        #v_mnxu = o2[:, :, self.t_nmo:, :]
        #t2 = self._t2s

        # for single reference case
        c0 = 0.
        #c0 = 2.0 * contr_dm3("mnxu", "xypq, mnypuq ->", v_mnxu, t2,
        #                      d3)

        # need a slice of the full ERI
        #v_mnxu = o2[:, :, self.t_nmo:, :]
        #slices = [0, self.nmo, 0, self.nmo, self.t_nmo, self.nmo,
        #          0, self.t_nmo, 0, self.nmo, 0, self.t_nmo]
        #d3 = get_d3_slice(dm1, dm2, slices)
        #t2 = self._t2s
        #c0 = 2.0 * lib.einsum("mnxu, xypq, mnypuq ->", v_mnxu, t2,
        #                      d3)
        #v_mnpu = o2[:, :, :self.t_nmo, :]
        #slices = [0, self.nmo, 0, self.nmo, 0, self.t_nmo,
        #          self.t_nmo, self.nmo, 0, self.nmo, self.t_nmo,
        #          self.nmo]
        #d3 = get_d3_slice(dm1, dm2, slices)
        #c0 -= 2.0 * lib.einsum("mnpu, xyqr, mnrxuy ->", v_mnpu, t2,
        #                       d3)

        return c0

    def get_c1_prime(self, o2=None, dm2_bar=None):
        """
        Function to compute the 2D tensor associated with the 1-body operator
        generated by [h2, T2]

        Args:
            o2: 4D ndarray. Two-body operator to be commuted with t1.
            dm2_bar: 4D tensor. As defined in equation (19) in Ref:
            Phys. Chem. Chem. Phys., 2012, 14, 7809–7820

        Returns:
            c1_prime_mn: 2D tensor. The 1-body contribution from [h2, T2]
        """

        if o2 is None:
            o2 = self.eri

        if dm2_bar is None:
            dm2_bar = get_d_bar(self.dm1, self.dm2)
        # need the original t2
        t2 = self._t2s
        # second term in 46b
        dm2_bar_xipj = dm2_bar[self.t_nmo:, :self.c_nmo, :self.t_nmo,
                       :self.c_nmo]
        s1_xipj = -1. / 2 * lib.einsum("xypq, xipj -> yiqj", t2,
                                       dm2_bar_xipj)
        dm2_bar_xijp = dm2_bar[self.t_nmo:, :self.c_nmo,
                       :self.c_nmo, :self.t_nmo]
        # first term in 46b
        s1_xipj -= 1. / 2 * lib.einsum("xyqp, xijp -> yiqj", t2,
                                       dm2_bar_xijp)

        # constructing s2
        dm2_bar_iypq = dm2_bar[:self.c_nmo, self.t_nmo:,
                       :self.t_nmo, :self.t_nmo]
        s2_xi = lib.einsum("xypq, iypq -> xi", t2, dm2_bar_iypq)

        # constructing s3
        dm2_bar_ipxy = dm2_bar[:self.c_nmo, :self.t_nmo,
                       self.t_nmo:, self.t_nmo:]
        s3_ip = lib.einsum("xypq, iqxy -> ip", t2, dm2_bar_ipxy)

        # constructing s4
        dm2_bar_ijpq = dm2_bar[:self.c_nmo, :self.c_nmo,
                       :self.t_nmo, :self.t_nmo]
        s4_xyij = 1. / 2 * lib.einsum("xypq, ijpq -> xyij", t2, dm2_bar_ijpq)

        # constructing s5
        dm2_bar_ijxy = dm2_bar[:self.c_nmo, :self.c_nmo,
                       self.t_nmo:, self.t_nmo:]
        s5_ijpq = 1. / 2 * lib.einsum("xypq, ijxy -> ijpq", t2, dm2_bar_ijxy)

        # s6
        # TODO: This contraction is very heavy. Need optimization!!
        s6_mn = lib.einsum("mnuv, pnuv -> pm", o2, dm2_bar)
        s6_px = s6_mn[:self.t_nmo, self.t_nmo:]
        s6_xp = s6_mn[self.t_nmo:, :self.t_nmo]

        # s0
        s0_xipj = lib.einsum("xypq, xipj -> yiqj", t2, dm2_bar_xipj)
        s0_xipj -= 1./2 * lib.einsum("xyqp, xipj -> yiqj", t2, dm2_bar_xipj)

        # constructing c1_prime
        c1_prime_mn = np.zeros([self.nmo, self.nmo])
        # adding contributions from e1_prime_pq
        # first term in equation 44
        v_mixj = o2[:, :self.c_nmo, self.t_nmo:,
                    :self.c_nmo]
        v_mijx = o2[:, :self.c_nmo, :self.c_nmo,
                    self.t_nmo:]
        c1_prime_mn[:, :self.t_nmo] -= lib.einsum("mixj, xipj -> mp", v_mixj,
                                                  s0_xipj)
        c1_prime_mn[:, :self.t_nmo] -= lib.einsum("mijx, xipj -> mp", v_mijx,
                                                  s1_xipj)

        # second term in equation 44
        v_minx = o2[:, :self.c_nmo, :, self.t_nmo:].copy()
        # why is there a transpose here???
        v_mixn = o2[:, :self.c_nmo, self.t_nmo:, :].copy()
        c1_prime_mn -= lib.einsum("minx, xi -> mn", v_minx, s2_xi)
        c1_prime_mn += 1. / 2 * lib.einsum("mixn, xi -> mn", v_mixn, s2_xi)

        # third term in equation 44
        v_ijxm = o2[:self.c_nmo, :self.c_nmo,
                    self.t_nmo:, :]
        c1_prime_mn[:, self.t_nmo:] += lib.einsum("ijxm, xyij -> my",
                                                  v_ijxm, s4_xyij)

        # fourth term in equation 44
        c1_prime_mn[:self.t_nmo, self.t_nmo:] -= \
            lib.einsum("xypq, qy -> px", t2, s6_px)
        
        c1_prime_mn[:self.t_nmo, self.t_nmo:] += \
            0.5 * lib.einsum("xyqp, qy -> px", t2, s6_px)

        # adding contributions from a1_prime_pq, note that the overall sign is -
        # first term in equation 45
        v_mipj = o2[:, :self.c_nmo, :self.t_nmo, :self.c_nmo]
        v_mijp = o2[:, :self.c_nmo, :self.c_nmo, :self.t_nmo]
        c1_prime_mn[:, self.t_nmo:] += lib.einsum("mipj, xipj -> mx",
                                                  v_mipj, s0_xipj)
        c1_prime_mn[:, self.t_nmo:] += lib.einsum("mijp, xipj -> mx",
                                                  v_mijp, s1_xipj)

        # second term in equation 45
        v_minp = o2[:, :self.c_nmo, :, :self.t_nmo].copy()
        v_mipn = o2[:, :self.c_nmo, :self.t_nmo, :].copy()
        c1_prime_mn += lib.einsum("minp, ip -> mn", v_minp, s3_ip)
        c1_prime_mn -= 1./2 * lib.einsum("mipn, ip -> mn", v_mipn, s3_ip)

        # third term in equation 45
        v_ijpm = o2[:self.c_nmo, :self.c_nmo, :self.t_nmo, :]
        c1_prime_mn[:, :self.t_nmo] -= lib.einsum("ijpm, ijpq -> mq",
                                                  v_ijpm, s5_ijpq)

        # fourth term in equation 45
        c1_prime_mn[:self.t_nmo, self.t_nmo:] += \
            lib.einsum("xypq, yq -> px", t2, s6_xp)
        c1_prime_mn[:self.t_nmo, self.t_nmo:] -= \
            0.5 * lib.einsum("xyqp, yq -> px", t2, s6_xp)

        c1_prime_mn *= 2.

        c1_prime_mn = symmetrize(c1_prime_mn)

        return c1_prime_mn
    
    def get_c2_dprime(self, o2=None):
        c2_dprime = self.get_c2_dprime_generic(o2)
        c2_dprime += self.get_c2_dprime_eno(o2)
        return c2_dprime
    
    def get_c2_dprime_generic(self, o2=None):
        if o2 is None:
            o2 = self.eri
        t2 = self._t2s
        c2_dprime_generic = np.zeros([self.nmo, self.nmo, self.nmo,
                                   self.nmo])

        # adding e_dprime_mnuv contribution
        # TODO: the on-the-fly slicing is not the most efficient way, fix it
        # first term in equation 48
        v_mnxy = o2[:, :, self.t_nmo:, self.t_nmo:]
        c2_dprime_generic[:, :, :self.t_nmo, :self.t_nmo] += \
            1. / 2 * lib.einsum("mnxy, xypq -> mnpq", v_mnxy, t2)
        
        # adding contributions from a_dprime_mnuv, notice the overall - sign
        v_mnpq = o2[:, :, :self.t_nmo, :self.t_nmo]
        c2_dprime_generic[:, :, self.t_nmo:, self.t_nmo:] -= \
            1. / 2 * lib.einsum("mnpq, xypq -> mnxy", v_mnpq, t2)
        
        c2_dprime_generic *= 4
        c2_dprime_generic = symmetrize(c2_dprime_generic)
        return c2_dprime_generic
        

    def get_c2_dprime_eno(self, o2=None, dm1=None):
        """
        Function to construct c_double_prime_mnuv, Eq (47)
        Args:
            o2: 4D ndarray. 2-body operator 
            dm1: 2D tensor. 1-RDM

        Returns:
            c2_dprime_mnuv: 4D tensor.
        """
        if o2 is None:
            o2 = self.eri
        
        if dm1 is None:
            dm1 = self.dm1

        # Construct T0-T3 intermediates first.
        t2 = self._t2s
        cap_t0_xyip = lib.einsum("xypq, ip -> xyiq", t2, dm1[:self.c_nmo,
                                                         :self.t_nmo])
        cap_t1_ixpq = lib.einsum("xypq, ix -> iypq", t2, dm1[:self.c_nmo,
                                                         self.t_nmo:])
        cap_t2_xp = lib.einsum("xypq, xp -> yq", t2, dm1[self.t_nmo:,
                                                     :self.t_nmo])
        cap_t2_xp -= 1. / 2 * lib.einsum("xyqp, xp -> yq", t2,
                                         dm1[self.t_nmo:,
                                         :self.t_nmo])
        cap_t3_mn = lib.einsum("mnuv, nv -> mu", o2, dm1)
        cap_t3_mn -= 1. / 2 * lib.einsum("mnvu, nv -> mu", o2, dm1)

        c2_dprime_mnuv = np.zeros([self.nmo, self.nmo, self.nmo,
                                   self.nmo])

        # adding e_dprime_mnuv contribution
        # TODO: the on-the-fly slicing is not the most efficient way, fix it
        # first term in equation 48
        
        # second term in equ 48
        # term 2 in note e"
        v_mxni = o2[:, self.t_nmo:, :, :self.c_nmo]
        c2_dprime_mnuv[:, :self.t_nmo, :, self.t_nmo:] += \
            lib.einsum("mxni, xyip -> mpny", v_mxni, cap_t0_xyip)


        # term 6 in note e"
        c2_dprime_mnuv[:, :self.t_nmo, :, self.t_nmo:] -= \
            1. / 2 * lib.einsum("mxni, yxip -> mpny", v_mxni, cap_t0_xyip)


        # term 5 in note e"
        v_mxin = o2[:, self.t_nmo:, :self.c_nmo, :]
        c2_dprime_mnuv[:, :self.t_nmo, :, self.t_nmo:] -= \
            1. / 2 * lib.einsum("mxin, xyip -> mpny", v_mxin, cap_t0_xyip)


        # third term in equ 48
        # term 4 in note e"
        c2_dprime_mnuv[:, :self.t_nmo, self.t_nmo:, :] -= \
            1. / 2 * lib.einsum("mxin, yxip -> mpyn", v_mxin, cap_t0_xyip)


        # fourth term in equ 48
        v_mnxi = o2[:, :, self.t_nmo:, :self.c_nmo]
        c2_dprime_mnuv[:, :, :self.t_nmo, :self.t_nmo] -= \
            1. / 2 * lib.einsum("mnxi, ixpq -> mnqp", v_mnxi, cap_t1_ixpq)
        

        # fifth term in equ 48
        v_mnxu = o2[:, :, self.t_nmo:, :]
        c2_dprime_mnuv[:, :, :self.t_nmo, :] += \
            lib.einsum("mnxu, xp -> mnpu", v_mnxu, cap_t2_xp)

        # sixth term in equ 48
        # term 1 + 3 in note e"
        c2_dprime_mnuv[:, self.t_nmo:, :self.t_nmo, :self.t_nmo] += \
            lib.einsum("xypq, mx -> mypq", t2,
                        cap_t3_mn[:, self.t_nmo:])


        # adding contributions from a_dprime_mnuv, notice the overall - sign
        # second term in equ 49
        v_mpni = o2[:, :self.t_nmo, :, :self.c_nmo]
        c2_dprime_mnuv[:, self.t_nmo:, :, :self.t_nmo] -= \
            lib.einsum("mpni, ixpq -> mxnq", v_mpni, cap_t1_ixpq)
        c2_dprime_mnuv[:, self.t_nmo:, :, :self.t_nmo] += \
            1. / 2 * lib.einsum("mpni, ixpq -> mxnq", v_mpni, cap_t1_ixpq)
        v_mpin = o2[:, :self.t_nmo, :self.c_nmo, :]
        c2_dprime_mnuv[:, self.t_nmo:, :, :self.t_nmo] += \
            1. / 2 * lib.einsum("mpin, ixpq -> mxnq", v_mpin, cap_t1_ixpq)

        # third term in equ 49
        c2_dprime_mnuv[:, self.t_nmo:, :self.t_nmo, :] += \
            1. / 2 * lib.einsum("mpin, ixqp -> mxqn", v_mpin, cap_t1_ixpq)


        # fourth term in equ 49
        v_mnpi = o2[:, :, :self.t_nmo, :self.c_nmo]
        c2_dprime_mnuv[:, :, self.t_nmo:, self.t_nmo:] += \
            1. / 2 * lib.einsum("mnpi, xyip -> mnyx", v_mnpi, cap_t0_xyip)
        
        # fifth term in equ 49
        v_mnpu = o2[:, :, :self.t_nmo, :]
        c2_dprime_mnuv[:, :, self.t_nmo:, :] -= \
            lib.einsum("mnpu, xp -> mnxu", v_mnpu, cap_t2_xp)

        # sixth term in equ 49
        c2_dprime_mnuv[:, :self.t_nmo, self.t_nmo:,
        self.t_nmo:] -= lib.einsum("xypq, mp -> mqxy", t2,
                                   cap_t3_mn[:, :self.t_nmo])

        c2_dprime_mnuv *= 4.

        c2_dprime_mnuv = symmetrize(c2_dprime_mnuv)
        return c2_dprime_mnuv

    def get_c2_dprime_sr(self, o2=None):
        """
        Function to construct c_double_prime_mnuv in
        single reference case.
        Normal ordered with respect to HF Det.
        Equations derived by hand.
        This function is for test purpose.
        Args:
            o2: 4D ndarray. 2-body operator 

        Returns:
            c2_dprime_mnuv: 4D tensor.
        """
        if o2 is None:
            o2 = self.eri
        
        t2 = self._t2s
        c2_dprime = np.zeros(o2.shape)

        # 2-body terms resulting from single contractions in 3-body operator
        # -4A^{ab}_{ij}g^{ip}_{qr}E^{qjr}_{abp}

        # term 1 from note.
        # term 1 in note a"
        v_ijpk = o2[:self.c_nmo, :self.c_nmo, :, :self.c_nmo]
        c2_dprime[:, :self.c_nmo, self.c_nmo:, self.c_nmo:] -= \
            2. * lib.einsum("ikqk, abij -> qjab", v_ijpk, t2)
        

        # term 3 in a"
        v_ijpq = o2[:self.c_nmo, :self.c_nmo, :, :]
        c2_dprime[:, :, self.c_nmo:, self.c_nmo:] += \
            lib.einsum("ijqr, abij -> qrab", v_ijpq, t2)


        # term 2 in note a"
        v_ijkp = o2[:self.c_nmo, :self.c_nmo, :self.c_nmo, :]
        c2_dprime[:, :self.c_nmo, self.c_nmo:, self.c_nmo:] += \
            lib.einsum("ikkr, abij -> rjab", v_ijkp, t2)

        
        # term 3 from note
        # 4 A^{ab}_{ij}g^{ap}_{qr}E^{ijp}_{qbr}
        # term 3 in e"
        v_aijp = o2[self.c_nmo:, :self.c_nmo, :self.c_nmo, :]
        c2_dprime[:self.c_nmo, :self.c_nmo, :, self.c_nmo:] -= \
            lib.einsum("akkr, abij -> ijrb", v_aijp, t2)

        # term 6 in note e"
        v_apiq = o2[self.c_nmo:, :, :self.c_nmo, :]
        c2_dprime[:self.c_nmo, :, self.c_nmo:, :] -= \
            lib.einsum("apjr, abij -> ipbr", v_apiq, t2)


        # term 2 in note e"
        v_apiq = o2[self.c_nmo:, :, :self.c_nmo, :]
        c2_dprime[:self.c_nmo, :, self.c_nmo:, :] += \
            2. * lib.einsum("apir, abij -> jpbr", v_apiq, t2)
        
        # term 1 in note e"
        v_aipj = o2[self.c_nmo:, :self.c_nmo, :, :self.c_nmo]
        c2_dprime[:self.c_nmo, :self.c_nmo, :, self.c_nmo:] += \
            2. * lib.einsum("akqk, abij -> ijqb", v_aipj, t2)

        # term 4 in note e"
        v_apqi = o2[self.c_nmo:, :, :, :self.c_nmo]
        c2_dprime[:self.c_nmo, :, :, self.c_nmo:] -= \
            lib.einsum("apqj, abij -> ipqb", v_apqi, t2)

        # term 5 in note e"
        c2_dprime[:, :self.c_nmo, :, self.c_nmo:] -= \
            lib.einsum("apqi, abij -> pjqb", v_apqi, t2)
        
        
        c2_dprime *= 4.

        c2_dprime = symmetrize(c2_dprime)

        return c2_dprime
    
    def get_c1_prime_sr(self, o2=None):
        """
        Function to compute the 2D tensor associated with the 1-body operator
        generated by [h2, T2] in single reference case

        Args:
            o2: 4D ndarray. Two-body operator to be commuted with t1.

        Returns:
            c1_prime_mn: 2D tensor. The 1-body contribution from [h2, T2]
        """

        if o2 is None:
            o2 = self.eri

        # need the original t2
        t2 = self._t2s

        c1_prime = np.zeros([self.nmo, self.nmo])
        o2_aijk = o2[self.c_nmo:, :self.c_nmo, :self.c_nmo, :self.c_nmo]
        o2_apij = o2[self.c_nmo:, :, :self.c_nmo, :self.c_nmo]

        c1_prime[:, self.c_nmo:] -= 2.*lib.einsum("abij, apij -> pb", t2, o2_apij)
        
        c1_prime[:, self.c_nmo:] += lib.einsum("abij, apji -> pb", t2, o2_apij)

        c1_prime[:self.c_nmo, self.c_nmo:] += 4.*lib.einsum("abij, akik -> jb", t2, o2_aijk)

        #c1_prime[:self.c_nmo, self.c_nmo:] -= lib.einsum("abij, akki -> jb", t2, o2_aijk)
        c1_prime[:self.c_nmo, self.c_nmo:] -= 2. * lib.einsum("abij, akki -> jb", t2, o2_aijk)

        c1_prime[:self.c_nmo, self.c_nmo:] -= 2. * lib.einsum("abij, akjk -> ib", t2, o2_aijk)
        #c1_prime[:self.c_nmo, self.c_nmo:] -= lib.einsum("abij, akjk -> ib", t2, o2_aijk)

        c1_prime[:self.c_nmo, self.c_nmo:] += lib.einsum("abij, akkj -> ib", t2, o2_aijk)


        c1_prime *= 2.
        c1_prime = symmetrize(c1_prime)
        return c1_prime

    def get_e_tot(self, c0=None, c1=None, c2=None):
        if c0 is None:
            c0 = self.ct_0
        if c1 is None:
            c1 = self.ct_o1
        if c2 is None:
            c2 = self.ct_o2

        e_hf = 2. * lib.einsum("ii -> ", c1[:self.c_nmo, :self.c_nmo])
        e_hf += 2. * lib.einsum("ijij -> ", c2[:self.c_nmo, :self.c_nmo,
                                             :self.c_nmo, :self.c_nmo])
        e_hf -= lib.einsum("ijji -> ", c2[:self.c_nmo, :self.c_nmo,
                                            :self.c_nmo, :self.c_nmo])
        e_hf += c0
        e_hf += self.mf.energy_nuc()
        return e_hf
    
    def get_e_corr(self):
        return self.get_e_tot() - self.mf.e_tot
        
    def get_fock(self, c1=None, c2=None):
        if c1 is None:
            c1 = self.ct_o1
        if c2 is None:
            c2 = self.ct_o2
        fock = c1.copy()
        fock += 2.*lib.einsum("piqi -> pq", c2[:, :self.nocc, :, :self.nocc])
        fock -= lib.einsum("piiq -> pq", c2[:, :self.nocc, :self.nocc, :])
        return fock

    @property
    def mo_energy(self):
        if self._mo_energy is None:
            mo_energy = self.get_mo_energy()
            self._mo_energy = mo_energy.copy()
        return self._mo_energy

    @mo_energy.setter
    def mo_energy(self, value):
        self._mo_energy = value

    def get_mo_energy(self):
        mo_energy = self.ct_o1.copy()
        mo_energy += 2 * lib.einsum("piqi -> pq", self.ct_o2[:, :self.c_nmo,
                                                  :, :self.c_nmo])
        mo_energy -= lib.einsum("piiq -> pq", self.ct_o2[:, :self.c_nmo,
                                              :self.c_nmo, :])
        mo_energy = mo_energy.diagonal()
        return mo_energy

    def canonicalize(self):
        """
        This function iteratively diagnolize the CT-Fock matrix until
        self-consistency is reached. In other words, it reoptimize the
        orbitals in the presence of the CT integrals. 

        Returns:
            ct_mo_coeff: matrix [nmo, nmo]. Transformation matrix for integrals.
        """
        logger.info(self, "ct mo_energy = %s", self.mo_energy)
        mol = gto.M()
        mol.verbose = 7
        mol.nelectron = self.nocc * 2
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: self.ct_o1
        mf.get_ovlp = lambda *args: np.eye(self.nmo)
        mf.energy_nuc = lambda *args: self.mf.energy_nuc()

        mf._eri = self.ct_o2.transpose((0, 2, 1, 3))
        dm0 = np.diag(self.mf.mo_occ)
        mf.kernel(dm0=dm0)
        logger.info(self, "mo_energy = %s", mf.mo_energy)
        mol.incore_anyway = True
        self.ct_o1 = self.ao2mo(self.ct_o1, mf.mo_coeff)
        self.ct_o2 = self.ao2mo(self.ct_o2, mf.mo_coeff, to_phy=False)

        return mf.mo_coeff


    @property
    def amps_algo(self):
        return self._amps_algo

    @amps_algo.setter
    def amps_algo(self, algo):
        if not isinstance(algo, str):
            raise RuntimeError("algo name has to be the following strings: "
                               "mp2, zero, f12")
        self._amps_algo = algo

    def ao2mo(self, t=None, mo_coeff=None, to_phy=True):
        """
        Args:
            t: 2D or 4D tensor in ao rep.
            mo_coeff: 2D matrix. [nao, nmo]

        Returns:
            t_mo: 2D or 4D tensor in mo rep, the same size as the input t.

        """
        if mo_coeff is None:
            mo_coeff = self.mf.mo_coeff
        if t is None:
            t = self.mf._eri
            t = ao2mo.restore(1, t, self.nmo)
        if len(t.shape) == 4:
            self.eri = ao2mo.incore.full(t, mo_coeff)
            # use no symmetries for initial implementation
            # transpose to Physicsts notation
            # self.eri = ao2mo.restore(1, self.eri, self.nmo)
            if to_phy:
                return self.eri.transpose((0, 2, 1, 3))
            else:
                return self.eri
        elif len(t.shape) == 2:
            t_mo = reduce(np.dot, (mo_coeff.conj().T, t, mo_coeff))
            return t_mo
        else:
            raise NotImplementedError

    @property
    def nocc(self):
        if self._nocc is None:
            self._nocc = np.count_nonzero(self.mf.mo_occ)
        return self._nocc
    @nocc.setter
    def nocc(self, nocc):
        self._nocc = nocc

    @property
    def t2s(self):
        # get _t2s
        return self._t2s

    @t2s.setter
    def t2s(self, t2):
        """
        Args:
            t2: 4D tensor. [nmo, nmo, nmo, nmo]
        """
        self._t2s = t2
        self.part_t2s()

    @property
    def t1s(self):
        return self._t1s

    @t1s.setter
    def t1s(self, t1):
        """
        Args:
            t1: 2D tensor. [nmo, nmo]
        """
        self._t1s = t1
        self.part_t1s()

    def part_amps(self):
        self.part_t1s()
        self.part_t2s()
        return self.t1, self.t2

    def part_t2s(self):
        t_xyij = self._t2s[:, :, :self.c_nmo, :self.c_nmo]
        t_xyab = self._t2s[:, :, self.c_nmo:, self.c_nmo:]
        t_xyai = self._t2s[:, :, self.c_nmo:, :self.c_nmo]
        self.t2 = {"xyij": t_xyij, "xyab": t_xyab, "xyai": t_xyai}

    def part_t1s(self):
        t_xi = self._t1s[:, :self.c_nmo]
        t_xa = self._t1s[:, self.c_nmo:]
        self.t1 = {"xi": t_xi, "xa": t_xa}

    def collect_amps(self):
        self.collect_t1s()
        self.collect_t2s()
        return self.t1s, self.t2s

    def collect_t2s(self):
        self._t2s[:, :, :self.c_nmo, :self.c_nmo] = self.t2["xyij"] 
        self._t2s[:, :, self.c_nmo:, self.c_nmo:] = self.t2["xyab"]
        self._t2s[:, :, self.c_nmo:, :self.c_nmo] = self.t2["xyai"]
        self._t2s[:, :, :self.c_nmo:, self.c_nmo:] = self.t2["xyai"].transpose((0,1,3,2))

    def collect_t1s(self):
        self._t1s[:, :self.c_nmo] = self.t1["xi"]
        self._t1s[:, self.c_nmo:] = self.t1["xa"]


    
    def create_eris(self, fock=None, eris=None, c2=None):
        if eris is None:
            eris = _ChemistsERIs()
        if c2 is None:
            c2 = self.ct_o2
        nocc = self.nocc
        eris.nocc = nocc
        eris.mol = self.mol
        nvir = self.nmo - nocc
        if fock is None:
            eris.fock = self.get_fock()
        else:
            eris.fock = fock
        eris.e_hf = self.get_e_tot()
        if fock is None:
            eris.mo_energy = self.mo_energy
        else:
            eris.mo_energy = fock.diagonal()
        eris.oooo = c2[:nocc, :nocc, :nocc, :nocc].transpose(0, 2, 1, 3)
        eris.ovoo = c2[:nocc, :nocc, nocc:, :nocc].transpose(0, 2, 1, 3)
        eris.oovv = c2[:nocc, nocc:, :nocc, nocc:].transpose(0, 2, 1, 3)
        eris.ovvo = c2[:nocc, nocc:, nocc:, :nocc].transpose(0, 2, 1, 3)
        eris.ovov = c2[:nocc, :nocc, nocc:, nocc:].transpose(0, 2, 1, 3)
        eris.ovvv = c2[:nocc, nocc:, nocc:, nocc:].transpose(0, 2, 1, 3)
        eris.vvvv = c2[nocc:, nocc:, nocc:, nocc:].transpose(0, 2, 1, 3)
        eris.ovvv = lib.pack_tril(eris.ovvv.reshape(-1,nvir,nvir)).reshape(nocc,nvir,-1)
        eris.vvvv = ao2mo.restore(1, eris.vvvv, nvir)

        return eris
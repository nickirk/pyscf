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

from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from functools import reduce


def get_d3_slice_direct(dm1=None, dm2=None, slices=None):
    """
    Function to compute a block of the 3-RDM, according to the provided
    slices *without exchange of indices*.

    Args:
        dm1: 2D array. 1-RDM
        dm2: 4D array. 2-RDM
        slices: integers specifying the starting and stop position
                for each of the six indices of D^{p1p2p3}_{q1q2q3}.
                When the length of slices is smaller than 12, the other
                unspecified index will assume full range.
                Example input:
                [0, 100, 0, 100, 0, 50, 0, 100, 0, 100, 0, 50]
    Returns:
        d3: 6D array.
    """
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
    # TODO: dm1 or dm2 not provided, get them from  mf.
    # TODO: if mf dm's are used, they might have special structure in mo.
    #  Avoid brutal force evaluation.
    # if dm2 is None:
    #     get dm2 from mf

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
    # TODO: dm1 or dm2 not provided, get them from  mf.
    # TODO: if mf dm's are used, they might have special structure in mo.
    #  Avoid brutal force evaluation.
    # if dm2 is None:
    #     get dm2 from mf

    d_bar = -dm2
    d_bar += 2. * lib.einsum("mn, uv -> munv", dm1, dm1)
    d_bar -= 2. * 1. / 2 * lib.einsum("mn, vu -> muvn", dm1, dm1)

    return d_bar


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
        ct_h_pq: transformed 1-body integrals.
        ct_V_pqrs: transformed 2-body integrals.
    """

    def __init__(self, mf, t_basis=None, e_basis=None,
                 v_nmo=None, mo_coeff=None, mo_occ=None,
                 dm1=None, dm2=None, eri=None):
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
        self.t_basis = t_basis
        self.e_basis = e_basis
        # total number of orbitals
        self.nmo = len(mf.mo_energy)
        # core orbitals are doubly occupied, indices i,j,k,l...
        self.c_nmo = int(np.sum(mf.mo_occ) / 2)
        # active orbitals are those which are fractionally or low-lying
        # virtuals, a,b,c,d...
        if v_nmo is None:
            v_nmo = int(self.c_nmo * 1.2)
        self.v_nmo = v_nmo
        # target orbitals are the combination of core and active orbitals, 
        # indices p,q,r,s
        self.t_nmo = v_nmo + self.c_nmo
        # external orbitals are empty virtuals excluding the ones that are 
        # included in active space, x,y,z
        self.e_nmo = self.nmo - self.t_nmo
        #self.incore_complete = self.incore_complete or self.mol.incore_anyway

        ##################################################
        # don't modify the following attributes, they are not input options
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.emp2 = None
        self.e_hf = None
        self.e_corr = None
        self.t1 = None
        # partitioned t2 is dict
        self.t2 = None
        self.t2_full = None
        self._n_occ = None
        self._nmo = None
        self.chkfile = mf.chkfile
        self.callback = None

        if eri is None:
           eri = self.get_full_eri()
        self.eri = eri

        # need 1-RDM and 2-RDM in mo representation.
        if dm1 is None or dm2 is None:
            dm1 = np.diag(self.mf.mo_occ)
            dm2 = (np.einsum('ij, kl -> ijkl', dm1, dm1)
                   - np.einsum('ij, kl -> iklj', dm1, dm1)/2)
        self.dm1 = dm1
        self.dm2 = dm2

        keys = set(('amps_algo'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def kernel(self, **kwargs):
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
        4. Return an eris that is compatible with existing solvers such as
        CCSD.

        Args:


        Returns:
            ct_eris: class. modified ERI
        """

        if self.eri is None:
            self.eri = self.get_full_eri()

        # initialize the amplitudes
        if self.t1 is None or self.t2 is None:
            self.t1, self.t2 = self.init_amps(algo="mp2")



        # we need the one-body part of the original Hamiltonian h_mn
        h_mn = self.mf.get_hcore()
        c1_mn = self.get_c1(h_mn)

        # [h1, T2] contribution is 0.
        # self.commute_o1_t2_12(h_mn)

        # need dm1 and dm2 (user defined)
        dm2_bar = get_d_bar(self.dm1, self.dm2)
        c1_prime_mn = self.get_c1_prime(dm2_bar)


        # [h2, T1]
        c2_prime_mnuv = self.get_c2_prime()

        # [h2, T2]
        ct_0 = self.get_c0(self.dm1, self.dm2)
        c2_dprime_mnuv = self.get_c2_dprime(self.dm1)

        ct_h1 = c1_mn + c1_prime_mn
        # symmetrize ct_h1
        ct_h1 += ct_h1.T
        ct_h1 /= 2.

        ct_h1 += h_mn

        ct_v2 = c2_prime_mnuv + c2_dprime_mnuv
        # symmetrize ct_v2
        ct_v2_tmp = ct_v2.copy()
        ct_v2_tmp += ct_v2.transpose((1, 0, 3, 2))
        ct_v2_tmp += ct_v2.transpose((2, 3, 0, 1))
        ct_v2_tmp += ct_v2.transpose((3, 2, 1, 0))
        ct_v2 = 1./4 * ct_v2_tmp

        # The second commutator 1/2*[[F, T], T]
        # First construct F, the Fock matrix, as an approximation to H
        fock_mn = self.mf.get_fock()
        ct_h1 += 1/.2 * self.commute_o1_t(self.commute_o1_t(fock_mn))


        # final step might need to do some transformation on the eris so that
        # other solvers can use it directly as usual integrals.

        return ct_0, ct_h1, ct_v2

    def get_full_eri(self):
        self.eri = ao2mo.incore.full(self.mf._eri, self.mf.mo_coeff)
        # use no symmetries for initial implementation
        self.eri = ao2mo.restore(1, self.eri, self.nmo)
        return self.eri
    def get_n_occ(self):
        # count the number of singly, doubly and fractionally occupied orbitals
        if self._n_occ is None:
            self._n_occ = np.count_nonzero(self.mf.mo_occ)
        return self._n_occ

    def get_t2_full(self):
        # get t2_full
        if self.t2_full is None:
            self.t2_full = np.zeros([self.e_nmo, self.e_nmo,
                                     self.t_nmo,
                                     self.t_nmo])
            if self.t2["xyij"] is None or self.t2["xyab"] is None or self.t2[
                "xyai"] is None:
                print("Block t2 amps not initialized! Initializing it using "
                      "default algo mp2.")
                self.init_amps(algo="mp2")
            self.t2_full[:, :, :self.c_nmo, :self.c_nmo] = self.t2["xyij"]
            self.t2_full[:, :, self.c_nmo:, self.c_nmo:] = self.t2["xyab"]
            self.t2_full[:, :, self.c_nmo:, :self.c_nmo] = self.t2["xyai"]
            self.t2_full[:, :, :self.c_nmo, self.c_nmo:] = np.transpose(
                self.t2["xyai"], (0, 1, 3, 2))
        return self.t2_full

    def init_amps(self, algo="mp2"):
        """
        Get the transformaiton amplitudes.

        Args:
            algo: string. Method to compute amps.

        Returns:

        """
        if algo == "mp2":
            t1, t2 = self.get_mp2_amps()
        elif algo == "f12":
            t1, t2 = self.get_f12_amps()
        else:
            raise NotImplementedError
        return t1, t2

    def get_mp2_amps(self):
        fock = self.mf.get_fock()

        mo_e_i = self.mf.mo_energy[:self.c_nmo]
        # if regularization is needed, one can do it here.
        mo_e_a = self.mf.mo_energy[self.c_nmo:self.t_nmo]
        mo_e_x = self.mf.mo_energy[self.t_nmo:]

        e_xi = mo_e_x[:, None] - mo_e_i[None, :]
        e_ai = mo_e_a[:, None] - mo_e_i[None, :]
        e_xa = mo_e_x[:, None] - mo_e_a[None, :]

        t_ai = fock[self.c_nmo:self.t_nmo, :self.c_nmo]
        t_xi = fock[self.t_nmo:, :self.c_nmo]
        t_xa = fock[self.t_nmo:, self.c_nmo:self.t_nmo]

        t_xyab = self.eri[self.t_nmo:, self.t_nmo:,
                 self.c_nmo:self.t_nmo, self.c_nmo:self.t_nmo]
        t_xyij = self.eri[self.t_nmo:, self.t_nmo:, :self.c_nmo, :self.c_nmo]
        t_xyai = self.eri[self.t_nmo:, self.t_nmo:, self.c_nmo:self.t_nmo,
                 :self.c_nmo]

        t_ai /= e_ai
        t_xi /= e_xi
        t_xa /= e_xa

        t_xyab /= lib.direct_sum("xa, yb -> xyab", e_xa, e_xa)
        t_xyij /= lib.direct_sum("xi, yj -> xyij", e_xi, e_xi)
        t_xyai /= lib.direct_sum("xa, yi -> xyai", e_xa, e_xi)

        t1 = {"ai": t_ai, "xi": t_xi, "xa": t_xa}
        t2 = {"xyij": t_xyij, "xyab": t_xyab, "xyai": t_xyai}

        return t1, t2

    def get_f12_amps(self):
        raise NotImplementedError

    def commute_o1_t(self, o1_mn):
        """
        This function assembles the two parts of the commutator
        [o1, T]=[o1, h1]+[o1, v2] together.

        Args:
            o1_mn: 2D array of size [nmo, nmo]. The one-body integral to be
            transformed

        Returns:
            [ct_o1, ct_v2]: list of two arrays of size [nmo, mno] and [nmo,
            nmo, nmo, nmo], respectively. The transformed integrals.
        """
        # Note get_c1 already symmetrize the transformed integral.
        ct_o1 = self.get_c1(o1_mn)

        # Note get_c2 has 0 contribution to integrals within the target space.
        # ct_v2 = self.get_c2(o1_mn)

        return ct_o1

    def get_c1(self, o_mn=None):
        """
        This function calculates the commutator between h1_mn and O_px.

        Args:
            o_mn: a 1-body (2 indices) tensor defined on the parent space

        Returns:
            c_mn: transformed integral, defined on the parent space
        """
        if o_mn is None:
            o_mn = self.mf.get_hcore()
        o_mx = o_mn[:, self.t_nmo:]
        o_mi = o_mn[:, :self.c_nmo]
        o_ma = o_mn[:, self.c_nmo:self.t_nmo]
        t_xa = self.t1["xa"]
        t_xi = self.t1["xi"]
        # equ (35) in Ref: Phys. Chem. Chem. Phys., 2012, 14, 7809–7820
        c_mn = np.zeros([self.nmo, self.nmo])
        # only the following terms are relevant to the target space
        c_mn[:, self.c_nmo:self.t_nmo] = 2. * lib.einsum("mx, xa -> ma",
                                                         o_mx, t_xa)
        c_mn[:, :self.c_nmo] = 2. * lib.einsum("mx, xi -> mi", o_mx, t_xi)

        # connections between target and external space
        c_mn[:, self.t_nmo:] -= 2. * lib.einsum("ma, xa -> mx", o_ma, t_xa)
        c_mn[:, self.t_nmo:] -= 2. * lib.einsum("mi, xi -> mx", o_mi, t_xi)

        # need to symmetrize it
        c_mn += lib.einsum("mn -> nm", c_mn)
        c_mn /= 2.
        return c_mn

    def get_c2(self, o_mn=None, only_target=True):
        """
        This function computes the [\hat{h}_1, \hat{T}_2]_{1,2}, equ (37) 
        and (38) in Ref: Phys. Chem. Chem. Phys., 2012, 14, 7809–7820

        Args:
            o_mn: Rank 2 tensor of size [nao, nao], defined on the parent
                  basis set

        Returns:
            ct_eris: Transformed rank 4 tensor defined on the parent basis set.
        """
        ct_eris = _PhysicistsERIs()
        o_xa = o_mn[self.t_nmo:, self.c_nmo:self.t_nmo]
        o_xi = o_mn[self.t_nmo:, :self.c_nmo]
        # o_xy = o_mn[self.t_nmo:, self.c_nmo:self.t_nmo]
        # o_ab = o_mn[self.c_nmo:self.t_nmo:, i
        #            self.c_nmo:self.t_nmo]
        # o_ai = o_mn[self.c_nmo:self.t_nmo:, :self.c_nmo]

        # TODO: check ontribution of this commutator to integrals within the
        #  target space is 0
        if only_target:
            return

        # Connections to external space are currently not fully implemented.
        ct_eris.oove = 4 * lib.einsum("xa, xyij -> ijax", o_xa, self.t2["xyij"])
        ct_eris.vvoe = 4 * lib.einsum("xi, xyab -> abiy", o_xi, self.t2["xyab"])
        ct_eris.vvve = 4 * lib.einsum("xc, xyab -> abcy", o_xa, self.t2["xyab"])
        ct_eris.oooe = 4 * lib.einsum("ak, xyij -> ijkx", o_xi, self.t2["xyij"])
        ct_eris.oove = 4 * lib.einsum("ab, xyij -> ijax", o_xa, self.t2["xyij"])
        ct_eris.vvoe = 4 * lib.einsum("ai, xyab -> abiy", o_xi, self.t2["xyab"])
        ct_eris.vvve = 4 * lib.einsum("ac, xyab -> abcy", o_xa, self.t2["xyab"])

        return ct_eris

    def get_c2_prime(self, v2=None):
        """
        This function computes the CT contirubtion [v2, t1].
        Only contribution to the target space is implemented. Connections
        between target and external space, or that within external space
        are currently not implemented.

        Args:
            v2: rank 4 tensor (ndarray). Specify which 2-body operator to
                commute with the T1 operator. If not supplied, the default is
                the Coulomb eri.
        Returns:
            c2_prime: rank 4 tensor. 
        
        """
        if v2 is None:
            v2 = self.eri
        c2_prime = np.zeros(v2.shape)
        v2_ooeo = v2[:self.c_nmo, :self.c_nmo, self.t_nmo:, :self.c_nmo]
        oooo = 4. * lib.einsum("ijxl, xk -> ijkl", v2_ooeo, self.t1["xi"])
        c2_prime[:self.c_nmo, :self.c_nmo, :self.c_nmo:, :self.c_nmo] = oooo

        v2_oveo = v2[:self.c_nmo, self.c_nmo:self.t_nmo, self.t_nmo:, :self.c_nmo]
        ovoo = 4. * lib.einsum("iaxk, xj -> iajk", v2_oveo, self.t1["xi"])
        c2_prime[:self.c_nmo, self.c_nmo:self.t_nmo, :self.c_nmo,
                 :self.c_nmo] = ovoo

        v2_ooev = v2[:self.c_nmo, :self.c_nmo, self.t_nmo:, self.c_nmo:self.t_nmo]
        oovv = 4. * lib.einsum("ijxb, xa -> ijab", v2_ooev, self.t1["xa"])
        c2_prime[:self.c_nmo, :self.c_nmo, self.c_nmo:self.t_nmo,
                 self.c_nmo:self.t_nmo] = oovv

        v2_oveo = v2[:self.c_nmo, self.c_nmo:self.t_nmo, self.t_nmo:, :self.c_nmo]
        ovvo = 4. * lib.einsum("iaxj, xb -> iabj", v2_oveo, self.t1["xa"])
        c2_prime[:self.c_nmo, self.c_nmo:self.t_nmo, self.c_nmo:self.t_nmo,
                 :self.c_nmo] = ovvo

        v2_ovev = v2[:self.c_nmo, self.c_nmo:self.t_nmo, self.t_nmo:,
                  self.c_nmo:self.t_nmo]
        ovov = 4. * lib.einsum("iaxb, xj -> iajb", v2_ovev, self.t1["xi"])
        c2_prime[:self.c_nmo, self.c_nmo:self.t_nmo, :self.c_nmo,
                 self.c_nmo:self.t_nmo] = ovov
        ovvv = 4. * lib.einsum("iaxc, xb -> iabc", v2_ovev, self.t1["xa"])
        c2_prime[:self.c_nmo, self.c_nmo:self.t_nmo, self.c_nmo:self.t_nmo,
                 self.c_nmo:self.t_nmo] = ovvv

        v2_vvev = v2[self.c_nmo:self.t_nmo, self.c_nmo:self.t_nmo, self.t_nmo:,
                     self.c_nmo:self.v_nmo]
        vvvv = 4. * lib.einsum("abxd, xc -> abcd", v2_vvev, self.t1["xa"])
        c2_prime[self.c_nmo:self.t_nmo, self.c_nmo:self.t_nmo,
                 self.c_nmo:self.t_nmo, self.c_nmo:self.t_nmo] = vvvv

        return c2_prime

    def get_c0(self, dm1=None, dm2=None):
        """
        Calculate the constant contribution to energy from CT.
        """
        # construct Do^{p1p2e2}_{a1q2a2}
        dm2 = get_d_zero(dm1, dm2)
        # need a slice of the full ERI
        # TODO: sorting out which is better, using full ERI and then slicing
        #  or sliced ERI and reconstruct. Here assume the full ERI is available

        v_mnxu = self.eri[:, :, self.t_nmo:, :]
        slices = [0, self.nmo, 0, self.nmo, self.t_nmo, self.nmo,
                  0, self.t_nmo, 0, self.nmo, 0, self.t_nmo]
        d3 = get_d3_slice(dm1, dm2, slices)
        t2 = self.get_t2_full()
        c0 = 2.0 * lib.einsum("mnxu, xypq, mnypuq ->", v_mnxu, t2,
                              d3)
        v_mnpu = self.eri[:, :, :self.t_nmo, :]
        slices = [0, self.nmo, 0, self.nmo, 0, self.t_nmo,
                  self.t_nmo, self.nmo, 0, self.nmo, self.t_nmo,
                  self.nmo]
        d3 = get_d3_slice(dm1, dm2, slices)
        c0 -= 2.0 * lib.einsum("mnpu, xyqr, mnrxuy ->", v_mnpu, t2,
                               d3)

        return c0

    def get_c1_prime(self, dm2_bar):
        """
        Function to compute the 2D tensor associated with the 1-body operator
        generated by [h2, T2]

        Args:
            dm2_bar: 4D tensor. As defined in equation (19) in Ref:
            Phys. Chem. Chem. Phys., 2012, 14, 7809–7820

        Returns:
            c1_prime_mn: 2D tensor. The 1-body contribution from [h2, T2]
        """

        # need the original t2
        t2 = self.get_t2_full()
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
        s6_mn = lib.einsum("mnuv, wnuv -> wm", self.eri, dm2_bar)
        s6_px = s6_mn[:self.t_nmo, self.t_nmo:]
        s6_xp = s6_mn[self.t_nmo:, :self.t_nmo]

        # s0
        # will this modify the original t2_full? TODO: check
        t2 -= 1. / 2 * np.transpose(t2, (0, 1, 3, 2))
        s0_xipj = lib.einsum("xypq, xipj -> yiqj", t2, dm2_bar_xipj)

        # constructing c1_prime
        c1_prime_mn = np.zeros([self.nmo, self.nmo])
        # adding contributions from e1_prime_pq
        # first term in equation 44
        v_mixj = self.eri[:, :self.c_nmo, self.t_nmo:,
                 :self.c_nmo]
        v_mijx = self.eri[:, :self.c_nmo, :self.c_nmo,
                 self.t_nmo:]
        c1_prime_mn[:, :self.t_nmo] -= lib.einsum("mixj, xipj -> "
                                                       "mp", v_mixj,
                                                  s0_xipj)
        c1_prime_mn[:, :self.t_nmo] -= lib.einsum("mijx, xipj -> "
                                                       "mp", v_mijx,
                                                  s1_xipj)

        # second term in equation 44
        v_minx = self.eri[:, :self.c_nmo, :, self.t_nmo:]
        v_minx -= 1. / 2 * self.eri[:, :self.c_nmo, self.t_nmo:,
                           :].transpose(0, 1, 3, 2)
        c1_prime_mn -= lib.einsum("minx, xi -> mn", v_minx, s2_xi)

        # third term in equation 44
        v_ijxm = self.eri[:self.c_nmo, :self.c_nmo,
                 self.t_nmo:, :]
        c1_prime_mn[:, self.t_nmo:] += lib.einsum("ijxm, xyij -> my",
                                                  v_ijxm, s4_xyij)

        # fourth term in equation 44
        # Note that s6_px has none zero values in block ap, but it does not
        # contribute to e1_prime_pq, since in the end it contracts with the
        # t2 amplitudes which have only none zero in block xypq.
        c1_prime_mn[:self.t_nmo, self.t_nmo:] -= \
            lib.einsum("xypq, qy -> px", t2, s6_px)

        # adding contributions from a1_prime_pq, note that the overall sign is -
        # first term in equation 45
        v_mipj = self.eri[:, :self.c_nmo, :self.t_nmo, :self.c_nmo]
        v_mijp = self.eri[:, :self.c_nmo, :self.c_nmo, :self.t_nmo]
        c1_prime_mn[:, self.t_nmo:] += lib.einsum("mipj, xipj -> mx",
                                                  v_mipj, s0_xipj)
        c1_prime_mn[:, self.t_nmo:] += lib.einsum("mijp, xipj -> mx",
                                                  v_mijp, s1_xipj)

        # second term in equation 45
        v_minp = self.eri[:, :self.c_nmo, :, :self.t_nmo]
        v_mipn = self.eri[:, :self.c_nmo, :self.t_nmo, :]
        v_minp -= 1. / 2 * v_mipn.transpose((0, 1, 3, 2))
        c1_prime_mn += lib.einsum("minp, ip -> mn", v_minp, s3_ip)

        # third term in equation 45
        v_ijpm = self.eri[:self.c_nmo, :self.c_nmo, :self.t_nmo, :]
        c1_prime_mn[:, :self.t_nmo] -= lib.einsum("ijpm, ijpq -> mq",
                                                  v_ijpm, s5_ijpq)

        # fourth term in equation 45
        c1_prime_mn[:self.t_nmo, self.t_nmo:] += \
            lib.einsum("xypq, yq -> px", t2, s6_xp)

        c1_prime_mn *= 2.

        return c1_prime_mn

    def get_c2_dprime(self, dm1=None):
        """
        Function to construct c_double_prime_mnuv, Eq (47)
        Args:
            dm1: 2D tensor. 1-RDM

        Returns:
            c2_dprime_mnuv: 4D tensor.
        """

        # Construct T0-T3 intermediates first.
        t2 = self.get_t2_full()
        cap_t0_xyip = lib.einsum("xypq, ip -> xyiq", t2, dm1[:self.c_nmo,
                                                         :self.t_nmo])
        cap_t1_ixpq = lib.einsum("xypq, ix -> iypq", t2, dm1[:self.c_nmo,
                                                         self.t_nmo:])
        cap_t2_xp = lib.einsum("xypq, xp -> yq", t2, dm1[self.t_nmo:,
                                                     :self.t_nmo])
        cap_t2_xp -= 1. / 2 * lib.einsum("yxqp, xp -> yq", t2,
                                         dm1[self.t_nmo:,
                                         :self.t_nmo])
        cap_t3_mn = lib.einsum("mnuv, nv -> mn", self.eri, dm1)
        cap_t3_mn -= 1. / 2 * lib.einsum("mnvu, nv -> mv", self.eri, dm1)

        c2_dprime_mnuv = np.zeros([self.nmo, self.nmo, self.nmo,
                                   self.nmo])

        # adding e_dprime_mnuv contribution
        # TODO: the on-the-fly slicing is not the most efficient way, fix it
        # first term in equation 48
        v_mnxy = self.eri[:, :, self.t_nmo:, self.t_nmo:]
        c2_dprime_mnuv[:, :, :self.t_nmo, :self.t_nmo] += \
            1. / 2 * lib.einsum("mnxy, xypq -> mnpq", v_mnxy, t2)
        # second term in equ 48
        v_mxni = self.eri[:, self.t_nmo:, :, :self.c_nmo]
        c2_dprime_mnuv[:, :self.t_nmo, :, self.t_nmo:] += \
            lib.einsum("mxni, xyip -> mpny", v_mxni, cap_t0_xyip)
        c2_dprime_mnuv[:, :self.t_nmo, :, self.t_nmo:] -= \
            1. / 2 * lib.einsum("mxni, yxip -> mpny", v_mxni, cap_t0_xyip)
        v_mxin = self.eri[:, self.t_nmo:, :self.c_nmo, :]
        c2_dprime_mnuv[:, :self.t_nmo, :, self.t_nmo:] -= \
            1. / 2 * lib.einsum("mxin, xyip -> mpny", v_mxin, cap_t0_xyip)
        # third term in equ 48
        c2_dprime_mnuv[:, :self.t_nmo, self.t_nmo:, :] -= \
            1. / 2 * lib.einsum("mxin, yxip -> mpxn", v_mxin, cap_t0_xyip)

        # fourth term in equ 48
        v_mnxi = self.eri[:, :, self.t_nmo:, :self.c_nmo]
        c2_dprime_mnuv[:, :, :self.t_nmo, :self.t_nmo] -= \
            1. / 2 * lib.einsum("mnxi, ixpq -> mnqp", v_mnxi, cap_t1_ixpq)

        # fifth term in equ 48
        v_mnxu = self.eri[:, :, self.t_nmo:, :]
        c2_dprime_mnuv[:, :, :self.t_nmo, :] += \
            lib.einsum("mnxu, xp -> mnpu", v_mnxu, cap_t2_xp)

        # sixth term in equ 48
        c2_dprime_mnuv[:, self.t_nmo:, :self.t_nmo,
        :self.t_nmo] += lib.einsum("xypq, mx -> mypq", t2,
                                   cap_t3_mn[:, self.t_nmo:])

        # adding contributions from a_dprime_mnuv, notice the overall - sign
        v_mnpq = self.eri[:, :, :self.t_nmo, :self.t_nmo]
        c2_dprime_mnuv[:, :, self.t_nmo:, self.t_nmo:] += \
            1. / 2 * lib.einsum("mnpq, xypq -> mnxy", v_mnpq, t2)
        # second term in equ 49
        v_mpni = self.eri[:, :self.t_nmo, :, :self.c_nmo]
        c2_dprime_mnuv[:, self.t_nmo:, :, :self.t_nmo] += \
            lib.einsum("mpni, ixpq -> mxnq", v_mpni, cap_t1_ixpq)
        c2_dprime_mnuv[:, self.t_nmo:, :, :self.t_nmo] -= \
            1. / 2 * lib.einsum("mpni, ixpq -> mxnq", v_mpni, cap_t1_ixpq)
        v_mpin = self.eri[:, :self.t_nmo, :self.c_nmo, :]
        c2_dprime_mnuv[:, self.t_nmo:, :, :self.t_nmo] -= \
            1. / 2 * lib.einsum("mpin, ixpq -> mxnq", v_mpin, cap_t1_ixpq)

        # third term in equ 49
        c2_dprime_mnuv[:, self.t_nmo:, :self.t_nmo, :] -= \
            1. / 2 * lib.einsum("mpin, ixqp -> mxqn", v_mpin, cap_t1_ixpq)

        # fourth term in equ 49
        v_mnpi = self.eri[:, :, :self.t_nmo, :self.c_nmo]
        c2_dprime_mnuv[:, :, self.t_nmo:, self.t_nmo:] -= \
            1. / 2 * lib.einsum("mnpi, xyip -> mnyx", v_mnpi, cap_t0_xyip)

        # fifth term in equ 49
        v_mnpu = self.eri[:, :, :self.t_nmo, :]
        c2_dprime_mnuv[:, :, self.t_nmo:, :] += \
            lib.einsum("mnpu, xp -> mnxu", v_mnpu, cap_t2_xp)

        # sixth term in equ 49
        c2_dprime_mnuv[:, :self.t_nmo, self.t_nmo:,
        self.t_nmo:] += lib.einsum("xypq, mp -> mqxy", t2,
                                   cap_t3_mn[:, :self.t_nmo])

        c2_dprime_mnuv *= 4.
        return c2_dprime_mnuv

    def get_hf_energy(self):
        return
class _PhysicistsERIs:
    """
    <pq|rs>

    Note that the target and external spaces might be spanned by two
    nonorthogonal basis sets. In an initial attempt, let's assume the
    target basis set is just a subset of a large basis set and the
    external space is spanned by the rest.
    """

    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.n_occ = None
        self.fock = None
        self.e_hf = None

        # There are 3x3x3 = 27 partitions...with 4-fold symmetry

        # Target space partitions
        # Use a dict will make implementation simpler, Python dict will also
        # allow different objects listed. E.g. for "abcd", one might need to 
        # store out-core and use hdf5.

        # However, it might make interfacing with other parts complicated.
        # If one only uses it as internal data structure and provide an 
        # interface to the outside, it might be Ok.

        self.oooo = None
        self.ovoo = None
        self.oovv = None
        self.ovvo = None
        self.ovov = None
        self.ovvv = None
        self.vvvv = None

        self.vooo = None
        self.ooov = None
        self.vvoo = None
        self.voov = None
        self.vovo = None
        self.vvov = None
        self.vovv = None
        self.vvvo = None

        # Connection to external space partitions
        # self.ooee = None
        # self.vvee = None
        # self.vvoe = None
        # self.vvve = None
        # self.veve = None
        # self.evee = None
        # self.eoee = None

        self.full = None

    def _common_init_(self, ct, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = ct.mo_coeff
        self.mo_coeff = mo_coeff

        # Note: Recomputed fock matrix and HF energy since SCF may not be 
        # fully converged.
        fock_ao = ct.mf.get_fock()
        self.fock = reduce(np.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        # self.e_hf = ct.mf.energy_tot(dm=dm, vhf=vhf)
        n_occ = self.n_occ
        self.full = self.get_eris_incore(ct)
        return self

    def get_eris_incore(self, ct, mo_coeff=None):

        if mo_coeff is None:
            mo_coeff = ct.mo_coeff
        if self.full is None:
            nmo = len(ct.mf.mo_energy)
            self.full = ao2mo.incore.full(ct.mf._eri, mo_coeff)

            if self.full.ndim == 4:
                self.full = ao2mo.restore(4, self.full, nmo)

        return self.full

    #def get_ovvv(self, *slices):
    #    '''To access a subblock of ovvv tensor'''
    #    ovw = np.asarray(self.ovvv[slices])
    #    n_occ, n_vir, n_vir_pair = ovw.shape
    #    ovvv = lib.unpack_tril(ovw.reshape(n_occ * n_vir, n_vir_pair))
    #    n_vir1 = ovvv.shape[2]
    #    return ovvv.reshape(n_occ, n_vir, n_vir1, n_vir1)

    def symmetrize(self):
        """
        This function symmetrize CT 2-body intermediates.
        Needs oooo, ovoo, vooo, ooov, oovo, oovv, vvoo, ovvo, voov, ovov,
        vovo, ovvv, vovv, vvov, vvvo

        """
        # TODO: oooo partition should already be symmetric: should check if
        #  indeed
        #  is true
        oooo_orig = self.oooo.copy()
        self.oooo += np.transpose(oooo_orig, (1, 0, 3, 2))
        self.oooo += np.transpose(oooo_orig, (2, 3, 0, 1))
        self.oooo += np.transpose(oooo_orig, (3, 2, 1, 0))
        self.oooo /= 4.

        # ovoo: 1/4(ovoo + vooo + ooov + oovo)
        self.ovoo += np.transpose(self.vooo, (1, 0, 3, 2))
        self.ovoo += np.transpose(self.ooov, (2, 3, 0, 1))
        self.ovoo += np.transpose(self.oovo, (3, 2, 1, 0))
        self.ovoo /= 4.

        self.vooo = self.ooov = self.oovo = self.ovoo

        # oovv: 1/4(oovv + oovv + vvoo + vvoo)
        self.oovv += np.transpose(self.oovv, (1, 0, 3, 2))
        self.oovv += np.transpose(self.vvoo, (2, 3, 0, 1))
        self.oovv += np.transpose(self.vvoo, (3, 2, 1, 0))
        self.oovv /= 4.

        self.vvoo = self.oovv

        # ovvo: 1/4(ovvo + voov + voov + ovvo)
        self.ovvo += np.transpose(self.ovvo, (3, 2, 1, 0))
        self.ovvo += np.transpose(self.voov, (1, 0, 3, 2))
        self.ovvo += np.transpose(self.voov, (2, 3, 0, 1))
        self.ovvo /= 4.

        self.voov = self.ovvo

        # ovov: 1/4(ovov + vovo + ovov + vovo)
        self.ovov += np.transpose(self.ovov, (3, 2, 1, 0))
        self.ovov += np.transpose(self.vovo, (1, 0, 3, 2))
        self.ovov += np.transpose(self.vovo, (2, 3, 0, 1))
        self.ovov /= 4.

        self.vovo = self.ovov

        # ovvv: 1/4(ovvv + vovv + vvov + vvvo)
        self.ovvv += np.transpose(self.vovv, (1, 0, 3, 2))
        self.ovvv += np.transpose(self.vvov, (2, 3, 0, 1))
        self.ovvv += np.transpose(self.vvvo, (3, 2, 1, 0))
        self.ovvv /= 4.

        self.vovv = self.vvov = self.vvvo = self.ovvv

        # vvvv: vvvv
        # really need to verify if the following is necessary, because for
        # large number of orbitals, the following operations are expensive.
        vvvv_orig = self.vvvv.copy()
        self.vvvv += np.transpose(vvvv_orig, (1, 0, 3, 2))
        self.vvvv += np.transpose(vvvv_orig, (2, 3, 0, 1))
        self.vvvv += np.transpose(vvvv_orig, (3, 2, 1, 0))
        self.vvvv /= 4.

        return

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
# 
# Author: Ke Liao <ke.liao.whu@gmail.com>

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.ct import ctsd
from pyscf.ct.ctsd import symmetrize
from pyscf.pbc import scf
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)  # noqa



class RCTSD(ctsd.CTSD):
    """
    Restricted k-CTSD.
    """
    def __init__(self, mf, a_nmo=None, mo_coeff=None, mo_occ=None,
                 dm1=None, dm2=None, eri=None):
        assert (isinstance(mf, scf.khf.KSCF))
        self.kpts = mf.kpts
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)

    @property
    def nkpts(self):
        return len(self.kpts)
    
    def init_amps(self):
        
        if self._amps_algo == "mp2":
            return self.get_mp2_amps
        else:
            raise NotImplementedError
    
    def get_mp2_amps(self):
        time0 = logger.process_clock(), logger.perf_counter()
        # TODO: adapt kccsd_rhf._ERI for kctsd
        t_nmo = self.t_nmo
        e_nmo = self.e_nmo
        nkpts = self.nkpts
        t1 = np.zeros((nkpts, e_nmo, t_nmo), dtype=eris.fock.dtype)
        t2 = np.empty((nkpts, nkpts, nkpts, e_nmo, e_nmo, t_nmo, t_nmo), dtype=eris.fock.dtype)
        mo_e_p = [eris.mo_energy[k][:t_nmo] for k in range(nkpts)]
        mo_e_x = [eris.mo_energy[k][t_nmo:] for k in range(nkpts)]

        # Get location of padded elements in occupied and virtual space
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self, kind="split")

        kconserv = self.khelper.kconserv
        touched = np.zeros((nkpts, nkpts, nkpts), dtype=bool)
        for kp, kq, kx in kpts_helper.loop_kkk(nkpts):
            if touched[kp, kq, kx]:
                continue

            ky = kconserv[kp, kx, kq]
            # For discussion of LARGE_DENOM, see t1new update above
            e_xp = _get_epq([0, t_nmo, kp, mo_e_p, nonzero_opadding],
                           [t_nmo, e_nmo, kx, mo_e_x, nonzero_vpadding],
                           fac=[1.0,-1.0])

            e_yq = _get_epq([0, t_nmo, kq, mo_e_p, nonzero_opadding],
                           [t_nmo, e_nmo, ky, mo_e_x, nonzero_vpadding],
                           fac=[1.0,-1.0])


            e_pqxy = e_xp[:, None, :, None] + e_yq[:, None, :]

            eris_pqxy = eris.oovv[kp, kq, kx]
            eris_pqyx = eris.oovv[kp, kq, ky]
            t2[kp, kq, kx] = eris_pqxy.conj() / e_pqxy

            if kx != ky:
                e_pqyx = e_pqxy.transpose(0, 1, 3, 2)
                t2[kp, kq, ky] = eris_pqyx.conj() / e_pqyx

            touched[kp, kq, kx] = touched[kp, kq, ky] = True

        logger.timer(self, 'init mp2', *time0)
        return t1, t2

    get_mp2_amps = get_mp2_amps

    def kernel(self, t1=None, t2=None, eri=None, amps_algo='mp2'):
        if t1 is None and t2 is None:
            self.amps_algo = amps_algo
            t1, t2 = self.init_amps()
        elif t2 is None:
            t2 = self.init_amps()[1]
        
        if eri is None:
            self.eri = self.ao2mo()

        ct_0, ct_h1, ct_v2 = self.commute(h1=h_mn, v2=self.eri)

    def commute(self, o0=0., o1=None, o2=None):
        """commutation operator. Needs to take the k-conservation into account.
        """
        c1 = None
        c2 = None
        c0 = o0
        if o1 is not None:
            c1, c2 = self.commute_o1_t(o1)
        if o2 is not None:
            c0, c1_prime, c2_prime, c2_dprime = self.commute_o2_t(o2)
            c1 += c1_prime
            c2 += c2_prime + c2_dprime
        
        if c1 is not None:
            c1 = symmetrize(c1)
        if c2 is not None:
            c2 = symmetrize(c2)

        return c0, c1, c2

    def commute_o1_t(self, o1, t1, t2):
        ct_o1 = self.get_k_c1(o1, t1)
        ct_o2 = self.get_k_c2(o1, t2)

        return ct_o1, ct_o2
    
    def get_k_c1(self, o1, t1):
        c1_kmn = np.zeros(o1.shape)
        for ki in range(self.nkpts):
            o1_mn = o1[ki].copy
            c1_kmn[ki] = self.get_c1(o1_mn, t1=t1[ki])
        
        return c1_kmn
        
    def get_k_c2(self, o1, t2):
        c2 = np.empty(t2.shape)
        kconserv = self.khelper.kconserv
        touched = np.zeros((self.nkpts, self.nkpts, self.nkpts), dtype=bool)

        for ki, kj, ka in kpts_helper.loop_kkk(self.nkpts):
            if touched[ki, kj, ka]:
                continue
            kb = kconserv[ki, ka, kj]
            o_mx = o1[ki, :, self.t_nmo:]
            o_mp = o1[ka, :, :self.t_nmo]
            c2_kikjka = 4.*lib.einsum(
                "mx, nypq -> mypq", o_mx, t2[ki, kj, ka]
            )
            c2_kakbki = -4.*lib.einsum(
                "mp, xypq -> mqxy", o_mp, t2[ki, kj, ka]
            )
            c2[ki, kj, ka, self.t_nmo:, :self.t_nmo, :self.t_nmo] = c2_kikjka
            c2[ka, kb, ki, :self.t_nmo, self.t_nmo:, self.t_nmo:] = c2_kakbki

        return c2

    def commute_o2_t(self, v2):
        pass

    class _ERIS:
        def __init__(self, ct, method='incore'):
            cell = ct._scf.cell
            kpts = ct.kpts
            nkpts = ct.nkpts
            nocc = ct.nocc
            nmo = ct.nmo
            nvir = nmo - nocc

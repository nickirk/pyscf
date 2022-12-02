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
from pyscf.pbc import scf
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)  # noqa


kernel = pyscf.ct.ctsd.kernel

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

    def commute(self):
        """commutation operator. Needs to take the k-conservation into account.
        """
        pass
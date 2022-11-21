#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Authors: Ke Liao <ke.liao.whu@gmail.com>
# Based on kintermediates_rhf.py
#          
#

import numpy as np

from itertools import product
from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM  # noqa
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)  # noqa
from pyscf.pbc.cc.kintermediates_rhf import cc_Foo, cc_Fov, cc_Fvv

#einsum = np.einsum
einsum = lib.einsum

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"


### Eqs. (40)-(41) "lambda"

def Loo(t1,t2,eris,kconserv, ki=0):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lki = cc_Foo(t1,t2,eris,kconserv)[ki]
    Lki[ki] += einsum('kc,ic->ki',fov[ki],t1[ki])
    for kl in range(nkpts):
        Lki += 2*einsum('klic,lc->ki',eris.ooov[ki,kl,ki],t1[kl])
        Lki +=  -einsum('lkic,lc->ki',eris.ooov[kl,ki,ki],t1[kl])
    return Lki

def Lvv(t1,t2,eris,kconserv, ka=1):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lac = cc_Fvv(t1,t2,eris,kconserv)[ka]
    Lac += -einsum('kc,ka->ac',fov[ka],t1[ka])
    for kk in range(nkpts):
        Svovv = 2*eris.vovv[ka,kk,ka] - eris.vovv[ka,kk,kk].transpose(0,1,3,2)
        Lac += einsum('akcd,kd->ac',Svovv,t1[kk])
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(t1, t2, eris, kconserv, out=None, ki=0, kj=0):
    nkpts, nocc, nvir = t1.shape
    shape_Wklij = [nkpts, nkpts, 1] + eris.oooo.shape[3:]
    Wklij = _new(shape_Wklij, t1.dtype, out)
    for kk in range(nkpts):
        kl = kconserv[ki, kk, kj]
        oooo  = einsum('klic,jc->klij',eris.ooov[kk,kl,ki],t1[kj])
        oooo += einsum('lkjc,ic->klij',eris.ooov[kl,kk,kj],t1[ki])
        oooo += eris.oooo[kk,kl,ki]

        vvoo = eris.oovv[kk,kl].transpose(0,3,4,1,2).reshape(nkpts*nvir,nvir,nocc,nocc)
        t2t  = t2[ki,kj].copy().transpose(0,3,4,1,2)
        t2t[ki] += einsum('ic,jd->cdij',t1[ki],t1[kj])
        t2t = t2t.reshape(nkpts*nvir,nvir,nocc,nocc)
        oooo += einsum('cdkl,cdij->klij',vvoo,t2t)
        Wklij[kk,kl, ki] = oooo

        # Be careful about making this term only after all the others are created
        Wklij[kl,kk,kj] = Wklij[kk,kl,ki].transpose(1,0,3,2)
    return Wklij

def cc_Wvvvv(t1, t2, eris, kconserv, out=None, ka=0, kb=0):
    nkpts, nocc, nvir = t1.shape
    shape_Wabcd = [1, 1, nkpts] + eris.vvvv.shape[3:]
    Wabcd = _new(shape_Wabcd, t1.dtype, out)
    for kc in range(nkpts):
        kd = kconserv[ka,kc,kb]
        vvvv  = einsum('akcd,kb->abcd', eris.vovv[ka,kb,kc], -t1[kb])
        vvvv += einsum('bkdc,ka->abcd', eris.vovv[kb,ka,kd], -t1[ka])
        vvvv += eris.vvvv[ka,kb,kc]
        Wabcd[ka,kb,kc] = vvvv

        Wabcd[kb,ka,kd] = Wabcd[ka,kb,kc].transpose(1,0,3,2)

    return Wabcd

def cc_Wvoov(t1, t2, eris, kconserv, out=None, ki=0, ka=0):
    nkpts, nocc, nvir = t1.shape
    shape_Wakic = [1, nkpts, 1] + eris.voov.shape[3:]
    Wakic = _new(shape_Wakic, t1.dtype, out)
    for kk in range(nkpts):
        voov_i  = einsum('akdc,id->akic',eris.vovv[ka,kk,ki],t1[ki])
        voov_i -= einsum('lkic,la->akic',eris.ooov[ka,kk,ki],t1[ka])
        voov_i += eris.voov[ka,kk,ki]

        kc = kconserv[ka,ki,kk]

        kd = kconserv[ka,kc,kk]
        tau = t2[:,ki,ka].copy()
        tau[ka] += 2*einsum('id,la->liad',t1[kd],t1[ka])
        
        oovv_tmp = np.array(eris.oovv[kk,:,kc])
        voov_i -= 0.5*einsum('xklcd,xliad->akic',oovv_tmp,tau)

        Soovv_tmp = 2*oovv_tmp - eris.oovv[:,kk,kc].transpose(0,2,1,3,4)
        voov_i[ki] += 0.5*einsum('xklcd,xilad->akic',Soovv_tmp,t2[ki,:,ka])

        Wakic[ka,kk,ki] = voov_i
    return Wakic

def cc_Wvovo(t1, t2, eris, kconserv, out=None, ki=0, ka=0):
    nkpts, nocc, nvir = t1.shape
    shape_Wakci = [1, nkpts, nkpts, nvir, nocc, nvir, nocc]
    Wakci = _new(shape_Wakci, t1.dtype, out)

    for kk in range(nkpts):
        kc = kconserv[kk,ki,ka]
        vovo  = einsum('akcd,id->akci',eris.vovv[ka,kk,kc],t1[ki])
        vovo -= einsum('klic,la->akci',eris.ooov[kk,ka,ki],t1[ka])
        vovo += np.asarray(eris.ovov[kk,ka,ki]).transpose(1,0,3,2)

        oovvf = eris.oovv[:,kk,kc].reshape(nkpts*nocc,nocc,nvir,nvir)
        t2f   = t2[:,ki,ka].copy() #This is a tau like term

        kd = kconserv[ka,kc,kk]
        t2f[ka] += 2*einsum('id,la->liad',t1[kd],t1[ka])
        t2f = t2f.reshape(nkpts*nocc,nocc,nvir,nvir)

        vovo -= 0.5*einsum('lkcd,liad->akci',oovvf,t2f)
        Wakci[ka,kk,kc] = vovo
    return Wakci

def Wooov(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wklid = _new(eris.ooov.shape, t1.dtype, out)
    for kk in range(nkpts):
        for kl in range(nkpts):
            for ki in range(nkpts):
                ooov = einsum('ic,klcd->klid',t1[ki],eris.oovv[kk,kl,ki])
                ooov += eris.ooov[kk,kl,ki]
                Wklid[kk,kl,ki] = ooov
    return Wklid

def Wvovv(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Walcd = _new(eris.vovv.shape, t1.dtype, out)
    for ka in range(nkpts):
        for kl in range(nkpts):
            for kc in range(nkpts):
                vovv = einsum('ka,klcd->alcd', -t1[ka], eris.oovv[ka,kl,kc])
                vovv += eris.vovv[ka,kl,kc]
                Walcd[ka,kl,kc] = vovv
    return Walcd

def W1ovvo(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wkaci = _new((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), t1.dtype, out)
    for kk in range(nkpts):
        for ka in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kk,kc,ka]
                # ovvo[kk,ka,kc,ki] => voov[ka,kk,ki,kc]
                ovvo = np.asarray(eris.voov[ka,kk,ki]).transpose(1,0,3,2).copy()
                for kl in range(nkpts):
                    kd = kconserv[ki,ka,kl]
                    St2 = 2.*t2[ki,kl,ka] - t2[kl,ki,ka].transpose(1,0,2,3)
                    ovvo +=  einsum('klcd,ilad->kaci',eris.oovv[kk,kl,kc],St2)
                    ovvo += -einsum('kldc,ilad->kaci',eris.oovv[kk,kl,kd],t2[ki,kl,ka])
                Wkaci[kk,ka,kc] = ovvo
    return Wkaci

def W2ovvo(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wkaci = _new((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), t1.dtype, out)
    WWooov = Wooov(t1,t2,eris,kconserv)
    for kk in range(nkpts):
        for ka in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kk,kc,ka]
                ovvo =  einsum('la,lkic->kaci',-t1[ka],WWooov[ka,kk,ki])
                ovvo += einsum('akdc,id->kaci',eris.vovv[ka,kk,ki],t1[ki])
                Wkaci[kk,ka,kc] = ovvo
    return Wkaci

def Wovvo(t1, t2, eris, kconserv, out=None):
    Wovvo = W1ovvo(t1, t2, eris, kconserv, out)
    for k, w2 in enumerate(W2ovvo(t1, t2, eris, kconserv)):
        Wovvo[k] = Wovvo[k] + w2
    return Wovvo

def W1ovov(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wkbid = _new(eris.ovov.shape, t1.dtype, out)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kd = kconserv[kk,ki,kb]
                #   kk + kl - kc - kd = 0
                # => kc = kk - kd + kl
                ovov = eris.ovov[kk,kb,ki].copy()
                for kl in range(nkpts):
                    kc = kconserv[kk,kd,kl]
                    ovov -= einsum('klcd,ilcb->kbid',eris.oovv[kk,kl,kc],t2[ki,kl,kc])
                Wkbid[kk,kb,ki] = ovov
    return Wkbid

def W2ovov(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wkbid = _new((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), t1.dtype, out)
    WWooov = Wooov(t1,t2,eris,kconserv)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kd = kconserv[kk,ki,kb]
                ovov = einsum('klid,lb->kbid',WWooov[kk,kb,ki],-t1[kb])
                ovov += einsum('bkdc,ic->kbid',eris.vovv[kb,kk,kd],t1[ki])
                Wkbid[kk,kb,ki] = ovov
    return Wkbid

def Wovov(t1, t2, eris, kconserv, out=None):
    Wovov = W1ovov(t1, t2, eris, kconserv, out)
    for k, w2 in enumerate(W2ovov(t1, t2, eris, kconserv)):
        Wovov[k] = Wovov[k] + w2
    return Wovov

def Woooo(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wklij = _new(eris.oooo.shape, t1.dtype, out)
    for kk in range(nkpts):
        for kl in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kl]
                oooo  = einsum('klcd,ic,jd->klij',eris.oovv[kk,kl,ki],t1[ki],t1[kj])
                oooo += einsum('klid,jd->klij',eris.ooov[kk,kl,ki],t1[kj])
                oooo += einsum('lkjc,ic->klij',eris.ooov[kl,kk,kj],t1[ki])
                oooo += eris.oooo[kk,kl,ki]
                for kc in range(nkpts):
                    #kd = kconserv[kk,kc,kl]
                    oooo += einsum('klcd,ijcd->klij',eris.oovv[kk,kl,kc],t2[ki,kj,kc])
                Wklij[kk,kl,ki] = oooo
    return Wklij

def Wvvvv(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wabcd = _new((nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), t2.dtype, out)
    for ka in range(nkpts):
        for kb in range(nkpts):
            for kc in range(nkpts):
                Wabcd[ka,kb,kc] = get_Wvvvv(t1, t2, eris, kconserv, ka, kb, kc)
    return Wabcd

def get_Wvvvv(t1, t2, eris, kconserv, ka, kb, kc):
    kd = kconserv[ka, kc, kb]
    nkpts, nocc, nvir = t1.shape
    if getattr(eris, 'Lpv', None) is not None:
        # Using GDF to generate Wvvvv on the fly
        Lpv = eris.Lpv
        Lac = (Lpv[ka,kc][:,nocc:] -
               einsum('Lkc,ka->Lac', Lpv[ka,kc][:,:nocc], t1[ka]))
        Lbd = (Lpv[kb,kd][:,nocc:] -
               einsum('Lkd,kb->Lbd', Lpv[kb,kd][:,:nocc], t1[kb]))
        vvvv = einsum('Lac,Lbd->abcd', Lac, Lbd)
        vvvv *= (1. / nkpts)
    else:
        vvvv  = einsum('klcd,ka,lb->abcd',eris.oovv[ka,kb,kc],t1[ka],t1[kb])
        vvvv += einsum('alcd,lb->abcd',eris.vovv[ka,kb,kc],-t1[kb])
        vvvv += einsum('bkdc,ka->abcd',eris.vovv[kb,ka,kd],-t1[ka])
        vvvv += eris.vvvv[ka,kb,kc]

    for kk in range(nkpts):
        kl = kconserv[kc,kk,kd]
        vvvv += einsum('klcd,klab->abcd', eris.oovv[kk,kl,kc], t2[kk,kl,ka])
    return vvvv

def Wvvvo(t1, t2, eris, kconserv, _Wvvvv=None, out=None):
    nkpts, nocc, nvir = t1.shape
    Wabcj = _new((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc), t1.dtype, out)
    WW1ovov = W1ovov(t1,t2,eris,kconserv)
    WW1ovvo = W1ovvo(t1,t2,eris,kconserv)
    FFov = cc_Fov(t1,t2,eris,kconserv)
    for ka in range(nkpts):
        for kb in range(nkpts):
            for kc in range(nkpts):
                kj = kconserv[ka,kc,kb]
                # Wvovo[ka,kl,kc,kj] <= Wovov[kl,ka,kj,kc].transpose(1,0,3,2)
                vvvo  = einsum('alcj,lb->abcj',WW1ovov[kb,ka,kj].transpose(1,0,3,2),-t1[kb])
                vvvo += einsum('kbcj,ka->abcj',WW1ovvo[ka,kb,kc],-t1[ka])
                # vvvo[ka,kb,kc,kj] <= vovv[kc,kj,ka,kb].transpose(2,3,0,1).conj()
                vvvo += np.asarray(eris.vovv[kc,kj,ka]).transpose(2,3,0,1).conj()

                for kl in range(nkpts):
                    # ka + kl - kc - kd = 0
                    # => kd = ka - kc + kl
                    kd = kconserv[ka,kc,kl]
                    St2 = 2.*t2[kl,kj,kd] - t2[kl,kj,kb].transpose(0,1,3,2)
                    vvvo += einsum('alcd,ljdb->abcj',eris.vovv[ka,kl,kc], St2)
                    vvvo += einsum('aldc,ljdb->abcj',eris.vovv[ka,kl,kd], -t2[kl,kj,kd])
                    # kb - kc + kl = kd
                    kd = kconserv[kb,kc,kl]
                    vvvo += einsum('bldc,jlda->abcj',eris.vovv[kb,kl,kd], -t2[kj,kl,kd])

                    # kl + kk - kb - ka = 0
                    # => kk = kb + ka - kl
                    kk = kconserv[kb,kl,ka]
                    vvvo += einsum('lkjc,lkba->abcj',eris.ooov[kl,kk,kj],t2[kl,kk,kb])
                vvvo += einsum('lkjc,lb,ka->abcj',eris.ooov[kb,ka,kj],t1[kb],t1[ka])
                vvvo += einsum('lc,ljab->abcj',-FFov[kc],t2[kc,kj,ka])
                Wabcj[ka,kb,kc] = vvvo

    # Check if t1=0 (HF+MBPT(2))
    # einsum will check, but don't make vvvv if you can avoid it!
    if np.any(t1 != 0):
        for ka in range(nkpts):
            for kb in range(nkpts):
                for kc in range(nkpts):
                    kj = kconserv[ka,kc,kb]
                    if _Wvvvv is None:
                        Wvvvv = get_Wvvvv(t1, t2, eris, kconserv, ka, kb, kc)
                    else:
                        Wvvvv = _Wvvvv[ka, kb, kc]
                    Wabcj[ka,kb,kc] = (Wabcj[ka,kb,kc] +
                                       einsum('abcd,jd->abcj', Wvvvv, t1[kj]))
    return Wabcj

def Wovoo(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape

    WW1ovov = W1ovov(t1,t2,eris,kconserv)
    WWoooo = Woooo(t1,t2,eris,kconserv)
    WW1ovvo = W1ovvo(t1,t2,eris,kconserv)
    FFov = cc_Fov(t1,t2,eris,kconserv)

    Wkbij = _new((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), t1.dtype, out)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kb]
                ovoo  = einsum('kbid,jd->kbij',WW1ovov[kk,kb,ki], t1[kj])
                ovoo += einsum('klij,lb->kbij',WWoooo[kk,kb,ki],-t1[kb])
                ovoo += einsum('kbcj,ic->kbij',WW1ovvo[kk,kb,ki],t1[ki])
                ovoo += np.array(eris.ooov[ki,kj,kk]).transpose(2,3,0,1).conj()

                for kd in range(nkpts):
                    # kk + kl - ki - kd = 0
                    # => kl = ki - kk + kd
                    kl = kconserv[ki,kk,kd]
                    St2 = 2.*t2[kl,kj,kd] - t2[kj,kl,kd].transpose(1,0,2,3)
                    ovoo += einsum('klid,ljdb->kbij',  eris.ooov[kk,kl,ki], St2)
                    ovoo += einsum('lkid,ljdb->kbij', -eris.ooov[kl,kk,ki],t2[kl,kj,kd])
                    kl = kconserv[kb,ki,kd]
                    ovoo += einsum('lkjd,libd->kbij', -eris.ooov[kl,kk,kj],t2[kl,ki,kb])

                    # kb + kk - kd = kc
                    #kc = kconserv[kb,kd,kk]
                    ovoo += einsum('bkdc,jidc->kbij',eris.vovv[kb,kk,kd],t2[kj,ki,kd])
                ovoo += einsum('bkdc,jd,ic->kbij',eris.vovv[kb,kk,kj],t1[kj],t1[ki])
                ovoo += einsum('kc,ijcb->kbij',FFov[kk],t2[ki,kj,kk])
                Wkbij[kk,kb,ki] = ovoo
    return Wkbij

def _new(shape, dtype, out):
    if out is None: # Incore:
        out = np.empty(shape, dtype=dtype)
    else:
        assert (out.shape == shape)
        assert (out.dtype == dtype)
    return out
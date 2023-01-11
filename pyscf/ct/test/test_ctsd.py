import tempfile
import unittest
import numpy as np
from pyscf import gto, scf, lib
from pyscf import ct, mp, ao2mo
from pyscf.ct import ctsd
from pyscf import cc
from pyscf.cc import ccsd


def setUpModule():
    global mol, mf, myct
    mol = gto.Mole()
    mol.verbose = 7
    #mol.output = '/dev/null'
    #mol.atom = '''
    #    N  0.  0 0;
    #    N  0.  0 1.09768;
    #    '''
    mol.atom = '''
        O    0.000000    0.000000    0.117790
        H    0.000000    0.755453   -0.471161
        H    0.000000   -0.755453   -0.471161'''
    #mol.atom = '''
    #    H    0.000000   0   0
    #    H    0.000000   0   2.0'''
    mol.unit = 'A'
    mol.basis = 'ccpvdz'
    mol.build()
    mf = scf.RHF(mol)
    mf.chkfile = tempfile.NamedTemporaryFile().name
    mf.conv_tol_grad = 1e-8
    mf.kernel()

    # active space is set to 0. The reference energy of the
    # CT Ham should reproduce MP2 energy
    myct = ctsd.CTSD(mf, a_nmo=0)


def tearDownModule():
    global mol, mf, myct
    mol.stdout.close()


class KnownValues(unittest.TestCase):

    def test_d0(self):
        np.random.seed(1)
        dim = 2
        dm1 = np.diag(np.ones(dim))
        dm2 = np.ones([dim, dim, dim, dim])
        d0_eval = np.zeros(dm2.shape)
        d0_eval[0, 0, 0, 0] = 4./3
        d0_eval[0, 1, 0, 1] = 4./3
        d0_eval[1, 0, 1, 0] = 4./3
        d0_eval[1, 1, 1, 1] = 4./3
        d0_eval[0, 0, 0, 0] -= 4./3 * 1./2
        d0_eval[0, 1, 1, 0] -= 4./3 * 1./2
        d0_eval[1, 0, 0, 1] -= 4./3 * 1./2
        d0_eval[1, 1, 1, 1] -= 4./3 * 1./2
        d0_eval -= dm2

        d0 = ctsd.get_d_zero(dm1, dm2)
        self.assertAlmostEqual(lib.fp(d0_eval), lib.fp(d0), 11)
        # random dm
        dim = 10
        dm1 = np.random.random([dim, dim])
        dm2 = np.random.random([dim, dim, dim, dim])

        d0 = ctsd.get_d_zero(dm1, dm2)
        self.assertAlmostEqual(lib.fp(d0), 11.82033156712288, 11)

    def test_d_bar(self):
        np.random.seed(2)
        # dimension = 2
        dim = 2
        dm1 = np.diag(np.ones(dim))
        dm2 = np.ones([dim, dim, dim, dim])
        d_bar_eval = np.zeros(dm2.shape)
        d_bar_eval[0, 0, 0, 0] = 2.
        d_bar_eval[0, 1, 0, 1] = 2.
        d_bar_eval[1, 0, 1, 0] = 2.
        d_bar_eval[1, 1, 1, 1] = 2.
        d_bar_eval[0, 0, 0, 0] -= 2 * 1./2
        d_bar_eval[0, 1, 1, 0] -= 2 * 1./2
        d_bar_eval[1, 0, 0, 1] -= 2 * 1./2
        d_bar_eval[1, 1, 1, 1] -= 2 * 1./2
        d_bar_eval -= dm2

        d_bar = ctsd.get_d_bar(dm1, dm2)
        self.assertAlmostEqual(lib.fp(d_bar_eval), lib.fp(d_bar), 11)
        # random dm
        dim = 10
        dm1 = np.random.random([dim, dim])
        dm2 = np.random.random([dim, dim, dim, dim])

        d_bar = ctsd.get_d_zero(dm1, dm2)
        self.assertAlmostEqual(lib.fp(d_bar), -9.64328148551661, 11)

    def test_get_d3_slice(self):
        # for dm obtained from HF wavefunction, slices of d3 containing
        # indices other than the occupied (core) will be 0.
        nmo = 5
        dm1 = np.diag(np.ones(nmo)*2)
        dm2 = np.zeros([nmo, nmo, nmo, nmo])
        slices = [0, nmo, 0, nmo, 0, nmo, 1, nmo, 2, nmo, 3, nmo]
        d3 = ctsd.get_d3_slice_direct(dm1, dm2)

    def test_get_c1(self):
        t1 = np.ones((myct.e_nmo, myct.t_nmo))
        c1 = myct.get_c1(t=t1)
        pass

    
    def test_get_c1_prime(self):
        #c1_prime = myct.get_c1_prime()
        pass

    def test_get_c2_prime(self):
        pass

    def test_get_c2_dprime(self):
        pass

    def test_mp2_amps(self):
        t1, t2 = myct.get_mp2_amps()
        t2_xyij = t2["xyij"]
        mymp2 = mp.MP2(mf)
        eris = mymp2.ao2mo(mymp2.mo_coeff)
        # assert mo energies are the same
        assert np.array_equal(myct.mf.mo_energy, eris.mo_energy)

        # assert eri are the same
        # ct eri is in physicsts format
        ct_ovov = myct.eri[:myct.c_nmo, :myct.c_nmo, myct.c_nmo:,
                           myct.c_nmo:].transpose((0, 2, 1, 3))
        nocc = mymp2.nocc
        nvir = mymp2.nmo - nocc
        mp2_ovov = eris.ovov.reshape(nocc, nvir, nocc, nvir)
        assert np.allclose(ct_ovov, mp2_ovov)

        mo_e_i = myct.mf.mo_energy[:myct.c_nmo]
        mo_e_x = myct.mf.mo_energy[myct.t_nmo:]

        ct_eeoo = myct.eri[myct.t_nmo:, myct.t_nmo:, :myct.c_nmo,
                  :myct.c_nmo]
        e_xi = -(mo_e_x[:, None] - mo_e_i[None, :])
        t_xyij = ct_eeoo/lib.direct_sum(
            "xi+yj->xyij", e_xi, e_xi)

        # use mp2 algo from pyscf to test
        mp2_e, t2_mp2 = mymp2.kernel(with_t2=True)


        shift_ind = myct.a_nmo
        t2_mp2_xyij = t2_mp2[:, :, shift_ind:,
                      shift_ind:].transpose((2, 3, 0, 1))
        # test symmetries in t2_mp2
        assert np.allclose(t2_mp2_xyij, t2_mp2_xyij.transpose((1, 0, 3, 2)))
        assert np.allclose(t_xyij, t2_mp2_xyij, atol=1e-7)
        assert np.allclose(t2_xyij, t2_mp2_xyij, atol=1e-7)


    def test_get_c0(self):
        # for dm obtained from HF wavefunction, slices of d3 containing
        # indices other than the occupied (core) will be 0.
        # so c0 = 0
        myct.kernel()
        print("Using amps_algo = ", myct.amps_algo)
        ct_0 = myct.get_c0()
        assert ct_0 == 0.

    def test_zero_amp_ct_ref_energy(self):
        # if the CT amplitudes are 0, then the transformation has no effect,
        # one should get the original HF energy and mo_energy

        # get  mf hcore in mo
        hcore = myct.ao2mo(myct.mf.get_hcore())

        myct.amps_algo = "zero"
        c0, h1, v2 = myct.kernel()
        assert np.allclose(hcore, h1)
        # get original eri from mf and mo_coeff
        eri = ao2mo.full(mf._eri, mf.mo_coeff)
        eri = ao2mo.restore(1, eri, myct.nmo).transpose((0, 2, 1, 3))
        assert np.allclose(eri, v2)
        mo_energy_mf = mf.mo_energy

        mo_energy = myct.mo_energy
        assert np.allclose(mo_energy_mf, mo_energy)
        e_hf_ct = myct.get_hf_energy()
        e_hf = mf.e_tot
        self.assertAlmostEqual(e_hf_ct, e_hf, 8)

    def test_get_mo_energy(self):
        c0, c1, c2 = myct.kernel()
        ct_hf_e = myct.get_hf_energy(c0, c1, c2)
        print("CT HF energy = ", ct_hf_e)

        mo_energy = mf.mo_energy.copy()
        ct_mo_energy = myct.get_mo_energy()
        print("HF HOMO-LOMO gap = ", 
              mo_energy[myct.c_nmo+1]-mo_energy[myct.c_nmo])
        print("CT HOMO-LOMO gap = ", 
              ct_mo_energy[myct.c_nmo+1]-ct_mo_energy[myct.c_nmo])
        fock = myct.get_fock()
        assert np.allclose(fock.diagonal(), ct_mo_energy)

        hf_e_from_fock = 2. * np.sum(fock[:myct.c_nmo, :myct.c_nmo].diagonal())
        hf_e_from_fock -= 2. * np.einsum("ijij", c2[:myct.c_nmo, :myct.c_nmo,
                                             :myct.c_nmo, :myct.c_nmo])
        hf_e_from_fock += np.einsum("ijji", c2[:myct.c_nmo, :myct.c_nmo,
                                             :myct.c_nmo, :myct.c_nmo])
        hf_e_from_fock += mf.energy_nuc()
        hf_e = myct.get_hf_energy()
        self.assertAlmostEqual(hf_e, hf_e_from_fock, 8)
        
    def test_ct_ref(self):
        #mf = scf.RHF(mol).run()
        mymp = mp.MP2(mf).run()
        mp2_total = mymp.e_tot
        myct.amps_algo = "mp2"
        c0, c1, c2 = myct.kernel()
        ct_hf_e = myct.get_hf_energy(c0, c1, c2)
        print("CT HF energy = ", ct_hf_e)

        ct_mo_energy = myct.get_mo_energy()
        e_ia = -(ct_mo_energy[myct.c_nmo:myct.t_nmo, None] \
                - ct_mo_energy[None, :myct.c_nmo])
        ct_vvoo = c2[myct.c_nmo:myct.t_nmo, myct.c_nmo:myct.t_nmo, 
                     :myct.c_nmo, :myct.c_nmo]
        ct_oovv = c2[:myct.c_nmo, :myct.c_nmo, 
                     myct.c_nmo:myct.t_nmo, myct.c_nmo:myct.t_nmo]
        ct_t2 = ct_vvoo / lib.direct_sum("ai+bj -> abij", e_ia, e_ia)
        ct_mp2_e = 2. * lib.einsum("abij, ijab -> ", ct_t2, ct_oovv)
        ct_mp2_e -= lib.einsum("abji, ijab -> ", ct_t2, ct_oovv)
        ct_mp2_total = ct_hf_e + ct_mp2_e
        print("CT MP2 corr = ", ct_mp2_e)
        print("CT MP2 total = ", ct_mp2_total)
        self.assertAlmostEqual(ct_mp2_total, mp2_total, 8)

    def test_per_term_analysis(self):
        c0, c1, c2 = myct.kernel()
        ct_hf_e = myct.get_hf_energy(c0, c1, c2)
        print("CT HF energy = ", ct_hf_e)
        c1 = myct.get_c1() 
        print("="*79)
        print("Per term analysis")
        print("="*79)
        print("***** c1")
        ctsd.tensor_analysis(c1)

    
        c1_prime = myct.get_c1_prime()
        print("***** c1_prime")
        ctsd.tensor_analysis(c1_prime)

        c2 = myct.get_c2()
        print("***** c2")
        ctsd.tensor_analysis(c2)
        print("***** c2 occ-occ-occ-occ block")
        ctsd.tensor_analysis(c2[:myct.nocc, :myct.nocc, :myct.nocc, :myct.nocc])

        c2_prime = myct.get_c2_prime()
        print("***** c2_prime")
        ctsd.tensor_analysis(c2_prime)
        print("***** c2_prime occ-occ-occ-occ block")
        ctsd.tensor_analysis(c2_prime[:myct.nocc, :myct.nocc, :myct.nocc, :myct.nocc])

        c2_dprime = myct.get_c2_dprime()
        print("***** c2_dprime")
        ctsd.tensor_analysis(c2_dprime)
        print("***** c2_dprime occ-occ-occ-occ block")
        ctsd.tensor_analysis(c2_dprime[:myct.nocc, :myct.nocc, :myct.nocc, :myct.nocc])

        # The following [o1, t] gives rise to 1- and 2-body terms
        fock_nm = myct.mf.get_fock()
        fock_nm = myct.ao2mo(fock_nm)
        c0_f, c1_f, c2_f = myct.commute(*myct.commute(o1=fock_nm))
        print("***** c1_f")
        ctsd.tensor_analysis(c1_f)
        print("***** c2_f")
        ctsd.tensor_analysis(c2_f)
        print("***** c2_f occ-occ-occ-occ block")
        ctsd.tensor_analysis(c2_f[:myct.nocc, :myct.nocc, :myct.nocc, :myct.nocc])
        print("***** Per term analysis complete *****")

        print("***** Reproducing MP2 energy from these terms *****")
        nocc = myct.nocc

        e_hf = myct.mf.e_tot

        e_mp2 = 0.
        de = 0.
        e_mp2 += e_hf

        de += 2.*np.einsum("ii ->", c1[:nocc, :nocc])
        de += 2.*np.einsum("ii ->", c1_prime[:nocc, :nocc])
        e_mp2 += de
        print("***** c1 contribution = ", de)

        de = 0.
        de += 2.*np.einsum("ijij ->", c2[:nocc, :nocc, :nocc, :nocc])
        de -= np.einsum("ijji ->", c2[:nocc, :nocc, :nocc, :nocc])
        e_mp2 += de
        print("***** c2 contribution = ", de)

        de = 0.
        de += 2.*np.einsum("ijij ->", c2_prime[:nocc, :nocc, :nocc, :nocc])
        de -= np.einsum("ijji ->", c2_prime[:nocc, :nocc, :nocc, :nocc])
        e_mp2 += de
        print("***** c2_prime contribution = ", de)

        de = 0.
        de += 2.*np.einsum("ijij ->", c2_dprime[:nocc, :nocc, :nocc, :nocc])
        de -= np.einsum("ijji ->", c2_dprime[:nocc, :nocc, :nocc, :nocc])
        e_mp2 += de
        print("***** c2_dprime contribution = ", de)

        c0_f /= 2.
        c1_f /= 2.
        c2_f /= 2.

        de = c0_f
        e_mp2 += de
        print("***** c0_f contribution = ", de)

        de = 0.
        de += 2.*np.einsum("ii ->", c1_f[:nocc, :nocc])
        e_mp2 += de
        print("***** c1_f contribution = ", de)

        de = 0.
        de += 2.*np.einsum("ijij ->", c2_f[:nocc, :nocc, :nocc, :nocc])
        de -= np.einsum("ijji ->", c2_f[:nocc, :nocc, :nocc, :nocc])
        e_mp2 += de
        print("***** c2_f contribution = ", de)
        print("***** Reproduced e_mp2 = ", e_mp2)
        self.assertAlmostEqual(e_mp2, ct_hf_e)


    def test_canonicalize(self):
        myct.kernel()
        ctsd.tensor_analysis(myct.ct_o2)
        ct_mo_energy = myct.get_mo_energy()
        print("Before canonicalization CT HOMO-LOMO gap = ", 
              ct_mo_energy[myct.c_nmo+1]-ct_mo_energy[myct.c_nmo])
        mo_coeff = myct.canonicalize()
        ct_fock = myct.get_fock()
        #ct_fock = myct.ao2mo(ct_fock, mo_coeff)
        #ct_eri  = myct.ao2mo(myct.ct_o2, mo_coeff, to_phy=False)
        print("***** CT Fock matrix *****")
        print(ct_fock)
        ctsd.tensor_analysis(ct_fock)
        ct_mo_energy = myct.get_mo_energy()
        print("After canonicalization CT HOMO-LOMO gap = ", 
              ct_mo_energy[myct.c_nmo+1]-ct_mo_energy[myct.c_nmo])

        

    def test_ct_mp2(self):
        myct.amps_algo = "mp2"
        c0, c1, c2 = myct.kernel()
        ct_hf_e = myct.get_hf_energy(c0, c1, c2)
        print("CT HF energy = ", ct_hf_e)
        eris = myct.create_eris()
        mymp = mp.MP2(mf)
        mymp.kernel(eris=eris)
        mycc = ccsd.CCSD(mf)
        mycc.kernel()
        print("CT-MP2 corr = ", mymp.e_corr)
        print("CT-MP2 total = ", ct_hf_e + mymp.e_corr)
        print("CCSD corr = ", mycc.e_corr)


    def test_ct_ccsd(self):
        myct.amps_algo = "mp2"
        myct.kernel()
        myct.canonicalize()
        ct_hf_e = myct.get_hf_energy()
        print("CT HF energy = ", ct_hf_e)
        
        eris = myct.create_eris()
        mycc = cc.GCCSD(mf)
        mycc.max_cycle = 1
        mycc.kernel()
        can_mp2_e = mycc.emp2 + mycc.e_hf
        can_cc_e = mycc.e_tot
        myct_cc = ccsd.CCSD(mf)
        myct_cc.max_cycle = 1
        myct_cc.kernel(eris=eris)
        ct_cc_e = myct_cc.e_tot
        ct_mp2_e = myct_cc.emp2 + ct_hf_e
        print("Canonical MP2 e tot = ", can_mp2_e)
        print("Canonical CCSD e tot = ", can_cc_e)
        print("CT HF e = ", ct_hf_e)
        print("CT MP2 e tot = ", ct_mp2_e)
        print("CT CCSD e tot = ", ct_cc_e)
        print("END")


if __name__ == "__main__":
    print("Full Tests for CT")
    unittest.main()

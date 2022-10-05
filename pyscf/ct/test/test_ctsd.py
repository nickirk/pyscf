import tempfile
import unittest
import numpy as np
from functools import reduce
from pyscf import gto, scf, lib
from pyscf import ct, mp
from pyscf.ct import ctsd

def setUpModule():
    global mol, mf, myct
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol)
    mf.chkfile = tempfile.NamedTemporaryFile().name
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    e_hf_orig = mf.e_tot
    myct = ctsd.CTSD(mf)
    c0, h1, v2 = myct.kernel()
    mo_energy = myct.get_mo_energy()
    mo_energy_mf = mf.mo_energy
    e_hf_ct = myct.get_hf_energy(c0, h1, v2)
    diff = e_hf_orig - e_hf_ct


def tearDownModule():
    global mol, mf, myct
    mol.stdout.close()
    del mol, mf, myct

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


        shift_ind = myct.v_nmo
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
        ct_0 = myct.get_c0()
        assert ct_0 == 0.



    def test_get_c1(self):
        pass

    def test_get_c1_prime(self):
        pass

    def test_get_c2(self):
        pass

    def test_get_c2_prime(self):
        pass
    def test_get_c2_dprime(self):
        pass
    def test_ct(self):
        mf = scf.ROHF(mol).run()
        myct = ct.ctsd.CTSD(mf).run()

    def test_get_hf_energy(self):
        e_hf = 0.
        pass

    def test_ct_mp2(self):
        mf = scf.ROHF(mol).run()
        mp2_e = mp.MP2(mf).run()
        myct = ct.ctsd.CTSD(mf).run()

if __name__ == "__main__":
    print("Full Tests for CT")
    unittest.main()

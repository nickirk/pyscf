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

    myct = ctsd.CTSD(mf)
    c0, h1, v2 = myct.kernel()

def tearDownModule():
    global mol, mf, myct
    mol.stdout.close()
    del mol, mf, myct

class KnownValues(unittest.TestCase):

    def test_d0(self):
        np.random.seed(1)
        # dimension = 2
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

    def test_dm3(self):
        pass

    def test_mp2_amps(self):
        # Are the signs of the amplitudes consistently defined between refs?
        t1, t2 = myct.get_mp2_amps()
        # use mp2 algo from pyscf to test
        mp2_e, t2_mp2 = mp.MP2(mf).kernel(with_t2=True)

        shift_ind = myct.v_nmo
        t2_mp2_xyij = t2_mp2[:, :, shift_ind:,
                      shift_ind:].transpose((2, 3, 0, 1))
        # ??? What is happenning with the mp2 amplitudes in pyscf.mp???
        # It is not symmetric?
        assert np.array_equal(t2_mp2, t2_mp2.transpose(1, 0, 3, 2))
        assert np.array_equal(t2["xyij"], t2_mp2_xyij)

        pass

    def test_get_c0(self):
        pass
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
        pass

    def test_ct_mp2(self):
        mf = scf.ROHF(mol).run()
        mp2_e = mp.MP2(mf).run()
        myct = ct.ctsd.CTSD(mf).run()

if __name__ == "__main__":
    print("Full Tests for CT")
    unittest.main()

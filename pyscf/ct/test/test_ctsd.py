import tempfile
import unittest
import numpy
from functools import reduce
from pyscf import gto, scf
from pyscf import ct
from pyscf.ct import ctsd

def setUpModule():
    global mol, mf, eris, myct
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
    global mol, mf, eris, myct
    mol.stdout.close()
    del mol, mf, eris, myct

class KnownValues(unittest.TestCase):
    def test_ct(self):
        mf = scf.ROHF(mol).run()
        myct = ct.ctsd.CTSD(mf).run()

if __name__ == "__main__":
    print("Full Tests for CT")
    unittest.main()

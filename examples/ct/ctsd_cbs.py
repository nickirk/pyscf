import numpy as np
from pyscf import gto, scf, lib
from pyscf import ct, mp, ao2mo
from pyscf.ct import ctsd
from pyscf.cc import ccsd
from pyscf.mp import mp2

def set_up_mole(basis):
    mol = gto.Mole()
    mol.verbose = 3
    #mol.atom = '''
    #    O    0.000000    0.000000    0.117790
    #    H    0.000000    0.755453   -0.471161
    #    H    0.000000   -0.755453   -0.471161'''
    #mol.atom = '''
    #    Li    0.000000    0.000000    0.00000
    #    H    0.000000    0.0000000    1.5957
    #    '''
    mol.atom = '''Ne 0 0 0'''

    mol.basis = basis 
    mol.build()
    mol.unit = 'A'
    return mol



def main():

    ecc = []
    ct_ecc = []
    nmos = []
    for basis in ['aug-ccpvdz', 'ccpvtz', 'ccpvqz', 'ccpv5z']:
    #for basis in ['sto6g']:
        mol = set_up_mole(basis=basis)
        mf = scf.RHF(mol)
        mf.conv_tol_grad = 1e-8
        mf.kernel()

        nmo = len(mf.mo_energy)
        nmos.append(nmo)
        nocc = int(np.sum(mf.mo_occ))
        #for a_nmo in range(0, nmo-nocc, 5):
        for a_nmo in [0]:
            myct = ctsd.CTSD(mf, a_nmo=a_nmo)

            c0, c1, c2 = myct.kernel()
            ct_hf_e = myct.get_hf_energy(c0, c1, c2)
            print("CT HF energy = ", ct_hf_e)

            # calculate CCSD with CT-Hamiltonian
            mycc = ccsd.CCSD(mf)
            mycc.kernel()
            tot_emp2 = mycc.emp2+mycc.e_hf 
            assert np.isclose(tot_emp2, ct_hf_e, 7)
            ecc.append(mycc.e_tot)
            eris = mycc.ao2mo()
            eris = myct.create_eris(eris=eris)
            mymp2 = mp2.MP2(mf)
            mymp2.kernel(eris=eris)
            #mycc.kernel(eris=eris)
            #assert np.isclose(mymp2.e_corr, mycc.emp2, 7)
            #ct_ecc.append(mycc.e_tot)
        print("ecc = ", ecc)
        print("nmo = ", nmos)
        print("ct_ecc = ", ct_ecc)

    return



if __name__ == "__main__":
    main()
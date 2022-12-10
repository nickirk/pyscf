import numpy as np
from pyscf import gto, scf, lib
from pyscf import ct, mp, ao2mo
from pyscf.ct import ctsd
from pyscf.cc import ccsd
from pyscf.mp import mp2

def set_up_mole(basis, sep=1.0):
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
    mol.atom = '''H 0 0 0
                  H 0 0 ''' + str(sep)

    mol.basis = basis 
    mol.build()
    mol.unit = 'A'
    return mol



def main():

    ecc = []
    emp2 = []
    ct_ecc = []
    ct_emp2 = []
    nmos = []
    #for basis in ['ccpvdz', 'ccpvtz', 'ccpvqz', 'ccpv5z']:
    for basis in ['sto6g']:
        for sep in np.arange(0.6, 3.0, 0.2):
            mol = set_up_mole(basis=basis, sep=sep)
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
                emp2.append(tot_emp2)
                assert np.isclose(tot_emp2, ct_hf_e, 7)
                ecc.append(mycc.e_tot)
                eris = mycc.ao2mo()
                eris = myct.create_eris(eris=eris)
                mymp2 = mp2.MP2(mf)
                mymp2.kernel(eris=eris)
                ct_emp2.append(mymp2.e_tot)
                mycc.kernel(eris=eris)
                assert np.isclose(mymp2.e_corr, mycc.emp2, 7)
                ct_ecc.append(mycc.e_tot)
            print("nmo = ", nmos)
            print("emp2 = ", emp2)
            print("ecc = ", ecc)
            print("ct_mp2 = ", ct_emp2)
            print("ct_ecc = ", ct_ecc)

    return



if __name__ == "__main__":
    main()
import numpy as np
from pyscf import gto, scf, lib
from pyscf import ct, mp, ao2mo
from pyscf.ct import ctsd
from pyscf.cc import ccsd

def set_up_mole(basis):
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = '''
        O    0.000000    0.000000    0.117790
        H    0.000000    0.755453   -0.471161
        H    0.000000   -0.755453   -0.471161'''

    mol.basis = basis 
    mol.build()
    return mol



def main():

    #for basis in ['ccpvdz', 'ccpvtz', 'ccpvqz']:
    for basis in ['sto6g']:
        mol = set_up_mole(basis=basis)
        mf = scf.RHF(mol)
        mf.conv_tol_grad = 1e-8
        mf.kernel()

        nmo = len(mf.mo_energy)
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
            eris_cc = mycc.ao2mo()
            eris = myct.create_eris(eris=eris)
            mycc.kernel(eris=eris)

    return



if __name__ == "__main__":
    main()
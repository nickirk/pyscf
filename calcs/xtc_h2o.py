from pyscf import gto, scf, cc

mol = gto.M(atom="C 0 0 0; O 1.128323 0 0", basis=basis, unit="Angstrom") # ref: https://doi.org/10.1063/1.1527013
mf = scf.RHF(mol).run()

mycc = cc.CCSD(mf).run()
# get the ip energies
eip = mycc.ipccsd(nroots=3).run()
print("IP energies (eV):", eip.e)
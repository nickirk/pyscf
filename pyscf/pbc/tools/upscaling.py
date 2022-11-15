import numpy as np
from pyscf.pbc import gto, mp, cc
from pyscf.pbc import scf
from pyscf.pbc.tools import upscale, madelung
import h5py

def set_up_h2():
    '''
    Example calculation on H2 chain
    '''
    cell = gto.Cell()
    cell.pseudo = 'gth-pade'
    cell.basis = 'sto6g'
    cell.ke_cutoff = 50
    #cell.atom='''
    #    H 3.00   3.00   2.10
    #    H 3.00   3.00   3.90
    #    '''
    #cell.a = '''
    #    6.0   0.0   0.0
    #    0.0   6.0   0.0
    #    0.0   0.0   6.0
    #    '''
    cell.atom='''
        H 2.00   2.00   1.20
        H 2.00   2.00   2.60
        '''
    cell.a = '''
        4.0   0.0   0.0
        0.0   4.0   0.0
        0.0   0.0   4.0
        '''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()
    
    return cell
def set_up_lih():
    cell = gto.Cell()
    cell.pseudo = 'gth-pade'
    cell.basis = 'sto6g'
    cell.ke_cutoff = 50
    cell.atom='''
        Li 0.00   0.00   0.0
        H  2.0    0.     0.
        '''
    cell.a = '''
        4.0   0.0   0.0
        0.0   4.0   0.0
        0.0   0.0   4.0
        '''
    cell.unit = 'A'
    cell.verbose = 4
    cell.build()
    return cell



def write_amps(t, f_name="data.hdf5"):
    dim = len(t.shape)
    dset_name = "amps"
    if dim == 3: 
        dset_name = "t1"
    elif dim == 7:
        dset_name = "t2"
    else:
        raise ValueError("Unknown amplitudes type!")
    f = h5py.File(f_name, "w")
    print("Writing t")
    f.create_dataset(dset_name, data=t)
    f.close()
    
def set_up_diamond():
    cell = gto.Cell()
    cell.pseudo = 'gth-pade'
    cell.basis = 'gth-szv'
    cell.ke_cutoff = 100
    cell.atom = '''
        C     0.      0.      0.
        C     1.26349729, 0.7294805 , 0.51582061
        '''
    cell.a = '''
        2.52699457, 0.        , 0.
        1.26349729, 2.18844149, 0.
        1.26349729, 0.7294805 , 2.06328243
        '''
    cell.unit = 'angstrom'
    cell.verbose = 4
    cell.build()
    return cell


def main():
    
    cell = set_up_diamond()
    #cell = set_up_lih()
    #cell = set_up_h2()
    sys_name = "diamond"

    
    for nk_d in [5]:
        #nks_mf = [1, 1, nk_d]
        nks_mf = [1, 1, nk_d]
        kmf_d, kmp_d = upscale.set_up_method(cell, nks_mf)
        madelung_const = madelung(cell, kmp_d.kpts)
        # FIXME:MP2 ill behaves in Diamond
        emp2_d, t2_d = kmp_d.kernel()
        mycc_d = cc.KCCSD(kmf_d)
        mycc_d.keep_exxdiv = True
        mycc_d.max_cycle = 0
        ecc0_d, t1_cc0_d, t2_cc0_d = mycc_d.kernel()
        kmp_d.t2 = t2_cc0_d.copy()

        write_amps(t2_cc0_d, "/Users/keliao/Work/project/upscaling/"+sys_name+"_cc0_t2_nk_"+str(nk_d)+".hdf5")

        for nk_s in [1]:
            nks_mp = [nk_s, nk_s, nk_s]
            ncg = int(np.prod(np.asarray(nks_mf)/np.asarray(nks_mp)))
            kmf_s, kmp_s = upscale.set_up_method(cell, nks_mp)
            kmp_s.mo_coeff = kmp_d.mo_coeff[::ncg].copy()
            kmp_s.mo_energy = kmp_d.mo_energy[::ncg].copy()
            kmf_s.mo_coeff = kmp_d.mo_coeff[::ncg].copy()
            kmf_s.mo_energy = kmp_d.mo_energy[::ncg].copy()
            emp2_s, t2_s = kmp_s.kernel()
            dist_nm = upscale.get_nn_dist(kmf_s.kpts, kmf_d.kpts)
            #kmf_s.mo_coeff = kmf_d.mo_coeff[::ncg]
            #kmf_s.mo_energy = kmf_d.mo_coeff[::ncg]
            mycc = cc.KCCSD(kmf_s)
            mycc.max_cycle = 50
            e_cc_s, t1_s, t2_s = mycc.kernel()
            t1_us, t2_us = upscale.upscale(mycc, kmp_s, kmp_d, dist_nm)
            t1_us = t1_cc0_d.copy()
            print("t1_us init =", t1_us)
            #t2_us = t2_d.copy()
            t1_cc_d = np.zeros(t1_us.shape, dtype=t1_us.dtype)
            #t2_us = t2_cc0_d.copy()
            mycc_d._scf.exxdiv=None
            for i in range(20):
                print("iteration = ", i)
                t2_last = t2_us[0, 0, 1].copy() 
                write_amps(t2_us, "/Users/keliao/Work/project/upscaling/"+sys_name+"_cc_t2_add_"+str(i)+"_nk"+str(nk_d)+".hdf5")
                write_amps(t1_cc_d, "/Users/keliao/Work/project/upscaling/"+sys_name+"_cc_t1_add_"+str(i)+"_nk"+str(nk_d)+".hdf5")
                for j in range(1):
                    print("micro iter = ", j)
                    mycc_d.max_cycle = 1
                    ecc_d_us, t1_cc_d, t2_cc_d = mycc_d.kernel(t1=t1_cc_d, t2=t2_us)
                for k in range(1):
                    for ki in range(kmp_d.nkpts):
                        print("processing at ", ki)
                        for kj in range(kmp_d.nkpts):
                            for ka in range(kmp_d.nkpts):
                                phase = t2_cc0_d[ki, kj, ka] / np.abs(t2_cc0_d[ki, kj, ka])
                                phase[np.abs(t2_cc0_d[ki, kj, ka]) < 1e-10] = 0.
                                t2_us[ki, kj, ka] += (np.abs(t2_cc_d[0, 0, 1]) - np.abs(t2_last)) * phase
                t2_last = t2_us[0, 0, 1].copy()

            mycc_d = cc.KCCSD(kmf_d)
            mycc_d.max_cycle = 50
            ecc_d, t1_cc_scf, t2_cc_scf = mycc_d.kernel()
            write_amps(t2_cc_scf, "/Users/keliao/Work/project/upscaling/"+sys_name+"_cc_t2_nk_"+str(nk_d)+".hdf5")
            write_amps(t1_cc_scf, "/Users/keliao/Work/project/upscaling/"+sys_name+"_cc_t1_nk_"+str(nk_d)+".hdf5")
        
            # replace q=0 in t2_us and re-evaluate the energy
            #for i in range(kmp_d.nkpts):
            #t2_us[0,0,0] = t2_cc_scf[0,0,0].copy()
            #write_amps(t2_us, "/Users/keliao/Work/project/upscaling/"+sys_name+"_us_cc_t2_nk_"+str(nk_d)+".hdf5")
            #mycc_d.max_cycle = 0
            #ecc_d, t1_cc_d, t2_cc_d = mycc_d.kernel(t1=t1_cc_d, t2=t2_us)
            #ecc_d, t1_cc_d, t2_cc_d = mycc_d.kernel(t1=t1_cc_d, t2=t2_cc_scf)


if __name__ == "__main__":
    main()
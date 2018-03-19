#! /usr/bin/env python
from __future__ import print_function, division

def do_scf(inp):
    '''Do the requested SCF.'''

    from pyscf import gto, scf, dft, cc, fci, ci, ao2mo, mcscf, mrpt, lib
    from pyscf.cc import ccsd_t
    import numpy as np
    from fcidump import fcidump

    # sort out the method
    mol = inp.mol
    method = inp.scf.method.lower()

    # UHF
    if method == 'uhf':
        ehf, mSCF = do_hf(inp, unrestricted=True)
        print_energy('UHF', ehf)

    # RHF
    elif method in ('rhf', 'hf'):
        ehf, mSCF = do_hf(inp)
        print_energy('RHF', ehf)

    # CCSD and CCSD(T)
    elif method in ('ccsd', 'ccsd(t)'):
        ehf, tSCF = do_hf(inp)
        print_energy('RHF', ehf)

        inp.timer.start('ccsd')
        frozen = 0
        if inp.scf.freeze is not None: frozen = inp.scf.freeze
        mSCF = cc.RCCSD(tSCF, frozen=frozen)
        mSCF.max_cycle = inp.scf.maxiter
        eccsd, t1, t2 = mSCF.kernel()
        print_energy('CCSD', ehf + eccsd)
        inp.timer.end('ccsd')

        if method in ('ccsd(t)'):
            inp.timer.start('ccsd(t)')
            eris = mSCF.ao2mo()
            e3 = ccsd_t.kernel(mSCF, eris)
            print_energy('CCSD(T)', ehf + eccsd + e3)
            inp.timer.end('ccsd(t)')

    elif method in ('cisd'):
        ehf, tSCF = do_hf(inp)
        print_energy('RHF', ehf)

        inp.timer.start('cisd')
        frozen = 0
        if inp.scf.freeze is not None: frozen = inp.scf.freeze
        mSCF = ci.CISD(tSCF, frozen=frozen)
        ecisd = mSCF.kernel()[0]
        print_energy('CISD', ehf + ecisd)
        inp.timer.end('cisd')

    # UKS
    elif method in ('uks' or 'udft'):
        inp.timer.start('uks')
        inp.timer.start('grids')
        grids = dft.gen_grid.Grids(mol)
        grids.level = inp.scf.grid
        grids.build()
        inp.timer.end('grids')
        mSCF = dft.UKS(mol)
        mSCF.grids = grids
        mSCF.xc = inp.scf.xc
        mSCF.conv_tol = inp.scf.conv
        mSCF.conv_tol_grad = inp.scf.grad
        mSCF.max_cycle = inp.scf.maxiter
        mSCF.init_guess = inp.scf.guess
        mSCF.small_rho_cutoff = 1e-20
        eks = mSCF.kernel()
        print_energy('UKS', eks)
        inp.timer.end('uks')

    # RKS
    elif method in ('rks', 'ks', 'rdft', 'dft'):
        inp.timer.start('ks')
        inp.timer.start('grids')
        grids = dft.gen_grid.Grids(mol)
        grids.level = inp.scf.grid
        grids.build()
        inp.timer.end('grids')
        if mol.nelectron%2 == 0:
            mSCF = dft.RKS(mol)
        else:
            mSCF = dft.ROKS(mol)
        mSCF.grids = grids
        mSCF.xc = inp.scf.xc
        mSCF.conv_tol = inp.scf.conv
        mSCF.conv_tol_grad = inp.scf.grad
        mSCF.max_cycle = inp.scf.maxiter
        mSCF.init_guess = inp.scf.guess
        mSCF.small_rho_cutoff = 1e-20
        eks = mSCF.kernel()
        print_energy('RKS', eks)
        inp.timer.end('ks')

    # Unrestricted FCI
    elif method == 'ufci':
        ehf, mSCF = do_hf(inp, unrestricted=True)
        print_energy('UHF', ehf)

        inp.timer.start('fci')
        cis = fci.direct_uhf.FCISolver(mol)
        norb = mSCF.mo_energy[0].size
        nea = (mol.nelectron+mol.spin) // 2
        neb = (mol.nelectron-mol.spin) // 2
        nelec = (nea, neb)
        mo_a = mSCF.mo_coeff[0]
        mo_b = mSCF.mo_coeff[1]
        h1e_a = reduce(np.dot, (mo_a.T, mSCF.get_hcore(), mo_a))
        h1e_b = reduce(np.dot, (mo_b.T, mSCF.get_hcore(), mo_b))
        g2e_aa = ao2mo.incore.general(mSCF._eri, (mo_a,)*4, compact=False)
        g2e_aa = g2e_aa.reshape(norb,norb,norb,norb)
        g2e_ab = ao2mo.incore.general(mSCF._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
        g2e_ab = g2e_ab.reshape(norb,norb,norb,norb)
        g2e_bb = ao2mo.incore.general(mSCF._eri, (mo_b,)*4, compact=False)
        g2e_bb = g2e_bb.reshape(norb,norb,norb,norb)
        h1e = (h1e_a, h1e_b)
        eri = (g2e_aa, g2e_ab, g2e_bb)

        eci = fci.direct_uhf.kernel(h1e, eri, norb, nelec)[0]
        print_energy('FCI', eci)
        inp.timer.end('fci')

    # FCI
    elif method in ('fci'):
        ehf, mSCF = do_hf(inp)
        print_energy('RHF', ehf)

        inp.timer.start('fci')
        if inp.scf.freeze is None:
            mCI = fci.FCI(mSCF)
            mCI.kernel()[0]
            eci = mCI.eci
        else:
            nel  = mol.nelectron - inp.scf.freeze * 2
            ncas = mol.nao_nr() - inp.scf.freeze
            if mol.spin == 0:
                nelecas = nel
            else:
                nelecas = (nel//2 + nel%2, nel//2)
            mCI = mcscf.CASCI(mSCF, ncas, nelecas)
            eci = mCI.kernel()[0]
        print_energy('FCI', eci)
        inp.timer.end('fci')

    # CASCI
    elif method == 'casci':
        if inp.scf.cas is None:
            print ('ERROR: Must specify CAS space')
            return inp
        ehf, mSCF = do_hf(inp)
        print_energy('RHF', ehf)

        inp.timer.start('casci')
        if mol.spin == 0:
            nelecas = inp.scf.cas[0]
        else:
            nelecas = (inp.scf.cas[0]//2 + inp.scf.cas[0]%2,
                       inp.scf.cas[0]//2)
        mCI = mcscf.CASCI(mSCF, inp.scf.cas[1], nelecas)
        eci = mCI.kernel()[0]
        print_energy('CASCI', eci)
        inp.timer.end('casci')

    # CASSCF
    elif method == 'casscf' or method == 'ucasscf':
        if inp.scf.cas is None:
            print ('ERROR: Must specify CAS space')
            return inp

        lunrestricted = (method == 'ucasscf')
        ehf, mSCF = do_hf(inp, unrestricted=lunrestricted)
        print_energy('HF', ehf)

        inp.timer.start('casci')
        if mol.spin == 0:
            nelecas = inp.scf.cas[0]
        else:
            nelecas = (inp.scf.cas[0]//2 + inp.scf.cas[0]%2,
                       inp.scf.cas[0]//2)
        mCI = mcscf.CASSCF(mSCF, inp.scf.cas[1], nelecas)
        eci = mCI.kernel()[0]
        print_energy('CASSCF', eci)
        inp.timer.end('casci')

    elif method == 'nevpt2' or method == 'unevpt2':
        if inp.scf.cas is None:
            print ('ERROR: Must specify CAS space')
            return inp

        ehf, mSCF = do_hf(inp)
        print_energy('RHF', ehf)

        inp.timer.start('casci')
        if mol.spin == 0:
            nelecas = inp.scf.cas[0]
        else:
            nelecas = (inp.scf.cas[0]//2 + inp.scf.cas[0]%2,
                       inp.scf.cas[0]//2)
        mCI = mcscf.CASCI(mSCF, inp.scf.cas[1], nelecas)
        eci = mCI.kernel()[0]
        print_energy('CASCI', eci)
        inp.timer.end('casci')

        inp.timer.start('nevpt2')
        ept2 = mrpt.NEVPT2(mCI) + eci
        print_energy('NEVPT2', ept2)
        inp.timer.end('nevpt2')

    else:
        print ('ERROR: Unrecognized SCF method!')

    # dump fcidump file if needed
    if inp.fcidump:
        if inp.filename[-4:].lower() == '.inp':
            fcifile = inp.filename[:-4] + '.fcidump'
        else:
            fcifile = inp.filename + '.fcidump'
        fcidump(mSCF, filename=fcifile, tol=1e-6)

    # plot MOs if needed
    if inp.mo2cube:
        from mo_2_cube import save_MOs
        save_MOs(inp, mSCF, mSCF.mo_coeff)

    # save and return
    inp.mf = mSCF
    return inp


def do_hf(inp, unrestricted=False):
    '''Do a RHF, ROHF, or UHF calculation.'''

    from pyscf import scf

    mol = inp.mol
    timer = 'rhf'
    if unrestricted:
        timer = 'uhf'
    elif mol.spin > 0:
        timer = 'rohf'

    inp.timer.start(timer)

    # create SCF object
    if unrestricted:
        mSCF = scf.UHF(mol)
    elif mol.spin > 0:
        mSCF = scf.ROHF(mol)
    else:
        mSCF = scf.RHF(mol)

    # set values from input
    mSCF.level_shift = inp.scf.shift
    mSCF.conv_tol = inp.scf.conv
    mSCF.conv_tol_grad = inp.scf.grad
    mSCF.max_cycle = inp.scf.maxiter
    mSCF.init_guess = inp.scf.guess
    mSCF.diis_space = inp.scf.diis

    # do SCF
    ehf = mSCF.kernel()

    inp.timer.end(timer)

    # return energy and SCF object
    return ehf, mSCF


def print_energy(string, e):
    '''Prints energy in some standard format.'''
    string = string.rstrip() + ' Energy'
    string = '{0:30s} = {1:25.15f}'.format(string, e)
    ld = 80 - len(string) - 2
    lb = ld // 2
    le = ld // 2 + ld % 2
    string = ('!'*lb) + ' ' + string + ' ' + ('!'*le)
    print (string)

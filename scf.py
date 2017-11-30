#! /usr/bin/env python
from __future__ import print_function, division

def do_scf(inp):
    '''Do the requested SCF.'''

    from pyscf import gto, scf, dft, cc, fci, ci, ao2mo
    from pyscf.cc import ccsd_t
    import numpy as np
    from fcidump import fcidump

    # sort out the method
    mol = inp.mol
    method = inp.scf.method.lower()

    # UHF
    if method == 'uhf':
        inp.timer.start('uhf')
        mSCF = scf.UHF(mol)
        mSCF.conv_tol = inp.scf.conv
        mSCF.conv_tol_grad = inp.scf.grad
        mSCF.max_cycle = inp.scf.maxiter
        mSCF.init_guess = inp.scf.guess
        mSCF.kernel()
        inp.timer.end('uhf')

    # RHF
    elif method in ('rhf', 'hf'):
        inp.timer.start('hf')
        if mol.nelectron%2 == 0:
            mSCF = scf.RHF(mol)
        else:
            mSCF = scf.ROHF(mol)
        mSCF.conv_tol = inp.scf.conv
        mSCF.conv_tol_grad = inp.scf.grad
        mSCF.max_cycle = inp.scf.maxiter
        mSCF.init_guess = inp.scf.guess
        mSCF.kernel()
        inp.timer.end('hf')

    # CCSD and CCSD(T)
    elif method in ('ccsd', 'ccsd(t)'):
        inp.timer.start('hf')
        tSCF = scf.RHF(mol)
        tSCF.conv_tol = inp.scf.conv
        tSCF.conv_tol_grad = inp.scf.grad
        tSCF.max_cycle = inp.scf.maxiter
        tSCF.init_guess = inp.scf.guess
        ehf = tSCF.kernel()
        inp.timer.end('hf')
        inp.timer.start('ccsd')
        mSCF = cc.RCCSD(tSCF)
        mSCF.max_cycle = inp.scf.maxiter
        eccsd, t1, t2 = mSCF.kernel()
        inp.timer.end('ccsd')

        if method in ('ccsd(t)'):
            inp.timer.start('ccsd(t)')
            eris = mSCF.ao2mo()
            e3 = ccsd_t.kernel(mSCF, eris)
            print ('Total CCSD(T) = {0:20.15f}'.format(ehf + eccsd + e3))
            inp.timer.end('ccsd(t)')

    elif method in ('cisd'):
        inp.timer.start('hf')
        tSCF = scf.RHF(mol)
        tSCF.conv_tol = inp.scf.conv
        tSCF.conv_tol_grad = inp.scf.grad
        tSCF.max_cycle = inp.scf.maxiter
        tSCF.init_guess = inp.scf.guess
        ehf = tSCF.kernel()
        print ('HARTREE-FOCK = {0:20.15f}'.format(ehf))
        inp.timer.end('hf')

        inp.timer.start('cisd')
        mSCF = ci.CISD(tSCF)
        ecisd = mSCF.kernel()[0]
        print ('Total CISD   = {0:20.15f}'.format(ehf + ecisd))
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
        mSCF.kernel()
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
        mSCF.kernel()
        inp.timer.end('ks')

    # Unrestricted FCI
    elif method in ('ufci'):
        inp.timer.start('hf')
        mSCF = scf.UHF(mol)
        mSCF.conv_tol = inp.scf.conv
        mSCF.conv_tol_grad = inp.scf.grad
        mSCF.max_cycle = inp.scf.maxiter
        mSCF.init_guess = inp.scf.guess
        ehf = mSCF.kernel()
        print ('HF Energy =     {0:20.15f}'.format(ehf))
        inp.timer.end('hf')

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


#        mCI = fci.FCI(mSCF)
#        eci = mCI.kernel()[0]
        print ('FCI Energy =    {0:20.15f}'.format(eci))
        inp.timer.end('fci')

    # FCI
    elif method in ('fci'):
        inp.timer.start('hf')
        mSCF = scf.RHF(mol)
        mSCF.conv_tol = inp.scf.conv
        mSCF.conv_tol_grad = inp.scf.grad
        mSCF.max_cycle = inp.scf.maxiter
        mSCF.init_guess = inp.scf.guess
        ehf = mSCF.kernel()
        print ('HF Energy =     {0:20.15f}'.format(ehf))
        inp.timer.end('hf')

        inp.timer.start('fci')
        mCI = fci.FCI(mSCF)
        eci = mCI.kernel()[0]
        print ('FCI Energy =    {0:20.15f}'.format(eci))
        inp.timer.end('fci')

    else:
        print ('ERROR: Unrecognized SCF method!')

    # dump fcidump file if needed
    if inp.fcidump:
        if inp.filename[-4:].lower() == '.inp':
            fcifile = inp.filename[:-4] + '.fcidump'
        else:
            fcifile = inp.filename + '.fcidump'
        fcidump(mSCF, filename=fcifile)

    # save and return
    inp.mf = mSCF
    return inp


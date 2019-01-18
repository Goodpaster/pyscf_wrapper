def vibrations(inp):
    '''Calculate the vibrational frequencies.'''

    import constants
    import numpy as np
    import scipy as sp
    from pyscf.future import hessian
    from .scf import do_scf
    from .read_input import pstr

    na = inp.mol.natm
    n3 = na * 3

    # get atom coords and sqrt of mass
    q = inp.mol.atom_coords().reshape(n3)
    m = np.zeros((na,3))
    for i in range(na):
        sym = constants.elem[inp.mol.atom_charge(i)]
        m[i] = constants.atomic_mass(sym)
    m = m.reshape(n3)
    m = np.sqrt(m)

    # get mass-weighted coordinates
    x = q * m

    inp.timer.start('scf')
    inp = do_scf(inp)
    inp.timer.end('scf')

    # get hessian
    inp.timer.start('hessian')
    if hasattr(inp, 'hessian'):
        h = inp.hessian
    else:
        if inp.scf.method in ('uhf', 'rhf', 'hf'):
            h = hessian.RHF(inp.mf).kernel()
        else:
            h = hessian.RKS(inp.mf).kernel()
        h = h.transpose(0,2,1,3).reshape(n3,n3)
    inp.timer.end('hessian')


    # mass-weighted hessian
    M = np.outer(m,m)
    h /= m
    print (h.reshape(na,3,na,3))

    # diagonalize
    w, X = sp.linalg.eig(h)
    imf = np.where( w < 0.)
    w = np.abs(w)
    w = np.sqrt(w)
    w[imf] = -w[imf]
    ii = np.argsort(w)
    w = w[ii]
    w = constants.HART2WAVENUM(w) / ( 4. * np.pi * np.sqrt(m.sum()) )

    pstr ('Vibrational frequencies', delim='=')
    for i in range(len(w)):
        print ('{0:9.4f} cm-1'.format(w[i]))

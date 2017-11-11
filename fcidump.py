#!/usr/bin/env python

def fcidump(mf, filename='FCIDUMP'):
    from pyscf.tools.fcidump import from_integrals
    from pyscf import ao2mo
    from numpy import dot

    h1e = reduce(dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    eri = ao2mo.kernel(mf.mol, mf.mo_coeff)

    nmo = mf.mo_coeff.shape[1]
    nelec = mf.mol.nelectron
    ms = 0
    orbsym = None

    fout = open(filename, 'w')

    write_head(fout, nmo, nelec, ms, orbsym)

    write_eri(fout, eri, nmo, tol=1e-8)

    write_hcore(fout, h1e, nmo, tol=1e-8)

    output_format = '%15.12f    0    0    0    0'
    fout.write(output_format % mf.energy_nuc())
    fout.close()

def write_head(fout, nmo, nelec, ms=0, orbsym=None):
    import numpy
    if not isinstance(nelec, (int, numpy.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
    if orbsym is not None and len(orbsym) > 0:
        fout.write('  ORBSYM=%s\n' % ','.join([str(x) for x in orbsym]))
    else:
        fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
    fout.write('  ISYM=1,\n')
    fout.write(' &END\n')

def write_eri(fout, eri, nmo, tol=1e-15, float_format='%15.12f'):
    npair = nmo*(nmo+1)//2
    output_format = float_format + ' %4d %4d %4d %4d\n'
    if eri.ndim == 2: # 4-fold symmetry
        assert(eri.size == npair**2)
        ij = 0 
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0 
                for k in range(0, nmo):
                    for l in range(0, k+1):
                        if abs(eri[ij,kl]) > tol:
                            fout.write(output_format % (eri[ij,kl], i+1, j+1, k+1, l+1))
                        kl += 1
                ij += 1
    else:  # 8-fold symmetry
        assert(eri.size == npair*(npair+1)//2)
        ij = 0 
        ijkl = 0 
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0 
                for k in range(0, i+1):
                    for l in range(0, k+1):
                        if ij >= kl: 
                            if abs(eri[ijkl]) > tol:
                                fout.write(output_format % (eri[ijkl], i+1, j+1, k+1, l+1))
                            ijkl += 1
                        kl += 1
                ij += 1

def write_hcore(fout, h, nmo, tol=1e-15, float_format='%15.12f'):
    h = h.reshape(nmo,nmo)
    output_format = float_format + ' %4d %4d    0    0\n'
    for i in range(nmo):
        for j in range(0, i+1):
            if abs(h[i,j]) > tol:
                fout.write(output_format % (h[i,j], i+1, j+1))

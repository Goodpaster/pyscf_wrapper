#! /usr/bin/env python
from __future__ import print_function, division

def read_input(filename):
    '''Reads a formatted pySCF input file, and
    generates a relevant pySCF Mole object from it.

    Input: the filename of the pySCF input file
    Output: a pySCF Mole object'''

    import input_reader
    from pyscf import gto, scf, dft, cc
    import numpy as np
    from pyscf.cc import ccsd_t
    from pyscf.gto.basis import load


    # initialize reader for a pySCF input
    reader = input_reader.InputReader(comment=['!', '#', '::', '//'],
             case=False, ignoreunknown=True)

    # add atoms block
    atoms = reader.add_block_key('atoms', required=True)
    atoms.add_regex_line('atom',
        '\s*([A-Za-z.]+)\s+(\-?\d+\.?\d*)\s+(\-?\d+.?\d*)\s+(\-?\d+.?\d*)', repeat=True)
    atoms.add_line_key('read', type=str, default=None)      # read geom from xyz file

    # add basis block
    basis = reader.add_block_key('basis')
    basis.add_regex_line('atom', '\s*([A-Za-z]+)\s+([A-Za-z0-9\+]+)', repeat=True)

    # add simple line keys
    reader.add_line_key('memory', type=(int, float))        # max memory in MB
    reader.add_line_key('unit', default='angstrom')         # coord unit
    reader.add_line_key('charge', type=int)                 # molecular charge
    reader.add_line_key('spin', type=int)                   # molecular spin
    reader.add_line_key('symmetry', type=int, default=None) # mol symmetry (False, 0, 1)
    reader.add_line_key('verbose', type=(0,1,2,3,4,5,6,7,8,9), default=4) # verbose level

    # add boolean key
    reader.add_boolean_key('vfreq', default=False)          # vibrational frequencies
    reader.add_boolean_key('fcidump', default=False)        # create FCIDUMP file
    reader.add_boolean_key('mo2cube', default=False)        # Cube files for each MO
    reader.add_boolean_key('molden', default=False)         # saves molden file

    # add scf block keys
    scf = reader.add_block_key('scf', required=True)
    scf.add_line_key('method', default='hf')                # SCF method
    scf.add_line_key('xc', default='lda,vwn')               # XC functional
    scf.add_line_key('conv', type=float, default=1e-8)      # conv_tol
    scf.add_line_key('grad', type=float, default=1e-6)      # conv_tol_grad
    scf.add_line_key('maxiter', type=int, default=50)       # max iterations
    scf.add_line_key('guess', type=('minao', 'atom', '1e'), default='minao') # intial density guess
    scf.add_line_key('grid', type=(1,2,3,4,5,6,7,8,9), default=2) # dft numint grid
    scf.add_line_key('damp', type=float, default=0)         # SCF damping factor
    scf.add_line_key('shift', type=float, default=0)        # level shift
    scf.add_line_key('diis', type=int, default=8)           # diis space
    scf.add_line_key('freeze', type=int, default=None)      # frozen core orbitals
    scf.add_line_key('cas', type=[int, int], default=None)  # CAS space for CASCI or CASSCF
    scf.add_line_key('casspin', type=int, default=None)
    scf.add_line_key('roots', type=int, default=None)       # number of states (FCI)

    # orbitals for active space
    aorbs = scf.add_block_key('casorb', required=False)
    aorbs.add_regex_line('orb', '\s*([0-9]*)', repeat=True)

    # add electric core potential (ECP) block
    ecp = reader.add_block_key('ecp')
    ecp.add_regex_line('atom', '\s*([A-Za-z]+)\s+([A-Za-z0-9]+)', repeat=True)

    # add geomopt block key
    geomopt = reader.add_block_key('geomopt', required=False)
    geomopt.add_line_key('conv', type=float, default=1e-6)      # energy convergence
    geomopt.add_line_key('rconv', type=float, default=1e-2)     # distance convergence
    geomopt.add_line_key('maxiter', type=int, default=25)       # geometry iterations
    geomopt.add_line_key('gradtol', type=float, default=1e-4)   # gradient tolerance

    # read the input filename
    inp  = reader.read_input(filename)
    inp.filename = filename

    # sanity checks
    if inp.atoms.atom is None and inp.atoms.read is None:
        sys.exit("Must specify atom coordinates or read from xyz file!")
    if inp.atoms.atom is not None and inp.atoms.read is not None:
        sys.exit("Must only specify either atom coordinates OR xyz file!")

    # set orbs
    if inp.scf.casorb is not None:
        temp = []
        for i in range(len(inp.scf.casorb.orb)):
            temp.append(int(inp.scf.casorb.orb[i].group(0))-1)
        inp.scf.casorb = np.array(temp)

    # print input file to screen
    pstr("Input File")
    [print (i[:-1]) for i in open(filename).readlines() if ((i[0] not in ['#', '!'])
        and (i[0:2] not in ['::', '//']))]
    pstr("End Input", addline=False)

    # initialze pySCF molecule object
    mol = gto.Mole()

    # basis block into basis dict
    basis = {'all': 'sto3g'}
    if inp.basis is not None:
        for r in inp.basis.atom:
            basis.update({r.group(1): r.group(2)})

    # read atoms and transform to pyscf format
    mol.atom = []
    mol.basis = {}

    ghbasis = []

    # collect from coordinates input
    if inp.atoms.atom is not None:
        for r in inp.atoms.atom:

            coord = np.array([r.group(2), r.group(3), r.group(4)], dtype=float)

            # ghost atom
            if 'ghost.' in r.group(1).lower() or 'gh.' in r.group(1).lower():
                basatm  = r.group(1).split('.')[1]
                atmstr  = ['ghost:{0}'.format(len(mol.atom)+1), coord]
                bastype = [basis[basatm] if basatm in basis.keys() else basis['all']][0]

                mol.basis.update({'ghost:{0}'.format(len(mol.atom)+1):
                    load(bastype, basatm)})
                mol.atom.append(atmstr)

            # regular atom
            else:
                basatm  = r.group(1)
                atmstr  = ['{0}:{1}'.format(basatm, len(mol.atom)+1), coord]
                bastype = [basis[basatm] if basatm in basis.keys() else basis['all']][0]

                mol.basis.update({'{0}:{1}'.format(basatm, len(mol.atom)+1):
                    load(bastype, basatm)})
                mol.atom.append(atmstr) 

    # collect from xyz file
    elif inp.atoms.read is not None:
        xyzlines = open(inp.atoms.read, 'r').readlines()
        natm = int(xyzlines[0])
        for i in range(2,natm+2):
            line = xyzlines[i].split()

            atmstr = ['{0}:{1}'.format(line[0], len(mol.atom)+1), (float(line[1]),
                float(line[2]), float(line[3]))]
            bastype = [basis[line[0]] if line[0] in basis.keys() else basis['all']][0]

            mol.basis.update({'{0}:{1}'.format(line[0], len(mol.atom)+1):
                 load(bastype, line[0])})
            mol.atom.append(atmstr)

    # electric core potential dict
    mol.ecp = None
    if inp.ecp is not None:
        mol.ecp = {}
        for r in inp.ecp.atom:
            mol.ecp.update({r.group(1): r.group(2)})

    # build molecule object
    if inp.memory is not None: mol.max_memory = inp.memory
    mol.unit = inp.unit
    if inp.charge is not None: mol.charge = inp.charge
    if inp.spin is not None: mol.spin = inp.spin
    mol.verbose = inp.verbose
    if inp.symmetry is not None: mol.symmetry = inp.symmetry
    mol.build()#dump_input=False)
    inp.mol = mol

    # return inp object
    return inp

def pstr(st, delim="=", l=80, fill=True, addline=True):
    '''Print formatted string <st> to output'''
    if addline: print ("")
    if len(st) == 0:
        print (delim*l)
    elif len(st) >= l:
        print (st)
    else:
        l1 = int((l-len(st)-2)/2)
        l2 = int((l-len(st)-2)/2 + (l-len(st)-2)%2)
        if fill:
            print (delim*l1+" "+st+" "+delim*l2)
        else:
            print (delim+" "*l1+st+" "*l2+delim)

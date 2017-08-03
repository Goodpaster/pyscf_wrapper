#!/usr/bin/env python

def main():
    '''Python wrapper to read a pySCF input file
    and run the corresponding calculations.'''

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent
    from read_input import read_input, pstr
    import sys
    from simple_timer import timer
    from scf import do_scf
    from geomopt import do_geomopt
    from vfreq import vibrations

    # generate a parser to read the input files
    parser = ArgumentParser(description=dedent(main.__doc__),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('input_files', nargs='*', default=sys.stdin,
                        help='The pySCF input files.')
    args = parser.parse_args()

    # Cycle through all input files
    for files in args.input_files:

        # print header
        pstr ("", delim="*", addline=False)
        pstr ("PySCF CALCULATION", delim="*", fill=False, addline=False)
        pstr ("", delim="*", addline=False)

        # get input data
        inp = read_input(files)

        # start timer
        inp.timer = timer()

        # do single point energy
        inp = do_scf(inp)

        # do geomotry optimization
        if inp.geomopt: inp = do_geomopt(inp)

        if inp.vfreq: vibrations(inp)

        # end timer
        inp.timer.close()

        # print footer
        pstr("", delim="*")
        pstr("END OF CALCULATION", delim="*", fill=False, addline=False)
        pstr("", delim="*", addline=False)

if __name__=='__main__':
    main()

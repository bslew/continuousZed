#!/usr/bin/env python
'''
continuousZed -- fetch/calculate/send time-averaged ZD pointing corrections

@author:     Bartosz Lew

@contact:    bartosz.lew@protonmail.com
@deffield    updated: Updated
'''

import sys
import os

import numpy as np

from RT32continuousPointing import config_file, zed_parser, pointingCorrections, rt32comm


__all__ = []
__version__ = 0.1
__date__ = '2022-02-02'
__updated__ = '2022-02-02'

DEBUG = 0
TESTRUN = 0
PROFILE = 0



def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    args = zed_parser.get_parser()
    cfg=config_file.readConfigFile(verbosity=args.verbose)

    
    if args.medianZD:
        pointingCorrections.get_median_ZD_correction(args,cfg)
    if args.test_rt32_comm:
        rt32comm.rt32tcpclient().connectRT4().send_cmd('flagM -10 0')
        # print(r)
        
    if args.setauto:
        dZD=pointingCorrections.get_median_ZD_correction(args,cfg)
        corr=int(dZD*10000)
        print("Sending RT-32 ZD pointing correction [10^-4 deg]: {}".format(corr))
        rt32comm.rt32tcpclient().connectRT4().send_cmd('flagM -10 %i' % corr)
        
    if args.set!='':
        try:
            dZD=float(args.set)
            corr=int(dZD*10000)
            print("Sending RT-32 ZD pointing correction [10^-4 deg]: {}".format(corr))
            rt32comm.rt32tcpclient().connectRT4().send_cmd('flagM -10 %i' % corr)
        except ValueError:
            print("Could not set value")
            pass
        
        
        

if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-h")
        sys.argv.append("-v")
        sys.argv.append("-r")
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = ''
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())
    
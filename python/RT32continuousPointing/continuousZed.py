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

from RT32continuousPointing import config_file, zed_parser, pointingCorrections, rt32comm, zedLogger


__all__ = []
__version__ = 0.1
__date__ = '2022-02-02'
__updated__ = '2022-02-02'

DEBUG = 0
TESTRUN = 0
PROFILE = 0


def sendZDoffset(corr,cfg):
    print("Sending RT-32 ZD pointing correction [10^-4 deg]: {}".format(corr))
    rt32comm.rt32tcpclient().connectRT4(
        host=cfg['RT32']['host'],
        port=cfg.getint('RT32','port'),
        ).send_cmd('flagM -10 %i' % corr)
    

def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    args = zed_parser.get_parser()
    cfg=config_file.readConfigFile(verbosity=args.verbose)

    logger=zedLogger.get_logging('continuousZed', fname='continuousZed.log', mode='a')
    
    
    if args.median:
        pointingCorrections.get_median_corrections(args,cfg)
    if args.test_rt32_comm:
        rt32comm.rt32tcpclient().connectRT4(
            host=cfg['RT32']['host'],
            port=cfg.getint('RT32','port'),
            ).send_cmd('flagM -10 0')
        # print(r)
        
    if args.set_dZD_auto:
        try:
            _,dZD=pointingCorrections.get_median_corrections(args,cfg)
            corr=int(dZD*10000)
            sendZDoffset(corr, cfg)
            logger.info("new continuous ZD correction: {}".format(corr))
        except ValueError:
            logger.info("Could not set ZD correction")
            pass
        
    if args.set_dZD!='':
        try:
            dZD=float(args.set_dZD)
            corr=int(dZD*10000) # convert from deg to 1e-4deg
            sendZDoffset(corr, cfg)
            logger.info("new continuous ZD correction: {}".format(corr))
        except ValueError:
            logger.info("Could not set ZD correction")
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
    
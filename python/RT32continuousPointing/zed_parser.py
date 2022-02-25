'''
Created on Dec 9, 2021

@author: blew
'''
import os,sys

__version__ = 0.1
__date__ = '2022-01-10'
__updated__ = '2021-01-10'

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

def get_parser():
    
    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    # try:
    #     program_shortdesc = __import__('__main__').__doc__.split("\n")[1] if len(__import__('__main__').__doc__.split("\n"))>=2 else ''
    # except:
    program_shortdesc="Script to manage slowly varying RT32 ZD pointing corrections."
    program_epilog ='''
    



    
Examples:

continuousZed.py --help

'''
    program_license = '''%s

  Created by Bartosz Lew on %s.
  Copyright 2021 Bartosz Lew. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, epilog=program_epilog, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %(default)s]", default=0)
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        parser.add_argument(dest="paths", help="logfile.log [default: %(default)s]", metavar="path", nargs='*')
        parser.add_argument('--median', action='store_true',
                            help='Calculate median corrections from recent observations. [default: %(default)s]', 
                            default=False)
        # parser.add_argument('-o', type=str,
        #                     help='Print content of widom file. [default: %(default)s]', 
        #                     default='')
        parser.add_argument('--test_rt32_comm', action='store_true',
                            help='test_rt32_comm [default: %(default)s]', 
                            default=False)

        parser.add_argument('--set_dZD_auto', action='store_true',
                            help='calculate median ZD correction and send the result to RT-32 control system [default: %(default)s]', 
                            default=False)
        parser.add_argument('--set_dxZD_auto', action='store_true',
                            help='calculate median xZD correction and send the result to RT-32 control system [default: %(default)s]', 
                            default=False)

        parser.add_argument('--logRT32stats', action='store_true',
                            help='request and store RT-32 control system pointing settings [default: %(default)s]', 
                            default=False)

        parser.add_argument('--set_dZD', type=str,
                            help='send ZD continuous correction to RT-32 control system (units: deg) [default: %(default)s]', 
                            default='')
        parser.add_argument('--set_dxZD', type=str,
                            help='send xZD continuous correction to RT-32 control system (units: deg) [default: %(default)s]', 
                            default='')
        
        parser.add_argument('--plot', action='store_true',
                            help='plot loaded corrections [debug stuff]')
        parser.add_argument('--plot_stats', action='store_true',
                            help='plot corrections stats')
        

        parser.add_argument('--temp', type=str, dest='temperatures_file',
                            help='File containing UTC dates and temperatueres in degC in format 2020-11-01T02:57:07 5.620000 [default: %(default)s]', 
                            default='')


        # Process arguments
        args = parser.parse_args()

    except KeyboardInterrupt:
        ## handle keyboard interrupt ###
        raise
#     except Exception as e:
#         if DEBUG or TESTRUN:
#             raise(e)
#         indent = len(program_name) * " "
#         sys.stderr.write(program_name + ": " + repr(e) + "\n")
#         sys.stderr.write(indent + "  for help use --help")
#         return 2
        
    return args

'''
Created on Feb 7, 2022

@author: blew
'''
import os,sys
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import copy

from RT32continuousPointing import config_file, zed_parser, pointingCorrections, zedLogger
from RT32continuousPointing import temperature

def plot_corrections(P,labels,**kwargs):
    '''
    '''
    
    if 'outfile' in kwargs.keys():
        outfile=kwargs['outfile']
    
    plt.figure(figsize=(16,10))    
    for i,(p,l) in enumerate(zip(P,labels)):
        plt.scatter(p.dt,p.dZD,label=l)


    if 'overplot_custom_model' in kwargs.keys():
        dt0=datetime.datetime(2020,10,1)
        dt1=datetime.datetime(2022,2,1)
        yr=365.25 
        x=np.array([ i for i,x in enumerate(np.arange(dt0,dt1,datetime.timedelta(days=1))) ],dtype=float)/yr
        dt=np.arange(dt0,dt1,datetime.timedelta(days=1))       

        ZDdrift=18.71*x/1000
        A=10/1000
        Tdrift=0.283*np.cos(2.0*np.pi*(x-(7.+2)/12))*A

        plt.plot(dt,ZDdrift, c='k', lw=2)
        plt.plot(dt,Tdrift, c='r', lw=2)
        plt.plot(dt,ZDdrift+Tdrift, c='b', lw=2)
        plt.plot(dt,Tdrift-ZDdrift, c='c', lw=2)


    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('$\Delta$ZD [deg]')
    plt.ylim([-0.05,0.05])
    plt.grid()
    if outfile:
        plt.savefig(outfile)
    plt.show()


def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    args = zed_parser.get_parser()
    cfg=config_file.readConfigFile(verbosity=args.verbose)

    P=[]
    labels=[]
    tmscale=cfg.getint('ZED','time_scale')

    f=os.path.join(cfg['DATA']['cross_scan_data_dir'],'FastScan_6.70.off')
    P.append(pointingCorrections.fastScanCorrections(f,tmscale=tmscale))
    labels.append('6.7 GHz')

    f=os.path.join(cfg['DATA']['cross_scan_data_dir'],'FastScan_12.00.off')
    P.append(pointingCorrections.fastScanCorrections(f,tmscale=tmscale))
    P[-1].addZDoffsetAfter('2021-08-05 00:00:00',-0.014)
    labels.append('12 GHz')

    f=os.path.join(cfg['DATA']['cross_scan_data_dir'],'FastScan_22.00.off')
    P.append(pointingCorrections.fastScanCorrections(f,tmscale=tmscale))
    labels.append('22 GHz')


    of=os.path.join(cfg['DATA']['cross_scan_data_dir'],'corrections.jpg')
    plot_corrections(P, labels, outfile=of)


    P=[]
    labels=[]
    f=os.path.join(cfg['DATA']['cross_scan_data_dir'],'FastScan_6.70.off')
    P.append(pointingCorrections.fastScanCorrections(f,tmscale=tmscale))
    labels.append('6.7 GHz')

    t0=datetime.datetime(2015,1,1)
    off=[ 18.71/1000*(t-t0).total_seconds()/(86400*365) for t in P[-1].dt]
    nodrift=copy.copy(P[-1])
    nodrift.addZDoffsetAfter('2016-08-05 00:00:00',off)
    nodrift.dZD-=np.median(nodrift.dZD)
    P.append(nodrift)
    labels.append('6.7 GHz - linear drift model removed (A=18.71 mdeg/yr)')
    print('rms ',np.std(nodrift.dZD))
    print('rms inside [-0.017, 0.017]: ',np.std(nodrift.dZD[np.abs(nodrift.dZD)<0.017]))

    of=os.path.join(cfg['DATA']['cross_scan_data_dir'],'corrections-no_ZDdrift.jpg')
    plot_corrections(P, labels, outfile=of)

    Tinter=temperature.tempModel(args.temperatures_file)
    dt=P[-1].get_dt()
    temp_offset=(0.283*np.array(Tinter(dt))-34.45)/1000.
    print(temp_offset)
    notemp=copy.deepcopy(nodrift)
    notemp.dZD+=temp_offset
    notemp.dZD-=np.median(notemp.dZD)
    P.append(notemp)
    labels.append('6.7 GHz - linear drift model and temp. dependence removed (A=18.71 mdeg/yr, C=0.283 mdeg/°C)')
    print('rms ',np.std(notemp.dZD))
    print('rms inside [-0.017, 0.017]: ',np.std(notemp.dZD[np.abs(notemp.dZD)<0.017]))

    of=os.path.join(cfg['DATA']['cross_scan_data_dir'],'corrections-no_ZDdrift-no_temp.jpg')
    plot_corrections(P, labels, outfile=of, overplot_custom_model=True)


if __name__ == "__main__":
    sys.exit(main())
    
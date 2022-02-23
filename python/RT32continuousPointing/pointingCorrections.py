'''
Created on Feb 2, 2022

@author: blew
'''

import os,sys
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

class fastScanCorrections():
    '''
    read pointing corrections provided by fast scan program
    '''


    def __init__(self, path,**kwargs):
        '''
        Constructor
        '''
        self.data=[]
        self.path=path
        if os.path.isfile(path):
            with open(path,"r") as f:
                self.data=f.readlines()

        
        self.pointing_data=[]
        self.dZD=[]
        self.time_offset=90
        if 'tmscale' in kwargs.keys():
            self.time_offset=kwargs['tmscale']

        self.process_data()
        
        self.verbose=0
        if 'verbose' in kwargs.keys():
            self.verbose=kwargs['verbose']
    
    def get_dt(self):
        return self.dt
    
    def process_data(self):
        tdelta=datetime.timedelta(days=self.time_offset)
        dt0=datetime.datetime.utcnow()-tdelta
        
        pointing_data=[ re.sub(',','',x).split(' ') for x in self.data 
                       if (len(x.split(' '))==13 and x[0]!='#') ]
        
        todtstr=lambda x: '{}-{}-{} {}'.format(x[3],x[2],x[1],x[4])
        dt=lambda x: datetime.datetime.strptime(todtstr(x),'%Y-%m-%d %H:%M:%S')
        
        '''
        select by time
        '''
        
        self.pointing_data=[ x for x in pointing_data if dt(x)>dt0]
        self.dt=[ dt(x) for x in pointing_data if dt(x)>dt0]
        
        '''
        extract dZD
        '''
        self.dZD=np.array([ float(x[12]) for x in self.pointing_data],dtype=float)
    
        '''
        extract dAZ*sin(ZD)
        '''
        self.dCrossElev=np.array([ float(x[6])*np.sin(float(x[11])*np.pi/180) for x in self.pointing_data],dtype=float)

    def addZDofset(self,offset):
        self.dZD+=offset
        return self
    
    def addContinuousCorrections(self,contcorr):
        '''
        add continuous corrections using their use history as 
        modeled by contcorr interpolation object
        '''
        cont=contcorr(self.dt)
        # print(self.dt)
        # print(cont)
        self.dZD=[x+c for x,c in zip(self.dZD,cont[0])]
        self.dCrossElev=[x+c for x,c in zip(self.dCrossElev,cont[1])]

    def addZDoffsetAfter(self,dtstr,offset,fmt='%Y-%m-%d %H:%M:%S'):
        dt=datetime.datetime.strptime(dtstr,fmt)
        print(dt)
        off=lambda x,t,off: x+off if t>dt else x
        if isinstance(offset,float):    
                self.dZD=[ off(x,t,offset) for x,t in zip(self.dZD,self.dt) ]
        else:
                self.dZD=[ off(x,t,o) for x,t,o in zip(self.dZD,self.dt,offset) ]
        print(len(self.dZD))
        print(len(self.dt))
        
    
    def get_median(self):
        '''
        returns tuple of median cross-elevation and median ZD corrections
        '''
        return np.median(self.dCrossElev),np.median(self.dZD)

    def get_std(self):
        '''
        returns tuple of std of cross-elevation and std pf ZD corrections
        '''
        return np.std(self.dCrossElev),np.std(self.dZD)
        
    def __repr__(self):
        # return ''.join(self.pointing_data)
        if self.verbose>1:
            for c1,c2 in zip(self.dCrossElev,self.dZD):
                print('dxZD: {:.1f}, dZD: {:.1f}'.format(c1*10000,c2*10000))
        mCrossElev,mdZD=self.get_median()
        sCrossElev,sdZD=self.get_std()
        s='Data file: {}\n'.format(self.path)
        s+='Pointing observations: {}\n'.format(len(self.dCrossElev))
        s+='Median ZD correction [mdeg]: {} (sigma: {:.1f})\n'.format(mdZD*1000,sdZD*1000)
        s+='Median cross-elevation correction [mdeg]: {} (sigma: {:.1f})\n'.format(mCrossElev*1000,sCrossElev*1000)
        return s
    
    def __str__(self):
        # return ''.join(self.pointing_data)
        return self.__repr__()

            

def get_median_corrections(args,cfg):
    '''
    calculate median pointing correction
    
    parameters
    ----------
        args - ArgumentParser
        cfg - config_file
        
    returns
    -------
        tuple median pointing correction in cross-zenith-distance and zenith distance [deg]  
        (dxZD,dZD)
    '''
    correction_freq=['5','6_7','12']
    # correction_files='cross_scan_data_file'+correction_freq
    for freq in correction_freq:
        f=os.path.join(cfg['DATA']['cross_scan_data_dir'],cfg['DATA']['cross_scan_data_file'+freq])
        tmscale=cfg.getint('ZED','time_scale')
        P=fastScanCorrections(f,tmscale=tmscale, verbose=args.verbose)
        print('Pointing corrections '+freq)
        print(P)

    
    print('-----------------')    
    f=os.path.join(cfg['DATA']['cross_scan_data_dir'],cfg['DATA']['cross_scan_data_file'])
    tmscale=cfg.getint('ZED','time_scale')
    P=fastScanCorrections(f,tmscale=tmscale, verbose=args.verbose)
    print('Pointing corrections ')
    print(P)

    f=os.path.join(cfg['DATA']['pointing_data_dir'],cfg['DATA']['pointing_data_file'])
    contCorr=continuousCorrections(f)
    
    P.addContinuousCorrections(contCorr)
    print('Continuous corrections')
    print(P)
    mCrossElev,mdZD=P.get_median()

    if args.plot:
        plot_corrections(P)
    
    return mCrossElev,mdZD


def plot_corrections(P):
    '''
    '''
    plt.plot(P.dt,P.dZD,label='$\Delta$ZD')
    plt.legend()
    plt.show()
    
def saveRT32pointingData(fname, kv : dict):
    '''
    Save RT32 control system pointing corrections dictionary to file.
    '''
    of=fname
    now=datetime.datetime.utcnow()
    now_str=datetime.datetime.strftime(now,'%Y-%m-%dT%H:%M:%S')
    kv['dt']=now_str
    for k,v in kv.items():
        kv[k]=list([v])
    print(kv)
    df=pd.DataFrame.from_dict(kv)
    df.to_csv(of, mode='a', header=not os.path.isfile(of),index=False)
    
    
def saveContinuousCorrections(fname, dZD, dxZD):
    of=fname
    with open(of,'a') as f:
        now=datetime.datetime.utcnow()
        s='{} {} {}\n'.format(
            datetime.datetime.strftime(now,'%Y-%m-%dT%H:%M:%S'),
            dZD,
            dxZD)
        f.write(s)
    

class continuousCorrections():
    '''
    continuous corrections interpolation object.
    Given a path to log file with history of continuous corrections changes 
    the class allows to obtain value for any given date
    '''
    def __init__(self, in_file=None):
        self.in_file=in_file
        
        self.temp=np.loadtxt(in_file, dtype=str)
        self.dt=[ datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S') for x in self.temp[:,0] ]
        self.dt0=self.dt[0]
        self.x=[(dt-self.dt0).total_seconds() for dt in self.dt]
        self.dZD=np.array(self.temp[:,1],dtype=float)
        self.dxZD=np.array(self.temp[:,2],dtype=float)
        self.val_inter_dZD=interpolate.interp1d(self.x,self.dZD,fill_value="extrapolate", kind='previous')
        self.val_inter_dxZD=interpolate.interp1d(self.x,self.dxZD,fill_value="extrapolate", kind='previous')
    def toX(self,dt : list):
        return [(x-self.dt0).total_seconds() for x in dt ]
    
    def __call__(self,dt : list):    
        '''
        calculate continuous corrections used up to given datetime
        
        parameters
        ----------
            dt - datetime object for which interpolated values are to be calculated
        
        returns
        -------
            tuple(dZD,dxZD)
        '''
        return self.val_inter_dZD(self.toX(dt)),self.val_inter_dxZD(self.toX(dt))
    
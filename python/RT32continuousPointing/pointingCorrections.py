'''
Created on Feb 2, 2022

@author: blew
'''

import os,sys
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt

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
        
    def __repr__(self):
        # return ''.join(self.pointing_data)
        mCrossElev,mdZD=self.get_median()
        s='Data file: {}\n'.format(self.path)
        s+='Pointing observations: {}\n'.format(len(self.dCrossElev))
        s+='Median ZD correction [mdeg]: {}\n'.format(mdZD*1000)
        s+='Median cross-elevation correction [mdeg]: {}\n'.format(mCrossElev*1000)
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
        median pointing correction [deg]
    '''
    f=os.path.join(cfg['DATA']['cross_scan_data_dir'],cfg['DATA']['cross_scan_data_file'])
    tmscale=cfg.getint('ZED','time_scale')
    P=fastScanCorrections(f,tmscale=tmscale)
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
    
    
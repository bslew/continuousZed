'''
Created on Feb 2, 2022

@author: blew
'''

import os,sys
import re
import datetime
import numpy as np

class fastScanCorrections():
    '''
    read pointing corrections provided by fast scan program
    '''


    def __init__(self, path,**kwargs):
        '''
        Constructor
        '''
        self.data=[]
        if os.path.isfile(path):
            with open(path,"r") as f:
                self.data=f.readlines()

        
        self.pointing_data=[]
        self.dZD=[]
        self.time_offset=90
        if 'tmscale' in kwargs.keys():
            self.time_offset=kwargs['tmscale']

        self.process_data()
        
        
    def process_data(self):
        tdelta=datetime.timedelta(days=self.time_offset)
        dt0=datetime.datetime.utcnow()-tdelta
        
        pointing_data=[ re.sub(',','',x).split(' ') for x in self.data]
        
        todtstr=lambda x: '{}-{}-{} {}'.format(x[3],x[2],x[1],x[4])
        dt=lambda x: datetime.datetime.strptime(todtstr(x),'%Y-%m-%d %H:%M:%S')
        
        '''
        select by time
        '''
        self.pointing_data=[ x for x in pointing_data if dt(x)>dt0]
        
        '''
        extract dZD
        '''
        self.dZD=np.array([ float(x[12]) for x in self.pointing_data],dtype=float)
    
    
    def get_median(self):
        return np.median(self.dZD)
        
    def __repr__(self):
        # return ''.join(self.pointing_data)
        return 'Median ZD correction [mdeg]: {}'.format(np.median(self.dZD))
    
    def __str__(self):
        # return ''.join(self.pointing_data)
        return 'Median ZD correction [mdeg]: {}'.format(np.median(self.dZD))
            

def get_median_ZD_correction(args,cfg):
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
    mdZD=P.get_median()
    return mdZD
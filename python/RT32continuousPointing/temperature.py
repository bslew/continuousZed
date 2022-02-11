'''
Created on Feb 8, 2022

@author: blew
'''
import numpy as np
from scipy import interpolate
import datetime

class tempModel():
    def __init__(self, temp_file=None):
        self.temp_file=temp_file
        
        self.temp=np.loadtxt(temp_file, dtype=str)
        self.dt=[ datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S') for x in self.temp[:,0] ]
        self.dt0=self.dt[0]
        self.x=[(dt-self.dt0).total_seconds() for dt in self.dt]
        self.y=self.temp[:,1]
        self.Tinter=interpolate.interp1d(self.x,self.y,fill_value="extrapolate")
        print("we're defined over range:")
        print(self.dt[0])
        print(self.dt[-1])
    def toX(self,dt : list):
        return [(x-self.dt0).total_seconds() for x in dt ]
    
    def __call__(self,dt : list):    
        return self.Tinter(self.toX(dt))
    
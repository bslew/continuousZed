'''
Created on Jun 10, 2016

@author: blew
'''

import numpy as np
import random

class confidenceRange():
    '''
    classdocs
    '''


    def __init__(self,data=[]):
        '''
        Constructor
        '''
        self.data=data
        
    '''
    CL - confidence level in per-cent. Eg 68.0 will return 68.0% CR for the provided data
    '''
    def getTwoSidedConfidenceRange(self,CL):
        return [np.percentile(self.data, 50.0-CL/2),np.percentile(self.data, 50.0+CL/2)]

    def getTwoSidedOneSigmaConfidenceRange(self):
        CL=68.2689
        return [np.percentile(self.data, 50.0-CL/2),np.percentile(self.data, 50.0+CL/2)]
    def getTwoSidedTwoSigmaConfidenceRange(self):
        CL=95.45
        return [np.percentile(self.data, 50.0-CL/2),np.percentile(self.data, 50.0+CL/2)]
    def getTwoSidedThreeSigmaConfidenceRange(self):
        CL=99.73
        return [np.percentile(self.data, 50.0-CL/2),np.percentile(self.data, 50.0+CL/2)]

    
    def test(self):
        d=[random.gauss(0.0,1.0) for x in range(10000)]
        print(d)
        np.savetxt('d.tmp', d)
        print(np.percentile(d, 50.0))
        print(np.percentile(d, 50.0+34.0))
        print(np.percentile(d, 50.0-34.0))

'''
Created on Feb 2, 2022

@author: blew
'''

import os,sys
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import interpolate
import pandas as pd
from RT32continuousPointing import confidenceRange
import pickle


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
        self.cont_corrections=None
        self.dZD=[]
        self.time_offset=90
        if 'tmscale' in kwargs.keys():
            self.time_offset=kwargs['tmscale']

        self.start_time=None
        if 'start_time' in kwargs.keys():
            if kwargs['start_time']!=None:
                self.start_time=datetime.datetime.strptime(kwargs['start_time'],'%Y-%m-%d %H:%M:%S')
        self.end_time=None
        if 'end_time' in kwargs.keys():
            if kwargs['end_time']!=None:
                self.end_time=datetime.datetime.strptime(kwargs['end_time'],'%Y-%m-%d %H:%M:%S')

        self.process_data()
        
        self.verbose=0
        if 'verbose' in kwargs.keys():
            self.verbose=kwargs['verbose']
    
    def get_dt(self):
        return self.dt
    
    def process_data(self):
        tdelta=datetime.timedelta(days=self.time_offset)
        dt0=datetime.datetime.utcnow()-tdelta
        dt1=datetime.datetime.utcnow()
        if self.start_time:
            if self.start_time>dt0:
                dt0=self.start_time
        if self.end_time:
            if self.end_time<dt1:
                dt1=self.end_time
        
        pointing_data=[ re.sub(',','',x).split(' ') for x in self.data 
                       if (len(x.split(' '))==13 and x[0]!='#') ]
        
        todtstr=lambda x: '{}-{}-{} {}'.format(x[3],x[2],x[1],x[4])
        dt=lambda x: datetime.datetime.strptime(todtstr(x),'%Y-%m-%d %H:%M:%S')
        '''
        select by time
        '''
        self.pointing_data_all=pointing_data
        self.mask=[ dt(x)>dt0 and dt(x)<dt1 for x in pointing_data ]

        self.pointing_data=[ x for x in pointing_data if dt(x)>dt0 and dt(x)<dt1]
        self.dt=[ dt(x) for x in pointing_data if dt(x)>dt0 and dt(x)<dt1]
        
        '''
        extract dZD
        '''
        self.dZD=np.array([ float(x[12]) for x in self.pointing_data],dtype=float)
    
        '''
        extract dAZ*sin(ZD)
        '''
        self.dxZD=np.array([ float(x[6])*np.sin(float(x[11])*np.pi/180) for x in self.pointing_data],dtype=float)

    def save(self,fname):
        '''
        re-assemble the pointing data the save to output file in the format compatible with
        the input format
        '''
        # self.pointing_data_all


    def stats_plots(self,receiver='',freq='', outfile=''):
        '''
        '''
        # if self.verbose>1:
        
        stats={}
        
        
        '''
        cross ZD corrections vs time
        '''
        fig=plt.figure(figsize=(12,8))
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        x=self.dt
        y=self.dxZD*1000
        ax1=plt.subplot(211)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        lab='receiver: {} ({} GHz)'.format(receiver,freq)
        plt.plot(x,y,'o-', label=lab)
        
        #plot annotations
        m=np.median(y)
        stats['Nobs']=len(x)
        stats['dt_start']=x[0]
        stats['dt_end']=x[-1]
        stats['dxZD']={}
        stats['dxZD']['median']='%.2f' % m
        stats['dxZD']['sigma']='%.2f' % y.std()

        
        plt.axhline(np.median(y),lw=2,c='k')
        plt.annotate('median={:.1f} mdeg'.format(m),xy=(0.01,0.01), xycoords=('axes fraction','axes fraction'), fontsize=12)
        l,h=np.array(confidenceRange.confidenceRange(y).getTwoSidedTwoSigmaConfidenceRange())
        stats['dxZD']['1sigma']=(l,h)
        plt.axhspan(l,h,alpha=0.2, color='green')
        l,h=np.array(confidenceRange.confidenceRange(y).getTwoSidedOneSigmaConfidenceRange())
        stats['dxZD']['2sigma']=(l,h)
        plt.axhspan(l,h,alpha=0.2, color='green')
        # plt.annotate('95% CR',xy=(0.02,(m+h)/2), xycoords=('axes fraction','data'), fontsize=15)
        
        plt.legend()
        plt.grid()
        plt.xlabel('time [UTC]')
        plt.ylabel('cross-ZD correction [mdeg]')

        '''
        ZD plot annotations
        '''

        plt.subplot(212,sharex=ax1)
        y=self.dZD*1000
        plt.plot(x,y,'o-')

        #plot annotations
        m=np.median(y)
        stats['dZD']={}
        stats['dZD']['median']='%.2f' % m
        stats['dZD']['sigma']='%.2f' % y.std()

        plt.axhline(m,lw=2,c='k')
        # plt.annotate('median={:.1f}'.format(m),xy=(0.01,0.01), xycoords=('axes fraction','data'))
        plt.annotate('median={:.1f} mdeg'.format(m),xy=(0.01,0.01), xycoords=('axes fraction','axes fraction'), fontsize=12)
        l,h=np.array(confidenceRange.confidenceRange(y).getTwoSidedTwoSigmaConfidenceRange())
        stats['dZD']['1sigma']=(l,h)
        plt.axhspan(l,h,alpha=0.2, color='green')
        l,h=np.array(confidenceRange.confidenceRange(y).getTwoSidedOneSigmaConfidenceRange())
        stats['dZD']['1sigma']=(l,h)
        plt.axhspan(l,h,alpha=0.2, color='green')
        # plt.annotate('95% CR',xy=(0.02,(m+h)/2), xycoords=('axes fraction','data'), fontsize=15)
        
        plt.grid()
        plt.xlabel('time [UTC]')
        plt.ylabel('ZD correction [mdeg]')


        if outfile!='':
            plt.savefig(outfile)
        # self.
        
        return stats

    def addZDoffset(self,offset):
        self.dZD+=offset
        return self

    def addxZDoffset(self,offset):
        self.dxZD+=offset
        return self
        

    def subContinuousCorrections(self,contcorr):
        '''
        subtract continuous corrections using their use history as 
        modeled by contcorr interpolation object
        
        TODO: make this smarter...
        '''
        cont=contcorr(self.dt)
        self.cont_corrections=cont
        # print(self.dt)
        if self.verbose>1:
            print(cont)
        self.dZD=[x-c for x,c in zip(self.dZD,cont[0])]
        self.dxZD=[x-c for x,c in zip(self.dxZD,cont[1])]
        
    
    def addContinuousCorrections(self,contcorr):
        '''
        add continuous corrections using their use history as 
        modeled by contcorr interpolation object
        '''
        cont=contcorr(self.dt)
        self.cont_corrections=cont
        # print(self.dt)
        if self.verbose>1:
            print(cont)
        self.dZD=[x+c for x,c in zip(self.dZD,cont[0])]
        self.dxZD=[x+c for x,c in zip(self.dxZD,cont[1])]

    def addZDoffsetAfter(self,dtstr,offset,fmt='%Y-%m-%d %H:%M:%S'):
        dt=datetime.datetime.strptime(dtstr,fmt)
        if self.verbose>1:
            print(dt)
        off=lambda x,t,off: x+off if t>dt else x
        if isinstance(offset,float):    
                self.dZD=[ off(x,t,offset) for x,t in zip(self.dZD,self.dt) ]
        else:
                self.dZD=[ off(x,t,o) for x,t,o in zip(self.dZD,self.dt,offset) ]
        if self.verbose>1:
            print(len(self.dZD))
            print(len(self.dt))
        
    
    def get_median(self):
        '''
        returns tuple of median cross-elevation and median ZD corrections
        '''
        return np.median(self.dxZD),np.median(self.dZD)

    def get_std(self):
        '''
        returns tuple of std of cross-elevation and std pf ZD corrections
        '''
        return np.std(self.dxZD),np.std(self.dZD)
        
    # def get_cont_corrections(self):
    #     '''
    #     returns a continuous corrections as a tuple even it it was not loaded
    #     In such case (0,0) corrections are returned
    #     '''
    #     if self.cont_corrections:
    #         return self.cont_corrections
    #     return (0.,0.)
    
    def __repr__(self):
        # return ''.join(self.pointing_data)
        
        if self.verbose>1:
            for i,(c1,c2) in enumerate(zip(self.dxZD,self.dZD)):
                if self.cont_corrections:
                    print('dxZD: {:.4f}, dZD: {:.4f}, cont.dxZD: {:.4f}, cont.dZD: {:.4f}, dt: {}'.format(
                        c1,c2,
                        self.cont_corrections[1][i],
                        self.cont_corrections[0][i],
                        self.dt[i],
                        ))
                else:
                    print('dxZD: {:.4f}, dZD: {:.4f}, dt: {}'.format(
                        c1,c2,
                        self.dt[i],
                        ))
                    
        mCrossElev,mdZD=self.get_median()
        sCrossElev,sdZD=self.get_std()
        s='Data file: {}\n'.format(self.path)
        s+='Pointing observations: {}\n'.format(len(self.dxZD))
        s+='Median ZD correction [mdeg]: {:.1f} (sigma: {:.1f})\n'.format(mdZD*1000,sdZD*1000)
        s+='Median cross-elevation correction [mdeg]: {:.1f} (sigma: {:.1f})\n'.format(mCrossElev*1000,sCrossElev*1000)
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
    correction_freq=['5','12','6_7']
    receivers=['C','X','M']
    # correction_files='cross_scan_data_file'+correction_freq
        # if args.plot_stats:
        #
        # receivers=['C','M','X']
        # P=[]
        # labels=[]
        # tmscale=cfg.getint('ZED','time_scale')
        #
        # for rec in receivers:    
        #     f=os.path.join(cfg['DATA']['data_dir'],cfg[rec]['cross_scan_data_file'])
        #     P.append(pointingCorrections.fastScanCorrections(f,tmscale=tmscale))
        #     labels.append(cfg[rec]['freq']+' GHz')

    start_time=None
    if cfg.has_option('ZED','start_time'):
        start_time=cfg['ZED']['start_time']
    end_time=None
    if cfg.has_option('ZED','end_time'):
        end_time=cfg['ZED']['end_time']
    stats={}
    for i,rec in enumerate(receivers):
        stats[rec]={}
        print('-----------------')    
        print('Receiver: {}'.format(rec))
        freq=cfg[rec]['freq']
        f=os.path.join(cfg['DATA']['data_dir'],cfg[rec]['cross_scan_data_file'])
        tmscale=cfg.getint('ZED','time_scale')
        P=fastScanCorrections(f,tmscale=tmscale, 
                              start_time=start_time,
                              end_time=end_time,
                              verbose=args.verbose)
        print('Time scale: last {} days'.format(tmscale))
        print('Frequency [GHz]: '+freq)
        print(P)

        f=os.path.join(cfg['DATA']['data_dir'],cfg[rec]['roh_hist'])
        rohCorr=continuousCorrections(f)
        rZD,rxZD=rohCorr.last()
        P.addContinuousCorrections(rohCorr)
        P.addxZDoffset(-rxZD)
        P.addZDoffset(-rZD)

        # P.subContinuousCorrections(rohCorr)
        # P.addxZDoffset(rxZD)
        # P.addZDoffset(rZD)

        print('Pointing corrections (unified to current roh epoch)')
        print(P)
        f=os.path.join(cfg['DATA']['data_dir'],'corrections_'+rec+'.'+cfg['DATA']['roh_unified_corrections_file_suffix']+'.jpg')
        stats[rec]=P.stats_plots(receiver=rec,freq=freq,outfile=f)
        if args.verbose>2:
            plt.show()
        


        f=os.path.join(cfg['DATA']['data_dir'],cfg['DATA']['cont_corr_data_file'])
        contCorr=continuousCorrections(f)
        P.addContinuousCorrections(contCorr)
        print('Continuous corrections')
        print(P)
        print('-----------------')    

    stats_file=os.path.join(cfg['DATA']['data_dir'],'corrections_stats.'+cfg['DATA']['roh_unified_corrections_file_suffix']+'.pkl')
    with open(stats_file,'wb') as f:
        pickle.dump(stats,f)
    
    # print('-----------------')    
    # f=os.path.join(cfg['DATA']['data_dir'],cfg['DATA']['cross_scan_data_file'])
    # tmscale=cfg.getint('ZED','time_scale')
    # P=fastScanCorrections(f,tmscale=tmscale, verbose=args.verbose)
    # print('Pointing corrections ')
    # print(P)
    #
    # f=os.path.join(cfg['DATA']['data_dir'],cfg['DATA']['cont_corr_data_file'])
    # contCorr=continuousCorrections(f)
    # P.addContinuousCorrections(contCorr)
    # print('Continuous corrections')
    # print(P)
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
    '''
    save continuous corrections if they have changed since the last save
    '''
    of=fname
    
    '''
    read last entry
    '''
    last_line=None
    try:
        with open(of,'r') as f:
            last_line=f.readlines()[-1]
    except:
        pass
    
    last_cont_corr=(0,0)
    if last_line:
        try:
            last_cont_corr=last_line.split()[1:]
        except:
            last_cont_corr=(0,0)
            pass
    # print('last_cont_corr:',last_cont_corr)
    
    if (float(dxZD),float(dZD))!=tuple([float(c) for c in last_cont_corr]):
        with open(of,'a') as f:
            now=datetime.datetime.utcnow()
            s='{} {} {}\n'.format(
                datetime.datetime.strftime(now,'%Y-%m-%dT%H:%M:%S'),
                dxZD,
                dZD)
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
        self.temp=self.temp.reshape((-1,3))
        if len(self.temp)==1:
            self.temp=np.vstack([self.temp[0],self.temp[0]])
            self.temp[0,0]='2020-01-01T00:00:00'
        self.dt=[ datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S') for x in self.temp[:,0] ]
        self.dt0=self.dt[0]
        self.x=[(dt-self.dt0).total_seconds() for dt in self.dt]
        self.dZD=np.array(self.temp[:,2],dtype=float)
        self.dxZD=np.array(self.temp[:,1],dtype=float)
        self.val_inter_dZD=interpolate.interp1d(self.x,self.dZD,fill_value="extrapolate", kind='previous')
        self.val_inter_dxZD=interpolate.interp1d(self.x,self.dxZD,fill_value="extrapolate", kind='previous')
    def toX(self,dt : list):
        return [(x-self.dt0).total_seconds() for x in dt ]
    
    def last(self) -> tuple:
        '''
        return last known correction
        returns
        -------
            tuple(dZD,dxZD)
        '''
        return self.dZD[-1],self.dxZD[-1]
        
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
    
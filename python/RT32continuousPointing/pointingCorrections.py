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
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy import interpolate
import pandas as pd
from RT32continuousPointing import confidenceRange
import pickle
import statsmodels.api as smapi
import copy


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
            
        self.filter_nsigma=3.
        if 'filter_nsigma' in kwargs.keys():
            self.filter_nsigma=kwargs['filter_nsigma']

        self.start_time=None
        if 'start_time' in kwargs.keys():
            if kwargs['start_time']!=None:
                self.start_time=datetime.datetime.strptime(kwargs['start_time'],'%Y-%m-%d %H:%M:%S')
        self.end_time=None
        if 'end_time' in kwargs.keys():
            if kwargs['end_time']!=None:
                self.end_time=datetime.datetime.strptime(kwargs['end_time'],'%Y-%m-%d %H:%M:%S')

        
        self.verbose=0
        if 'verbose' in kwargs.keys():
            self.verbose=kwargs['verbose']
            print("start_time: ",self.start_time)

        self.process_data()
    
    def get_dt(self):
        return self.dt
    
    def process_data(self):
        tdelta=datetime.timedelta(days=self.time_offset)
        dt0=datetime.datetime.utcnow()-tdelta
        dt1=datetime.datetime.utcnow()
        if self.start_time:
            # if self.start_time>dt0:
            dt0=self.start_time
        if self.end_time:
            # if self.end_time<dt1:
            dt1=self.end_time
        
        pointing_data=[ re.sub(',','',x).split(' ') for x in self.data 
                       if (len(x.split(' '))==13 and x[0]!='#') ]
        
        todtstr=lambda x: '{}-{}-{} {}'.format(x[3],x[2],x[1],x[4])
        dt=lambda x: datetime.datetime.strptime(todtstr(x),'%Y-%m-%d %H:%M:%S')

        '''
        elevation scan time
        '''
        todtstr_EL=lambda x: '{}-{}-{} {}'.format(x[9],x[8],x[7],x[10])
        dt_EL=lambda x: datetime.datetime.strptime(todtstr_EL(x),'%Y-%m-%d %H:%M:%S')

        '''
        select by time
        '''
        self.pointing_data_all=pointing_data
        self.selected=[ dt(x)>dt0 and dt(x)<dt1 for x in pointing_data ]

        self.pointing_data=[ x for x in pointing_data if dt(x)>dt0 and dt(x)<dt1]
        self.dt=[ dt(x) for x in self.pointing_data]
        


        '''
        filter outliers
        '''
        nsigma=self.filter_nsigma
        self.dZD=self.extract_dZD()
        
        if len(self.dZD)>1:
        
            l,h=np.array(confidenceRange.confidenceRange(self.dZD).getTwoSidedOneSigmaConfidenceRange())
            m=np.median(self.dZD)
            l=l-nsigma*(m-l)
            h=h+nsigma*(h-m)
            selected=np.logical_and(self.dZD<h,self.dZD>l)
            # self.pointing_data=[ x for i,x in enumerate(self.pointing_data) if selected[i]]
            if self.verbose>2:
                print('dZD outliers removal')
                print(l,h)
                print(selected)
                print(self.dZD)
                print('removed {} dZD outliers'.format(len(selected)-len(selected[selected==True])))
            
            self.dxZD=self.extract_dxZD()
            l,h=np.array(confidenceRange.confidenceRange(self.dxZD).getTwoSidedOneSigmaConfidenceRange())
            m=np.median(self.dxZD)
            l=l-nsigma*(m-l)
            h=h+nsigma*(h-m)
            selected=np.logical_and(selected,self.dxZD<h,self.dxZD>l)
            if self.verbose>2:
                print('dxZD outliers removal')
                print(l,h)
                print(selected)
                print(self.dxZD)
                print('removed {} dxZD outliers'.format(len(selected)-len(selected[selected==True])))
    
            self.pointing_data=[ x for i,x in enumerate(self.pointing_data) if selected[i]]

        '''
        extract dZD
        '''
        self.dZD=self.extract_dZD()
    
        '''
        extract dAZ*sin(ZD)
        '''
        self.dxZD=self.extract_dxZD()


        self.dt=[ dt(x) for x in self.pointing_data]
        self.dt_EL=[ dt_EL(x) for x in self.pointing_data]

        
    # def extract_dt(self) -> list:
    #     return [ dt(x) for x in self.pointing_data]

    def extract_src(self):
        return np.array([ x[0] for x in self.pointing_data],dtype=str)

    def extract_AZ(self):
        return np.array([ float(x[5]) for x in self.pointing_data],dtype=float)

    def extract_ZD(self):
        return np.array([ float(x[11]) for x in self.pointing_data],dtype=float)

    def extract_dZD(self):
        return np.array([ float(x[12]) for x in self.pointing_data],dtype=float)

    def extract_dxZD(self):
        return np.array([ float(x[6])*np.sin(float(x[11])*np.pi/180) for x in self.pointing_data],dtype=float)

    def save(self,fname):
        '''
        re-assemble the pointing data the save to output file in the format compatible with
        the input format
        '''
        # self.pointing_data_all
        src=self.extract_src()
        sZ=np.sin(self.extract_ZD()*np.pi/180)
        dA=self.dxZD
        dZ=self.dZD
        with open(fname,'w') as f:
            for i in range(len(self.pointing_data)):
                l='%s ' % src[i]
                l+='%s %s %s, %s ' % (self.pointing_data[i][1],self.pointing_data[i][2],self.pointing_data[i][3],self.pointing_data[i][4])
                l+='%s %.5f ' % (self.pointing_data[i][5],self.dxZD[i]/sZ[i])
                l+='%s %s %s, %s ' % (self.pointing_data[i][7],self.pointing_data[i][8],self.pointing_data[i][9],self.pointing_data[i][10])
                l+='%s %.5f' % (self.pointing_data[i][11],self.dZD[i])
                l+='\n'
                f.write(l)
        '''
        save to CSV as well
        '''
        with open(fname+".csv",'w') as f:
            l='src,dtAZ,AZ,dAZ,dtZD,ZD,dZD,dAZsinZD\n'
            f.write(l)
            for i in range(len(self.pointing_data)):
                l='%s,' % src[i]
                l+='%s-%s-%s %s,' % (self.pointing_data[i][3],self.pointing_data[i][2],self.pointing_data[i][1],self.pointing_data[i][4])
                l+='%s,%.5f,' % (self.pointing_data[i][5],self.dxZD[i]/sZ[i])
                l+='%s-%s-%s %s,' % (self.pointing_data[i][9],self.pointing_data[i][8],self.pointing_data[i][7],self.pointing_data[i][10])
                l+='%s,%.5f,' % (self.pointing_data[i][11],self.dZD[i])
                l+='%.5f' % (self.dxZD[i])
                l+='\n'
                f.write(l)
                

    def stats_plots(self,receiver='',freq='', outfile='', fwhp=None,**kwargs):
        '''
        returns dictionary with corrections statistics
        '''
        # if self.verbose>1:
        show=False
        if 'show' in kwargs.keys():
            show=kwargs['show']
        
        model=None
        if 'model' in kwargs.keys():
            model=kwargs['model']
            
        historical_data=False
        if 'historical_data' in kwargs.keys():
            historical_data=kwargs['historical_data']
        
        
            
        stats={}
        
        x=self.dt
        y=self.dxZD*1000

        '''
        cross ZD corrections vs time
        '''
        # if show:
        fig=plt.figure(figsize=(12,8))
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1=plt.subplot(211)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        lab='receiver: {} ({} GHz)'.format(receiver,freq)
        plt.plot(x,y,'o-', label=lab)
        if model:
            plt.plot(x,model['dxZD']['pred']*1000,'k--', label='model')
        
        #show annotations
        m=np.median(y)
        rms=np.sqrt(np.mean(y**2))
        stats['Nobs']=len(x)
        stats['dt_start']=x[0]
        stats['dt_end']=x[-1]
        stats['dxZD']={}
        stats['dxZD']['median']='%.2f' % m
        stats['dxZD']['sigma']='%.2f' % y.std()
        stats['dxZD']['rms']='%.2f' % rms

        
        # if show:
        if not historical_data:
            plt.axhline(np.median(y),lw=2,c='k')
            plt.annotate('median={:.1f} mdeg'.format(m),xy=(0.01,0.01), xycoords=('axes fraction','axes fraction'), fontsize=12)
            l,h=np.array(confidenceRange.confidenceRange(y).getTwoSidedTwoSigmaConfidenceRange())
            stats['dxZD']['1sigma']=(l,h)
            # if show:
            plt.axhspan(l,h,alpha=0.2, color='green')
            l,h=np.array(confidenceRange.confidenceRange(y).getTwoSidedOneSigmaConfidenceRange())
            stats['dxZD']['2sigma']=(l,h)
            # if show:
            plt.axhspan(l,h,alpha=0.2, color='green')
        # plt.annotate('95% CR',xy=(0.02,(m+h)/2), xycoords=('axes fraction','data'), fontsize=15)
        
        # if show:
        plt.legend(loc='lower right')
        plt.grid()
        plt.xlabel('time [UTC]')
        plt.ylabel('cross-ZD correction [mdeg]')

        '''
        ZD show annotations
        '''

        y=self.dZD*1000
        # if show:
        plt.subplot(212,sharex=ax1)
        plt.plot(x,y,'o-')
        if model:
            plt.plot(x,model['dZD']['pred']*1000,'k--', label='model')

        #show annotations
        m=np.median(y)
        rms=np.sqrt(np.mean(y**2))
        stats['dZD']={}
        stats['dZD']['median']='%.2f' % m
        stats['dZD']['sigma']='%.2f' % y.std()
        stats['dZD']['rms']='%.2f' % rms

        if not historical_data:
            plt.axhline(m,lw=2,c='k')
            plt.annotate('median={:.1f} mdeg'.format(m),xy=(0.01,0.01), xycoords=('axes fraction','axes fraction'), fontsize=12)
            l,h=np.array(confidenceRange.confidenceRange(y).getTwoSidedTwoSigmaConfidenceRange())
            stats['dZD']['1sigma']=(l,h)
            # if show:
            plt.axhspan(l,h,alpha=0.2, color='green')
            l,h=np.array(confidenceRange.confidenceRange(y).getTwoSidedOneSigmaConfidenceRange())
            stats['dZD']['1sigma']=(l,h)
            # if show:
            plt.axhspan(l,h,alpha=0.2, color='green')
            # plt.annotate('95% CR',xy=(0.02,(m+h)/2), xycoords=('axes fraction','data'), fontsize=15)
        
        # if show:
        plt.grid()
        plt.xlabel('time [UTC]')
        plt.ylabel('ZD correction [mdeg]')

        if outfile!='':
            plt.savefig(outfile)
        if show:
            plt.show()

        plt.close()
            
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
        self.cont_corrections=cont #if self.cont_corrections==None else self.cont_corrections-cont
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
        self.cont_corrections=cont #if self.cont_corrections==None else self.cont_corrections+cont
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

    def get_linear_model_prediction(self) -> dict:
        '''
        returns dict({
            'dxZD' : {'model' : statsmodels fit, 'pred' :array_like }, 
            'dZD' : {'model' : statsmodels fit, 'pred' : array_like },
            'name' : str
            })
        '''
        ret={ 'name' : 'linear' }
        ObsTime=[ self.dt, self.dt_EL]
        Y=[self.dxZD,self.dZD]
        lab=['dxZD','dZD']
        
        for DT,l,y in zip(ObsTime,lab,Y):
            tmstamp=[ dt.timestamp() for dt in DT ]
            X=np.vstack((tmstamp,np.ones(len(tmstamp)))).T
            model = smapi.RLM(y, X)
            m = model.fit()
            pred=m.predict()
            ret[l]={}
            ret[l]['model']=m
            ret[l]['pred']=pred
            ret[l]['dt']=DT
        
        return ret

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

    def __len__(self): 
        return len(self.dxZD)

def get_pointing_corrections(args,cfg):
    '''
    calculate pointing corrections and statistics
    
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
    # correction_freq=['6_7']
    # receivers=['M']
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
    
    if args.start_time!='':
        start_time=args.start_time
    if args.end_time!='':
        end_time=args.end_time
        
        
    model='median'
    if cfg.has_option('ZED','model'):
        model=cfg['ZED']['model']
    
    stats={}
    stats['last_update']=datetime.datetime.strftime(datetime.datetime.utcnow(),'%Y-%m-%d %H:%M:%S')
    tmscale=cfg.getint('ZED','time_scale')
    stats['tmscale']=tmscale
    stats['start_time']=start_time
    stats['end_time']=end_time
    stats['receivers']={}
    # stats['correction_freq']=correction_freq
    
    historical_data=args.historical_data
    
    rcu={} # ROH and contonuous corrections unified to most recent epoch
    # rcu_model={} # model of ROH and contonuous corrections unified to most recent epoch
    
    for i,rec in enumerate(receivers):
        print('-----------------')
        print('Raw pointing corrections')
        print('Receiver: {}'.format(rec))
        freq=cfg[rec]['freq']
        f=os.path.join(cfg['DATA']['data_dir'],cfg[rec]['cross_scan_data_file'])
        filter_nsigma=cfg.getfloat('DATA','filter_nsigma')
        P=fastScanCorrections(f,tmscale=tmscale, 
                              start_time=start_time,
                              end_time=end_time,
                              verbose=args.verbose,
                              filter_nsigma=filter_nsigma,
                              )
        if len(P)>0:
            stats['receivers'][rec]={}
            print('Time scale: last {} days'.format(tmscale))
            print('Frequency [GHz]: '+freq)
            print(P)
    
            '''
            ROH history corrected corrections
            '''
            print('Pointing corrections (unified to current roh)')
            f=os.path.join(cfg['DATA']['data_dir'],cfg[rec]['roh_hist'])
            rohCorr=continuousCorrections(f)
            rZD,rxZD=rohCorr.last()
            P.addContinuousCorrections(rohCorr)
            P.addxZDoffset(-rxZD)
            P.addZDoffset(-rZD)
            print(P)
    
            '''
            export corrections in format consistent with fast_scan program
            '''   
            ofile=os.path.join(cfg['DATA']['data_dir'],'corrections'+args.export_suff+'-'+rec+'.'+cfg['DATA']['roh_unified_corrections_file_suffix']+'.off')
            P.save(ofile)
            print("Corrections exported to file: {}".format(ofile))
            stats['receivers'][rec]['corr_rohuni_export_file']=os.path.basename(ofile)

    
            '''
            continuous-corrections-history-corrected corrections
            '''
            print('Pointing corrections (unified to current continuous corrections)')
            f=os.path.join(cfg['DATA']['data_dir'],cfg['DATA']['cont_corr_data_file'])
            contCorr=continuousCorrections(f)
            cZD,cxZD=contCorr.last()
            P.addContinuousCorrections(contCorr)
            P.addxZDoffset(-cxZD)
            P.addZDoffset(-cZD)
    
            '''
            export corrections in format consistent with fast_scan program
            '''   
            ofile=os.path.join(cfg['DATA']['data_dir'],'corrections'+args.export_suff+'-'+rec+'.'+cfg['DATA']['rohcont_unified_corrections_file_suffix']+'.off')
            P.save(ofile)
    
            # P.subContinuousCorrections(rohCorr)
            # P.addxZDoffset(rxZD)
            # P.addZDoffset(rZD)
            print(P)

            # corrections_model=None
            # if cfg['ZED']['model']=='linear':
            corrections_model=P.get_linear_model_prediction()
            
            '''
            calculate stats
            '''
            f=os.path.join(cfg['DATA']['data_dir'],'corrections'+args.export_suff+'-'+rec+'.'+cfg['DATA']['rohcont_unified_corrections_file_suffix']+'.jpg')
            # f=os.path.join(cfg['DATA']['data_dir'],'corrections_'+rec+'.'+cfg['DATA']['rohcont_unified_corrections_file_suffix']+'.jpg')
            stats['receivers'][rec]=P.stats_plots(receiver=rec,freq=freq,outfile=f,show=args.show, args=args, model=corrections_model, historical_data=historical_data)
            stats['receivers'][rec]['fwhp']=cfg[rec]['fwhp']
            stats['receivers'][rec]['freq']=cfg[rec]['freq']
            stats['receivers'][rec]['dxZD']['sigma2fwhp']='%.2f' % (float(stats['receivers'][rec]['dxZD']['sigma'])/cfg.getfloat(rec,'fwhp')/1000)
            stats['receivers'][rec]['dZD']['sigma2fwhp']='%.2f' % (float(stats['receivers'][rec]['dZD']['sigma'])/cfg.getfloat(rec,'fwhp')/1000)
            stats['receivers'][rec]['dxZD']['rms2fwhp']='%.2f' % (float(stats['receivers'][rec]['dxZD']['rms'])/cfg.getfloat(rec,'fwhp')/1000)
            stats['receivers'][rec]['dZD']['rms2fwhp']='%.2f' % (float(stats['receivers'][rec]['dZD']['rms'])/cfg.getfloat(rec,'fwhp')/1000)
            stats['receivers'][rec]['dxZD']['model']=corrections_model['dxZD']['pred']
            stats['receivers'][rec]['dZD']['model']=corrections_model['dZD']['pred']
            stats['receivers'][rec]['dZD']['active_model_name']=model
            print('dxZD rms [mdeg]: ',stats['receivers'][rec]['dxZD']['rms'])
            print('dZD rms [mdeg]: ',stats['receivers'][rec]['dZD']['rms'])
            print('dxZD rms2fwhp: ',stats['receivers'][rec]['dxZD']['rms2fwhp'])
            print('dZD rms2fwhp: ',stats['receivers'][rec]['dZD']['rms2fwhp'])
            print()
            if args.verbose>2:
                print(stats)

            rcu[rec]=copy.deepcopy(P)
            
    
            '''
            calculate current value of continuous corrections
            '''
            print('Continuous corrections')
            P.addxZDoffset(cxZD)
            P.addZDoffset(cZD)
            print(P)
    

            '''
            export corrections 
            '''   
            ofile=os.path.join(cfg['DATA']['data_dir'],'corrections'+args.export_suff+'-'+rec+'.'+cfg['DATA']['rohcontoff_unified_corrections_file_suffix']+'.off')
            P.save(ofile)
            print("Corrections exported to file: {}".format(ofile))
            stats['receivers'][rec]['corr_uni_off_export_file']=os.path.basename(ofile)

            print('-----------------') 
    
    
            if args.Tstruct_file!='':
                Tstruct=pd.read_csv(args.Tstruct_file)
                Tstruct['datetime']=[ datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S") for x in Tstruct['dt']]
                print('Tstruct has {} entries'.format(len(Tstruct)))
                Tstruct['TL1']=(Tstruct['T_11']+Tstruct['T_12']+Tstruct['T_13']+Tstruct['T_14'])/4
                Tstruct['TL2']=(Tstruct['T_21']+Tstruct['T_22']+Tstruct['T_23']+Tstruct['T_24'])/4
                Tstruct['TL3']=(Tstruct['T_31']+Tstruct['T_32']+Tstruct['T_33']+Tstruct['T_34'])/4
                Tstruct['TL4']=(Tstruct['T_41']+Tstruct['T_42']+Tstruct['T_43']+Tstruct['T_44'])/4
                Tstruct['T']=(Tstruct['TL1']+Tstruct['TL2']+Tstruct['TL3']+Tstruct['TL4'])/4

                # Tstruct['SN1']=((Tstruct['T_11']-Tstruct['TL1'])/(Tstruct['T_13']-Tstruct['TL1']))
                # Tstruct['SN2']=((Tstruct['T_21']-Tstruct['TL2'])/(Tstruct['T_23']-Tstruct['TL2']))
                # Tstruct['SN3']=((Tstruct['T_31']-Tstruct['TL3'])/(Tstruct['T_33']-Tstruct['TL3']))
                # Tstruct['SN4']=((Tstruct['T_41']-Tstruct['TL4'])/(Tstruct['T_43']-Tstruct['TL4']))
                # Tstruct['SN1']=Tstruct['T_11']-Tstruct['T_13']
                # Tstruct['SN2']=Tstruct['T_21']-Tstruct['T_23']
                # Tstruct['SN3']=Tstruct['T_31']-Tstruct['T_33']
                # Tstruct['SN4']=Tstruct['T_41']-Tstruct['T_43']
                # 
                Tstruct['SN1']=Tstruct['TL1']-Tstruct['TL2']
                Tstruct['SN2']=Tstruct['TL4']-Tstruct['TL3']
                Tstruct['FB']=0.5*(Tstruct['SN1']+Tstruct['SN2'])
                
                rcu['Tstruct']=Tstruct[::10]
                
                if args.plot_stats:
                    if args.verbose>2:
                        plt.figure(figsize=(12,8))
                        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
                        formatter = mdates.ConciseDateFormatter(locator)
                        ax1=plt.subplot(111)
                        ax1.xaxis.set_major_locator(locator)
                        ax1.xaxis.set_major_formatter(formatter)
                        dt=[ datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S") for x in Tstruct['dt'][::10]]
                        plt.scatter(dt,Tstruct['FB'][::10],label='front-back difference')
                        plt.legend()
                
                # control
                # Tstruct['SN1']=Tstruct['TL1']-Tstruct['TL4']
                # Tstruct['SN2']=Tstruct['TL2']-Tstruct['TL3']
                # Tstruct['FB']=0.5*(Tstruct['SN1']+Tstruct['SN2'])
                #---
                # print(Tstruct)
                # Tint1=InterpolateTimeSeries(Tstruct['dt'],Tstruct['SN1'],dtfmt='%Y-%m-%d %H:%M:%S')
                # Tint2=InterpolateTimeSeries(Tstruct['dt'],Tstruct['SN2'],dtfmt='%Y-%m-%d %H:%M:%S')
                # Tint3=InterpolateTimeSeries(Tstruct['dt'],Tstruct['SN3'],dtfmt='%Y-%m-%d %H:%M:%S')
                # Tint4=InterpolateTimeSeries(Tstruct['dt'],Tstruct['SN4'],dtfmt='%Y-%m-%d %H:%M:%S')
                TFB=InterpolateTimeSeries(Tstruct['dt'],Tstruct['FB'],dtfmt='%Y-%m-%d %H:%M:%S')
                dt=P.get_dt()
                # SN1=Tint1(dt)
                # SN2=Tint2(dt)
                # SN3=Tint3(dt)
                # SN4=Tint4(dt)
                Tfb=TFB(dt)

                X=np.vstack((Tfb,np.ones(len(Tfb)))).T
                model = smapi.RLM(P.dZD, X)
                m = model.fit()
                pred=m.predict()
                print(m.summary())

                if args.plot_stats:
                    if args.verbose>2:
                        plt.figure(figsize=(12,8))
                        plt.scatter(Tfb,P.dZD,label='delta Tfb')
                        plt.plot(Tfb,pred)
                        plt.legend()
                        plt.show()
                        plt.close()
            
    
    if args.plot_stats:
        plot_corrections_2panel(rcu, topPanel=['C','M','X'], bottomPanel=['Tstruct'])
    
            
    ofile=os.path.join(cfg['DATA']['data_dir'],'corrections_stats'+args.export_suff+'.'+cfg['DATA']['rohcont_unified_corrections_file_suffix']+'.pkl')
    # stats_file=os.path.join(cfg['DATA']['data_dir'],'corrections_stats.'+cfg['DATA']['rohcont_unified_corrections_file_suffix']+'.pkl')
    with open(ofile,'wb') as f:
        pickle.dump(stats,f)
        print(stats)
        print("Corrections stats. exported to file: {}".format(ofile))
    
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
    
    print("Calculating continuous corrections")
    print("model: {}".format(model))
    if model=="median":
        mCrossElev,mdZD=P.get_median()
    elif model=='linear':
        pred=P.get_linear_model_prediction()
        print("prediction for time: {}".format(pred['dxZD']['dt'][-1]))
        mCrossElev,mdZD=pred['dxZD']['pred'][-1],pred['dZD']['pred'][-1]
        mCrossElev,mdZD=pred['dxZD']['pred'][-1],pred['dZD']['pred'][-1]
    # if args.plot:
    #     plot_corrections(P)
    
    return mCrossElev,mdZD

def plot_corrections_2panel(D,topPanel=[],bottomPanel=[],xcol='datetime',corr='ZD'):
    '''
    D - dictionary containing continuousCorrections objects for plotting.
        If 'Tstruct' is given it is treated as pandas DataFrame which
        must contain 'dt' column with datetime objects. It is plotted in bottom plot
        
    topPanel - list of columns to plot in the top panel
    bottomPanel - list of columns to plot in the bottom panel
    '''
    
    plt.figure(figsize=(12,10))
    locator = mdates.AutoDateLocator(minticks=7, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax1=plt.subplot(211)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())    
    ax1.yaxis.set_minor_locator(AutoMinorLocator())    

    for c in topPanel:
        # for k,C in D.items():
        plt.plot(D[c].get_dt(),D[c].dZD*1000,label=c)
    plt.xlabel('time [UTC]')
    plt.ylabel('$\Delta$ ZD [mdeg]')
    plt.legend(loc='upper right')
    plt.grid()

    ax2=plt.subplot(212,sharex=ax1)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())    
    # for k,C in D.items():
    for c in bottomPanel:
        if c in D.keys():
            if c=='Tstruct':
                plt.plot(D[c][xcol],D[c]['T'], label='Mean Tstruct')
                plt.plot(D[c][xcol],D[c]['FB'], label='Mean Front Back diff.')
            else:
                plt.plot(D[c].get_dt(),D[c].dZD*1000,label=c)
            
    plt.xlabel('time [UTC]')
    plt.ylabel('T [degC]')
    plt.legend(loc='upper right')
    plt.grid()

    plt.show()
    plt.close()

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
    def __init__(self, in_file=None, **kwargs):
        self.in_file=in_file
        try:
            self.temp=np.loadtxt(in_file, dtype=str, ndmin=2)
        except ValueError:
            print("=============================================")
            print("WARNING")
            print("WARNING:detected file inconsistency in {}. Will try to load what makes sense.".format(in_file))
            print("LOADED STUFF")
            # try to load skipping bad lines
            with open(in_file,'r') as f:
                all=f.readlines()
                tmpdata=[]
                for l in all:
                    ls=l.split()
                    if len(ls)==3:
                        tmpdata.append(ls)
                    else:
                        print("detected bad line, skipping ({})".format(ls))
                self.temp=np.array(tmpdata)
                print(self.temp)
            print("=============================================")
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
    
    def load_csv(self,**kwargs):
        '''
        '''
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
    


class InterpolateTimeSeries():
    '''
    Interpolation object.
    The class allows to obtain value for any given date
    '''
    def __init__(self, dt,y,dtfmt='%Y-%m-%dT%H:%M:%S',**kwargs):
        assert len(dt)>1
        assert len(dt)==len(y)
        
        if isinstance(dt[0], str):
            self.dt=[ datetime.datetime.strptime(x, dtfmt) for x in dt ]
        elif isinstance(dt[0], datetime.datetime):
            self.dt=dt
        self.dt0=self.dt[0]
        self.x=self.toX(self.dt)
        
        self.y=y.to_numpy()
        self.val_inter=interpolate.interp1d(self.x,self.y,fill_value="extrapolate", kind='linear')
    
    def toX(self,dt : list):
        return [(x-self.dt0).total_seconds() for x in dt ]
    
    def last(self) -> tuple:
        '''
        return last known correction
        returns
        -------
            tuple(dZD,dxZD)
        '''
        return self.y[-1]
        
    def __call__(self,dt : list):    
        '''
        calculate continuous corrections used up to given datetime
        
        parameters
        ----------
            dt - datetime object for which interpolated values are to be calculated
        
        returns
        -------
            1-d array like
        '''
        return self.val_inter(self.toX(dt))


        

#import h5py
import h5pickle as h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import multiprocessing
import sys
import time
import sinetools.SineTools as st
import yappi
import warnings

import os
from adccaldata import Met4FOFADCCall
from scipy import stats ## for Student T Coverragefactor
from scipy.optimize import curve_fit #for fiting of Groupdelay
from scipy import interpolate #for 1D amplitude estimation



uncerval = np.dtype([("value", np.float), ("uncertainty", np.float)])


def getplotableunitstring(unitstr, Latex=False):
    if not Latex:
        convDict = {
            "\\degreecelsius": "°C",
            "\\degree": "°",
            "\\micro\\tesla": "µT",
            "\\radian\\second\\tothe{-1}": "rad/s",
            "\\metre\\second\\tothe{-2}": "m/s^2",
            "\\volt": "v",
            "\\hertz": "Hz",
        }
    else:
        convDict = {
            "\\degreecelsius": "$^\circ C$",
            "\\degree": "$^\circ$",
            "\\micro\\tesla": "$\micro T$",
            "\\radian\\second\\tothe{-1}": "$\\frac{m}{s}$",
            "\\metre\\second\\tothe{-2}": "$\\frac{m}{s^2}",
            "\\volt": "v",
            "\\hertz": "Hz",
        }
    try:
        result = convDict[unitstr]
    except KeyError:
        result = unitstr
    return result


# code from https://stackoverflow.com/questions/23681948/get-index-of-closest-value-with-binary-search  answered Dec 6 '14 at 23:51 Yaguang
def binarySearch(data, val):
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break

        # check if data[mid] is closer to val than data[best_ind]
        dtype=data.dtype
        if data.dtype==np.uint64:
            absmidint=abs(data[mid].astype(np.int64) - val.astype(np.int64))
            absbestind = abs(data[best_ind].astype(np.int64) - val.astype(np.int64))
        else:
            absmidint=abs(data[mid] - val)
            absbestind = abs(data[best_ind]- val)
        if absmidint<absbestind:
            best_ind = mid

    return best_ind


class hdfmet4fofdatafile:
    def __init__(self,hdffile):
        self.hdffile=hdffile
        self.senorsnames=list(self.hdffile['RAWDATA'].keys())
        self.sensordatasets={}
        for name in self.senorsnames:
            datasets=list(self.hdffile['RAWDATA/'+name].keys())
            keystocheckandremove=['Absolutetime', 'Absolutetime_uncertainty','Sample_number']
            for key in keystocheckandremove:
                try:
                    datasets.remove(key)
                except ValueError:
                    raise RuntimeWarning(str(name)+" doese not contain "+str(key)+" dataset is maybe corrupted!")
            self.sensordatasets[name]=datasets
        print("INIT DONE")
        print("RAW DataGroups are "+str(self.senorsnames))
        print("RAW Datasets are " + str(self.sensordatasets))

    def calcblockwiesestd(self,dataset,blocksize=100):
        start = time.time()
        blockcount=int(np.floor(dataset.size / blocksize))
        std=np.zeros(blockcount)
        split=np.split(dataset[:blocksize*blockcount], blockcount, axis=0)
        std=np.std(split,axis=1)
        end = time.time()
        print("bwstd for dataset "+str(dataset)+"took "+str(end - start)+ " secs")
        return std

    def detectnomovment(self,sensorname,quantitiy,treshold=0.05,blocksinrow=5,blocksize=100):
        tmpData = np.squeeze(self.hdffile['RAWDATA/'+sensorname + '/' + quantitiy])
        tmpTime = np.squeeze(self.hdffile['RAWDATA/'+sensorname + '/' + 'Absolutetime'])# fetch data from hdffile
        mag = np.linalg.norm(tmpData, axis=0)
        std = self.calcblockwiesestd(mag, blocksize=blocksize)
        wasvalide=0
        nomovementtidx=[]
        nomovementtimes = []
        for i in range(std.size):
            if std[i]<treshold:
                wasvalide=wasvalide+1
            else:
                if wasvalide>blocksinrow:
                    startidx=(i-wasvalide)*blocksize
                    stopidx = (i) * blocksize
                    if stopidx-startidx<0:
                        print("index error")
                    if tmpTime[stopidx]-tmpTime[startidx]<0:
                        print("timing error")
                    nomovementtidx.append([startidx,stopidx])
                    nomovementtimes.append([tmpTime[startidx], tmpTime[stopidx]])
                wasvalide = 0
        nomovementidx=np.array(nomovementtidx)
        return nomovementidx,np.array(nomovementtimes)

    def detectmovment(self,sensorname,quantitiy,treshold=0.5,blocksinrow=5,blocksize=100,plot=False):
        tmpData = np.squeeze(self.hdffile['RAWDATA/'+sensorname + '/' + quantitiy])
        tmpTime = np.squeeze(self.hdffile['RAWDATA/'+sensorname + '/' + 'Absolutetime'])# fetch data from hdffile
        mag = np.linalg.norm(tmpData, axis=0)
        std = self.calcblockwiesestd(mag, blocksize=blocksize)
        wasvalide=0
        movementtidx=[]
        movementtimes = []
        for i in range(std.size):
            if std[i]>treshold:
                wasvalide=wasvalide+1
            else:
                if wasvalide>blocksinrow:
                    startidx=(i-wasvalide)*blocksize
                    stopidx = (i) * blocksize
                    movementtidx.append([startidx,stopidx])
                    movementtimes.append([tmpTime[startidx], tmpTime[stopidx]])
                wasvalide = 0
        movementidx=np.array(movementtidx)
        if plot:
            fig,ax=plt.subplots()
            reltime=(tmpTime-tmpTime[0])/1e9
            blocktime=reltime[0::blocksize][:-1]#cut one block out--> garden fence problem
            ax.plot(blocktime,std,label='Data')
            for i in np.arange(len(movementtimes)):
                relmovementimes=(movementtimes[i]-tmpTime[0])/1e9
                ax.plot(relmovementimes,np.array([treshold,treshold]),label=str(i))
            fig.show()
        return movementidx,np.array(movementtimes)

    def getnearestidxs(self,sensorname,time):
        absolutimegroupname='RAWDATA/'+sensorname + '/' + 'Absolutetime'
        absolutetimes= np.squeeze(self.hdffile[absolutimegroupname])
        idxs=np.copy(time)
        with np.nditer(idxs, op_flags=['readwrite']) as it:
            for x in it:
                x[...]=binarySearch(absolutetimes ,x)
        return idxs

class transferfunktion:
    def __init__(self,tfgroup):
        self.group=tfgroup

    def getNearestTF(self,channel,freq):
        Freqs=self.group['Frequency'][:]
        testFreqIDX=np.argmin(abs(Freqs-freq))
        if Freqs[testFreqIDX]-freq==0:#ok we hit an calibrated point no need to interpolate
            return {'frequency':Freqs[testFreqIDX],
                    'magnitude':self.group['magnitude'][channel][testFreqIDX],
                    'phase':self.group['phase'][channel][testFreqIDX],
                    'N':self.group['N'][channel][testFreqIDX]}
        else:
            #interpolate
            A=self.getInterPolatedAmplitude(channel,freq)
            P=self.getInterPolatedPhase(channel,freq)
            return {'Frequency':freq,
                    'magnitude':A,
                    'phase':P,
                    'N':np.NaN}

    def __getitem__(self, key):
        if len(key)==4:
            return self.TransferFunctions[key]
        if len(key)==2:
            return self.getNearestTF(key[0],key[1])
        else:
            raise ValueError("Invalide Key:  > "+str(key)+" <Use either [Channel] eg ['ADC1] or [Channel,Frequency] eg ['ADC1',1000]  as key ")

    def getGroupDelay(self,Channel):
        freqs=self.TransferFunctions[Channel]['Frequencys']
        phases=self.TransferFunctions[Channel]['Phase']
        phaseUncer=self.TransferFunctions[Channel]['PhaseUncer']
        popt, pcov = curve_fit(PhaseFunc, freqs, phases,sigma=phaseUncer,absolute_sigma=True)
        return[popt,pcov]

    def getInterPolatedAmplitude(self,channel,freq):
        Freqs=self.group['Frequency'][:]
        Ampls=self.group['magnitude'][channel,:]
        testFreqIDX=np.argmin(abs(Freqs-freq))
        DeltaInterpolIDX=0
        if freq-Freqs[testFreqIDX]<0:
            DeltaInterpolIDX=-1
        if freq-Freqs[testFreqIDX]>0:
            DeltaInterpolIDX=1
        if testFreqIDX+DeltaInterpolIDX<0:
            assert RuntimeWarning(str(freq)+" is to SMALL->Extrapolation is not recomended! minimal Frequency is "+str(Freqs[0]))
            return Ampls[0]
        if testFreqIDX+DeltaInterpolIDX>=Freqs.size:
            raise ValueError(str(freq)+" is to BIG->Extrapolation not supported! maximal Frequency is "+str(Freqs[-1]))
        if DeltaInterpolIDX==0:
            return Ampls[testFreqIDX]
        elif DeltaInterpolIDX ==-1:
            IDX=[testFreqIDX-1,testFreqIDX]
        elif DeltaInterpolIDX==1:
            IDX=[testFreqIDX,testFreqIDX+1]
        x=Freqs[IDX]
        A=Ampls[IDX]['value']
        AErr=Ampls[IDX]["uncertainty"]
        fA = interpolate.interp1d(x, A)
        fAErr = interpolate.interp1d(x, AErr)
        print('Interpolateded transferfunction for Channel '+str(channel)+'at Freq '+str(freq))  # will not print anything
        result=np.empty([1], dtype=uncerval)
        result['value']=fA(freq)
        result['uncertainty']=fAErr(freq)
        return result

    def getInterPolatedPhase(self,channel,freq):
        Freqs=self.group['Frequency'][:]
        Phases=self.group['phase'][channel,:]
        testFreqIDX=np.argmin(abs(Freqs-freq))
        DeltaInterpolIDX=0
        if freq-Freqs[testFreqIDX]<0:
            DeltaInterpolIDX=-1
        if freq-Freqs[testFreqIDX]>0:
            DeltaInterpolIDX=1
        if testFreqIDX+DeltaInterpolIDX<0:
            assert RuntimeWarning(str(freq)+" is to SMALL->Extrapolation is not recomended! minimal Frequency is "+str(Freqs[0]))
            return Phases[0]
        if testFreqIDX+DeltaInterpolIDX>=Freqs.size:
            raise ValueError("Extrapolation not supported! maximal Frequency is"+Freqs[-1])
        if DeltaInterpolIDX==0:
            return Phases[testFreqIDX]
        elif DeltaInterpolIDX ==-1:
            IDX=[testFreqIDX-1,testFreqIDX]
        elif DeltaInterpolIDX==1:
            IDX=[testFreqIDX,testFreqIDX+1]
        x=Freqs[IDX]
        P=Phases[IDX]['value']
        PErr=Phases[IDX]['uncertainty']
        fP = interpolate.interp1d(x, P)
        fPErr = interpolate.interp1d(x, PErr)
        print('Interpolateded transferfunction for Channel '+str(channel)+'at Freq '+str(freq))  # will not print anything
        result = np.empty([1], dtype=uncerval)
        result['value'] = fP(freq)
        result['uncertainty'] = fPErr(freq)
        return result



class experiment:
    def __init__(self,hdfmet4fofdatafile,times):
        self.met4fofdatafile=hdfmet4fofdatafile
        self.timepoints=times
        self.idxs={}
        self.Data={}
        self.flags={}
        for name in self.met4fofdatafile.senorsnames:
            start = time.time()
            self.idxs[name]=self.met4fofdatafile.getnearestidxs(name,self.timepoints)
            end = time.time()
            self.Data[name] = {}
            for dataset in self.met4fofdatafile.sensordatasets[name]:
                self.Data[name][dataset] = {}
        print(self.idxs)

    def plotall(self,absolutetime=False):
        cols=len(self.met4fofdatafile.sensordatasets)#one colum for every sensor
        datasetspersensor=[]
        for sensors in self.met4fofdatafile.sensordatasets:
            datasetspersensor.append(len(self.met4fofdatafile.sensordatasets[sensors]))
        rows=np.max(datasetspersensor)
        fig,axs=plt.subplots(cols,rows, sharex='all')
        icol=0
        for sensor in self.met4fofdatafile.sensordatasets:
            irow=0
            idxs=self.idxs[sensor]
            axs[icol,0].annotate(sensor.replace('_',' '), xy=(0, 0.5), xytext=(-axs[icol,0].yaxis.labelpad - 5, 0),
                        xycoords=axs[icol,0].yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center',rotation=90)

            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                dsetattrs=self.met4fofdatafile.hdffile['RAWDATA/'+sensor+ '/' + dataset ].attrs
                time=self.met4fofdatafile.hdffile['RAWDATA/'+sensor + '/' + 'Absolutetime'][0,idxs[0]:idxs[1]]
                if not absolutetime:
                    time=time.astype('int64') - self.timepoints[0].astype('int64')
                time=time/1e9
                #print(dsetattrs.keys())
                axs[icol, irow].set_ylabel(getplotableunitstring(dsetattrs['Unit']))
                if not absolutetime:
                    axs[icol, irow].set_xlabel("Relative time in s")
                else :
                    axs[icol, irow].set_xlabel("Unixtime in s")
                axs[icol, irow].set_title(dataset.replace('_',' '))
                for i in np.arange(self.met4fofdatafile.hdffile['RAWDATA/'+sensor + '/' + dataset ].shape[0]):
                    label=dsetattrs['Physical_quantity'][i]
                    data=self.met4fofdatafile.hdffile['RAWDATA/'+sensor + '/' + dataset ][i,idxs[0]:idxs[1]]
                    axs[icol,irow].plot(time,data,label=label)
                    axs[icol,irow].legend()
                irow=irow+1
            icol=icol+1
        fig.show()

class sineexcitation(experiment):
    def __init__(self,hdfmet4fofdatafile,times):
        super().__init__(hdfmet4fofdatafile,times)
    def dofft(self):
        for sensor in self.met4fofdatafile.sensordatasets:
            idxs = self.idxs[sensor]
            points=idxs[1]-idxs[0]
            time = self.met4fofdatafile.hdffile['RAWDATA/' + sensor + '/' + 'Absolutetime'][0, idxs[0]:idxs[1]]
            reltime = time.astype('int64') - self.timepoints[0].astype('int64')
            self.Data[sensor]['Mean Delta T']=np.mean(np.diff(reltime/1e9))
            self.Data[sensor]['RFFT Frequencys']=np.fft.rfftfreq(points,self.Data[sensor]['Mean Delta T'])
            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                data=self.met4fofdatafile.hdffile['RAWDATA/'+sensor + '/' + dataset ][:,idxs[0]:idxs[1]]
                self.Data[sensor][dataset]['RFFT'] =np.fft.rfft(data,axis=1)
                self.Data[sensor][dataset]['FFT_max_freq']=self.Data[sensor]['RFFT Frequencys'][np.argmax(abs(np.sum(self.Data[sensor][dataset]['RFFT'][:,1:],axis=0)))+1]
                #print(self.Data[sensor][dataset]['FFT_max_freq'])
        self.flags['FFT Calculated'] = True

    def do3paramsinefits(self,freqs,periods=10):
        if not self.flags['FFT Calculated']:
            self.dofft()
        for sensor in self.met4fofdatafile.sensordatasets:
            idxs = self.idxs[sensor]
            points = idxs[1] - idxs[0]
            time = self.met4fofdatafile.hdffile['RAWDATA/' + sensor + '/' + 'Absolutetime'][0, idxs[0]:idxs[1]]
            reltime = time.astype('int64') - self.timepoints[0].astype('int64')
            reltime=reltime/1e9
            excitationfreqs=freqs
            uniquexfreqs = np.sort(np.unique(excitationfreqs))
            idxs = self.idxs[sensor]
            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                try:
                    fftmaxfreq=self.Data[sensor][dataset]['FFT_max_freq']
                except NameError:
                    self.dofft()
                freqidx=binarySearch(uniquexfreqs, fftmaxfreq)
                f0=uniquexfreqs[freqidx]
                self.Data[sensor][dataset]['Sin_Fit_freq'] =f0
                datasetrows=self.met4fofdatafile.hdffile['RAWDATA/' + sensor + '/' +dataset].shape[0]
                #calc first row and create output array[:,idxs[0]:idxs[1]]
                sineparams=st.seq_threeparsinefit(self.met4fofdatafile.hdffile['RAWDATA/' + sensor + '/' +dataset][0, idxs[0]:idxs[1]], reltime , f0,periods=10)
                self.Data[sensor][dataset]['SinPOpt']=np.zeros([datasetrows,4])
                self.Data[sensor][dataset]['SinPCov'] = np.zeros([datasetrows,4, 4])
                self.Data[sensor][dataset]['SinParams'] = np.zeros([datasetrows, sineparams.shape[0], 3])
                self.Data[sensor][dataset]['SinParams'][0]=sineparams
                for i in np.arange(1,datasetrows):
                    sineparams = st.seq_threeparsinefit(
                        self.met4fofdatafile.hdffile['RAWDATA/' + sensor + '/' + dataset][i, idxs[0]:idxs[1]], reltime, f0,periods=10)
                    self.Data[sensor][dataset]['SinParams'][i] = sineparams
                for i in np.arange(datasetrows):
                    sineparams = self.Data[sensor][dataset]['SinParams'][i]
                    Complex = sineparams[:, 1] + 1j * sineparams[:, 0]
                    DC = sineparams[:, 2]
                    Freq = np.ones(sineparams.shape[0])*f0
                    self.Data[sensor][dataset]['SinPOpt'][i,:] = [
                        np.mean(abs(Complex)),
                        np.mean(DC),
                        np.mean(Freq),
                        np.mean(np.unwrap(np.angle(Complex))),
                    ]
                    CoVarData = np.stack(
                        (abs(Complex), DC, Freq, np.unwrap(np.angle(Complex))), axis=0
                    )
                    self.Data[sensor][dataset]['SinPCov'][i,:] = np.cov(
                        CoVarData, bias=True
                    )  # bias=True Nomation With N like np.std
        self.flags['Sine fit calculated']=True
        return

    def calculatetanloguephaserf1d(self,refdatagroupname,refdataidx,analogrefchannelname,analogrefchannelidx):
        for sensor in self.met4fofdatafile.sensordatasets:
            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                datasetrows = self.met4fofdatafile.hdffile['RAWDATA/' + sensor + '/' + dataset].shape[0]
                self.Data[sensor][dataset]['TF']={}
                self.Data[sensor][dataset]['TF']['magnitude']=np.empty(datasetrows,dtype=uncerval)
                self.Data[sensor][dataset]['TF']['phase'] = np.empty(datasetrows,dtype=uncerval)
                self.Data[sensor][dataset]['TF']['ExAmp'] = np.empty(datasetrows,dtype=uncerval)
                for i in np.arange(0, datasetrows):
                    fitfreq = self.Data[sensor][dataset]['Sin_Fit_freq']
                    reffreq = self.met4fofdatafile.hdffile[refdatagroupname]['Frequency'][0][refdataidx]
                    ADCTF=transferfunktion(self.met4fofdatafile.hdffile[analogrefchannelname].attrs['Transferfunction'])
                    if fitfreq != reffreq:
                        warinigstr="Frequency mismatach in Sesnor"+sensor+' '+dataset+" fit["+str(i)+"]= "+str(fitfreq)+" ref["+str(refdataidx)+"]= "+str(reffreq)+" Transferfunction will be invaladie !!"
                        warnings.warn(warinigstr,RuntimeWarning)

                    else:
                        self.Data[sensor][dataset]['TF']['ExAmp'][i] = examp = self.met4fofdatafile.hdffile[refdatagroupname]['Excitation amplitude'][i][refdataidx]
                        phase = self.met4fofdatafile.hdffile[refdatagroupname]['Phasee'][i][analogrefchannelidx]#todo fix spell rerror
                        self.Data[sensor][dataset]['TF']['magnitude'][i]['value']= self.Data[sensor][dataset]['SinPOpt'][i][0]/examp['value']
                        self.Data[sensor][dataset]['TF']['magnitude'][i]['uncertainty']=np.NaN

def add1dsinereferencedatatohdffile(csvfilename,hdffile,axis=2):
    refcsv= pd.read_csv(csvfilename, delimiter=";",comment='#')
    hdffile=hdffile
    isaccelerationreference1d=False
    with open(csvfilename, "r") as file:
        first_line = file.readline()
        second_line = file.readline()
        third_line= file.readline()
        if r'loop;frequency;ex_amp;ex_amp_std;phase;phase_std' in first_line and r'#Number;Hz;m/s^2;m/s^2;deg;deg' in third_line:
            isaccelerationreference1d = True
            print("1D Accelerationrefference fiele given creating hdf5 data set")
        else:
            if not r'loop;frequency;ex_amp;ex_amp_std;phase;phase_std' in first_line:
                raise RuntimeError("Looking for >>>loop;frequency;ex_amp;ex_amp_std;phase;phase_std<<< in csvfile first row got"+first_line)
            if not r'#Number;Hz;m/s^2;m/s^2;deg;deg' in third_line:
                raise RuntimeError("Looking for >>>loop;frequency;ex_amp;ex_amp_std;phase;phase_std<<< in csvfile first row got"+third_line)
    if isaccelerationreference1d:
        Datasets = {}
        REFDATA = hdffile.create_group("REFENCEDATA")
        group = REFDATA.create_group("Acceleration_refference")
        group.attrs['Refference_name'] = "PTB HF acceleration standard"
        group.attrs['Sensor_name'] = group.attrs['Refference_name']
        group.attrs['Refference_type'] = "1D acceleration"
        Datasets['Frequency'] = group.create_dataset('Frequency', ([refcsv.shape[0]]),
                                                    dtype='float64')
        Datasets['Frequency'].make_scale("Frequency")
        Datasets['Frequency'].attrs['Unit'] = "/hertz"
        Datasets['Frequency'].attrs['Physical_quantity'] = "Excitation frequency"
        Datasets['Frequency']=refcsv['frequency'].to_numpy()
        Datasets['Repetition count'] = group.create_dataset('repetition count', ([refcsv.shape[0]]),
                                                    dtype='int32')
        Datasets['Repetition count'].attrs['Unit'] = "/one"
        Datasets['Repetition count'].attrs['Physical_quantity'] = "Repetition count"
        Datasets['Repetition count'] = refcsv['loop'].to_numpy()
        Datasets['Repetition count'].dims[0].label = 'Frequency'
        Datasets['Repetition count'].dims[0].attach_scale(Datasets['Frequency'])
        Datasets['Excitation amplitude'] = group.create_dataset('Excitation amplitude', ([3, refcsv.shape[0]]),
                                                    dtype=uncerval)
        Datasets['Excitation amplitude'].attrs['Unit'] = "\\metre\\second\\tothe{-2}"
        Datasets['Excitation amplitude'].attrs['Physical_quantity'] = ["X Excitation amplitude",
                                                                       "Y Excitation amplitude",
                                                                       "Z Excitation amplitude"]
        Datasets['Excitation amplitude'].attrs['UNCERTAINTY_TYPE'] = "95% coverage gausian"
        Datasets['Excitation amplitude'] = np.empty([3, refcsv.shape[0]])
        Datasets['Excitation amplitude'][axis, :,"value"] = refcsv['ex_amp']
        Datasets['Excitation amplitude'][axis, :,"uncertainty"] = refcsv['ex_amp_std']
        Datasets['Excitation amplitude'].dims[0].label = 'Frequency'
        Datasets['Excitation amplitude'].dims[0].attach_scale(Datasets['Frequency'])

        Datasets['Phase']=group.create_dataset('Phase', ([3, refcsv.shape[0]]),
                                                    dtype=uncerval)
        Datasets['Phase'].attrs['Unit'] = "\\degree"
        Datasets['Phase'].attrs['Physical_quantity'] = ["X Phase",
                                                        "Y Phase",
                                                        "Z Phase"]
        Datasets['Phase'].attrs['UNCERTAINTY_TYPE'] = "95% coverage gausian"
        Datasets['Phase'] = np.empty([3, refcsv.shape[0]])
        Datasets['Phase'][axis, :,"value"] = refcsv['phase']
        Datasets['Phase'][axis, :,"uncertainty"] = refcsv['phase_std']
        Datasets['Phase'].dims[0].label = 'Frequency'
        Datasets['Phase'].dims[0].attach_scale(Datasets['Frequency'])
        hdffile.flush()


def addadctransferfunctiontodset(topgroup,tragetsensor,jsonfilelist):
    ADCCal = Met4FOFADCCall(Filenames=jsonfilelist)
    TFs = {}
    for channel in ADCCal.fitResults.keys():
        TFs[channel] = ADCCal.GetTransferFunction(channel)
    channelcount = len(TFs.keys())
    freqpoints = np.empty(channelcount)
    i = 0
    freqsmatch = True
    for channel in TFs:
        freqpoints[i] = len(TFs[channel]['Frequencys'])
        if i > 0 and freqsmatch:
            result = (freqpoints[0] == freqpoints[i]).all()
            if result == False:
                freqsmatch = False
                raise ValueError("All ADC Channels need to have the same frequencys")
        i = i + 1
    channeloder = ['ADC1', 'ADC2', 'ADC3']
    Datasets = {}
    group=topgroup.create_group("Transferfunction")
    tragetsensor.attrs['Transferfunction']=group
    Datasets['frequency'] = group.create_dataset('Frequency', ([freqpoints[0]]), dtype='float64')
    Datasets['frequency'].make_scale("Frequency")
    Datasets['frequency'].attrs['Unit'] = "/hertz"
    Datasets['frequency'].attrs['Physical_quantity'] = "Excitation frequency"
    Datasets['frequency'][0:] = TFs[channeloder[0]]['Frequencys']
    Datasets['magnitude'] = group.create_dataset('magnitude', ([channelcount, freqpoints[0]]),
                                                 dtype=uncerval)
    Datasets['magnitude'].attrs['Unit'] = "\\one"
    Datasets['magnitude'].attrs['Physical_quantity'] = ['Magnitude response Voltage Ch 1',
                                                        'Magnitude response Voltage Ch 2',
                                                        'Magnitude response Voltage Ch 3']
    Datasets['magnitude'].attrs['UNCERTAINTY_TYPE'] = "95% coverage gausian"
    i = 0
    for channel in channeloder:
        Datasets['magnitude'][i, :, "value"] = TFs[channel]['AmplitudeCoefficent']
        Datasets['magnitude'][i, :, "uncertainty"] = TFs[channel]['AmplitudeCoefficentUncer']
        i = i + 1
    Datasets['magnitude'].dims[0].label = 'Frequency'
    Datasets['magnitude'].dims[0].attach_scale(Datasets['frequency'])

    Datasets['phase'] = group.create_dataset('phase', ([channelcount, freqpoints[0]]),
                                             dtype=uncerval)
    Datasets['phase'].attrs['Unit'] = "\\radian"
    Datasets['phase'].attrs['Physical_quantity'] = ['Phase response Voltage Ch 1',
                                                    'Phase response Voltage Ch 2',
                                                    'Phase response  Voltage Ch 3']
    Datasets['phase'].attrs['UNCERTAINTY_TYPE'] = "95% coverage gausian"
    i = 0
    for channel in channeloder:
        Datasets['phase'][i, :, "value"] = TFs[channel]['Phase']
        Datasets['phase'][i, :, "uncertainty"] = TFs[channel]['PhaseUncer']
        i = i + 1
    Datasets['phase'].dims[0].label = 'Frequency'
    Datasets['phase'].dims[0].attach_scale(Datasets['frequency'])

    Datasets['N'] = group.create_dataset('N', ([channelcount, freqpoints[0]]),
                                         dtype=np.int32)
    Datasets['N'].attrs['Unit'] = "\\one"
    Datasets['N'].attrs['Physical_quantity'] = ['Datapoints Voltage Ch 1',
                                                'Datapoints Voltage Ch 2',
                                                'Datapoints Voltage Ch 3']
    i = 0
    for channel in channeloder:
        Datasets['N'][i, :] = TFs[channel]['N']
        i = i + 1
    Datasets['N'].dims[0].label = 'Frequency'
    Datasets['N'].dims[0].attach_scale(Datasets['frequency'])



def processdata(i):
    #print(i)
    sys.stdout.flush()
    times=mpdata['movementtimes'][i]
    times[0] += 1000000000
    times[1] -= 1000000000
    if times[1].astype(np.int64)-times[0].astype(np.int64)<0:
        raise ValueError("time after cutting is <0")
    experiment = sineexcitation(mpdata['hdfinstance'], times)
    sys.stdout.flush()
    #print(experiment)
    sys.stdout.flush()
    start = time.time()
    experiment.dofft()
    end = time.time()
    #print("FFT Time "+str(end - start))
    start = time.time()
    experiment.do3paramsinefits(mpdata['uniquexfreqs'],periods=10)
    end = time.time()
    #print("Sin Fit Time "+str(end - start))
    sys.stdout.flush()
    experiment.calculatetanloguephaserf1d('REFENCEDATA/Acceleration_refference', i, 'RAWDATA/0x1fe40a00_STM32_Internal_ADC', 2)
    return experiment



if __name__ == "__main__":
    yappi.start()
    start = time.time()

    hdffilename = r"/media/benedikt/nvme/data/201118_BMA280_amplitude_frequency/20201118153703_BMA_280_0x1fe40000_00000.hdf5"
    revcsv = r"/media/benedikt/nvme/data/201118_BMA280_amplitude_frequency/20201118153703_BMA_280_0x1fe40000_00000_Ref_TF.csv"
    datafile = h5py.File(hdffilename, 'r+',driver='core')
    #add1dsinereferencedatatohdffile(revcsv, datafile)
    test=hdfmet4fofdatafile(datafile)
    #adc_tf_goup=datafile.create_group("REFENCEDATA/0x1fe40a00_STM32_Internal_ADC")
    #addadctransferfunctiontodset(adc_tf_goup,datafile["RAWDATA/0x1fe40a00_STM32_Internal_ADC"], [r"/media/benedikt/nvme/data/201118_BMA280_amplitude_frequency/200318_1FE4_ADC123_19V5_1HZ_1MHZ.json"])
    datafile.flush()
    #nomovementidx,nomovementtimes=test.detectnomovment('0x1fe40000_MPU_9250', 'Acceleration')
    movementidx,movementtimes=test.detectmovment('0x1fe40000_BMA_280', 'Acceleration',treshold=0.08,blocksinrow=1000,blocksize=50,plot=True)
    manager = multiprocessing.Manager()
    mpdata=manager.dict()
    mpdata['hdfinstance']=test
    mpdata['movementtimes']=movementtimes
    mpdata['uniquexfreqs'] = np.unique(test.hdffile['REFENCEDATA/Acceleration_refference/Frequency'][0, :], axis=0)
    i=np.arange(movementtimes.shape[0])
    i=np.arange(15)
    with multiprocessing.Pool(15) as p:
        results=p.map(processdata,i)
    end = time.time()
    print(end - start)

import os
import random
import numpy as np
import copy
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator,LogLocator
from brokenaxes import brokenaxes
import sinetools.SineTools as st
from multiprocessing import Pool
from contextlib import closing
import multiprocessing as mp
import tqdm
from tqdm.contrib.concurrent import process_map
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.signal import correlate
from scipy.signal import correlation_lags
from scipy.ndimage import gaussian_filter
import scipy as sp
import uncertainties
from tools.figPickel import saveImagePickle
#from mpi4py import MPI #multiprocessing read
import h5py as h5py
#import h5pickle as h5py
import functools
#import allantools
import sineTools2 as st2
#____________________ GLobal config begin_____________
# jitterGensForSimulations=manager.list()
jitterGensForSimulations = []
jitterSimuLengthInS=1.0
localFreqqCorr=True
askforFigPickelSave=False

lineSyles=[
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

tubscolors=[(0/255,112/255,155/255),(250/255,110/255,0/255), (109/255,131/255,0/255), (81/255,18/255,70/255),(102/255,180/255,211/255),(255/255,200/255,41/255),(172/255,193/255,58/255),(138/255,48/255,127/255)]
plt.rcParams['axes.prop_cycle'] = colorCycler=plt.cycler(color=tubscolors) #TUBS Blue,Orange,Green,Violet,Light Blue,Light Orange,Lieght green,Light Violet
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\boldmath'
LANG='EN'
if LANG=='DE':
    import locale
    trueFalseAnAus = {True: 'An', False: 'Aus'}
    locale.setlocale(locale.LC_NUMERIC,"de_DE.utf8")
    locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    plt.rcParams['text.latex.preamble'] = r'\usepackage{icomma}\usepackage{amsmath}\boldmath' # remove nasty Space behind comma in de_DE.utf8 locale https://stackoverflow.com/questions/50657326/matplotlib-locale-de-de-latex-space-btw-decimal-separator-and-number
    plt.rcParams['axes.formatter.use_locale'] = True
else:
    import locale
    trueFalseAnAus = {True: 'On', False: 'Off'}
    plt.rcParams['text.latex.preamble'] = r'\usepackage{icomma}\usepackage{amsmath}\boldmath' # remove nasty Space behind comma in de_DE.utf8 locale https://stackoverflow.com/questions/50657326/matplotlib-locale-de-de-latex-space-btw-decimal-separator-and-number
    locale.setlocale(locale.LC_NUMERIC,"en_US.utf8")
    locale.setlocale(locale.LC_ALL,"en_US.utf8")
    plt.rcParams['axes.formatter.use_locale'] = True
#plt.rcParams['mathtext.fontset'] = 'custom'
#plt.rcParams['mathtext.rm'] = 'NexusProSans'
#plt.rcParams['mathtext.it'] = 'NexusProSans:italic'
#plt.rcParams['mathtext.bf'] = 'NexusProSans:bold'
#plt.rcParams['mathtext.tt'] = 'NexusProSans:monospace'
plt.rcParams['svg.fonttype'] = 'none'  # This stores text as text in SVG files, not paths
plt.rc('text', usetex=True)
plt.rc("figure", figsize=[16,9])  # fontsize of the figure title
plt.rc("figure", dpi=300)
PLTSCALFACTOR = 2
SMALL_SIZE = 9 * PLTSCALFACTOR
MEDIUM_SIZE = 12 * PLTSCALFACTOR
BIGGER_SIZE = 15 * PLTSCALFACTOR
plt.rc("font", weight='bold') # controls default text sizes
plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
figSaveCounter = 33
SAVEFOLDER = './imagesV7'
SHOW=False


def gaus(x,a,sigma):
    x0=0
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def logGisticPICaped(x,k,x0,L):
    return L/(1+np.exp(-k*(x-x0)))

def align_yaxis(ax1, v1, ax2, v2):
    #adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1
    #https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin/10482477#10482477

    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)



class realWordJitterGen:
    def __rpr__(self):
        return str(self.title)+' fs= '+str(self.fs)+' Hz'
    def __init__(self,HDfFile,sensorName,title,nominalfreq=0,offset=[40000,10000],pollingFreq=None):
        try:
            self.floatType=np.float64
        except AttributeError as e:
            print ("Your system moast likely Windows does not support float128 with numpy switching back to float64")
            print(e)
            self.floatType=np.float64
        datafile=HDfFile
        self.title=title
        self.Dataset=datafile['RAWDATA/'+sensorName+'/Absolutetime'][0]
        self.dataPoints=datafile['RAWDATA/'+sensorName].attrs['Data_point_number']# use only valide points
        self.AbsoluteTime = self.Dataset[ 0 + offset[0]:self.dataPoints - offset[1]] - self.Dataset[ offset[0]]
        self.timeData=((self.AbsoluteTime-self.AbsoluteTime[0])/1e9).astype(self.floatType)
        #self.timeData=(self.Dataset[0,0+offset[0]:self.dataPoints-offset[1]]-self.Dataset[0,offset[0]])/1e9# substract first point to avoid precisionlos with f64 the divide by 1e9 to ahve seconds
        self.relSampleNumber = (datafile['RAWDATA/'+sensorName+'/Sample_number'][0, 0 + offset[0]:self.dataPoints-offset[1]] - datafile['RAWDATA/'+sensorName+'/Sample_number'][0, offset[0]])
        if nominalfreq==0:
            self.fs =  (self.relSampleNumber[-1].astype(self.floatType))/self.timeData[-1].astype(self.floatType)  # calculate smaple freq
        else:
            self.fs = nominalfreq
        self.fs_std=np.NaN
        #self.deltaT=self.length/(self.dataPoints-1)
        self.deltaT = 1.0/self.fs
        print("Sample frequency is "+str(self.fs)+' Hz')
        self.expectedTime=self.relSampleNumber.astype(self.floatType)*self.deltaT
        self.deviationFromNominal=self.timeData-self.expectedTime#calulate deviation from Expected Mean
        if pollingFreq!=None:
            self.PollDT=1.0/pollingFreq
            self.pollingTimes=np.arange(int(self.timeData[-1]/self.PollDT))*self.PollDT
            self.pollingTimeDiffFromNom=np.zeros_like(self.pollingTimes)
            lastDataIdx=0
            lastTime=0
            for i in range(self.pollingTimes.size-1):
                while self.timeData[lastDataIdx]-self.pollingTimes[i]<0:
                    lastDataIdx+=1
                self.pollingTimeDiffFromNom[i]=self.pollingTimes[i]-self.timeData[lastDataIdx-1]
            self.pollingTimeDiffFromNom[0]=0.0
            self.deviationFromNominal=self.pollingTimeDiffFromNom
            self.expectedTime=self.pollingTimes

            self.fs=pollingFreq

            """
            if pollingFreq<self.fs:
                raise RuntimeError("Polling Freq smaller than sensor Frequency this is not supportet yet!")
            else:
                self.pollingDT=1/pollingFreq
                self.deviationFromNominal=(self.timeData % self.pollingDT)
            """

        self.meandeviationFromNominal=np.mean(self.deviationFromNominal)
        self.std = np.std(self.deviationFromNominal)
        # INTERPOLATE MISSING DATA WITH NEAREST NIGBOUR
        self.interpolatedDeviationFromNominal=np.ones(self.relSampleNumber[-1])*self.meandeviationFromNominal

        diff=np.diff(self.relSampleNumber)
        jumpIDX=np.array(np.argwhere(diff>1))
        #jumpIDX=self.relSampleNumber[jumpIDX]
        jumpIDX = np.insert(jumpIDX, 0, 0)
        coppyOffset=0
        if jumpIDX.size>1:
            for i in range(jumpIDX.size-2):
                startIDX=jumpIDX[i]
                stopIDX=jumpIDX[i+1]
                tmp=np.copy(self.deviationFromNominal[startIDX:stopIDX])
                self.interpolatedDeviationFromNominal[(startIDX+coppyOffset):(stopIDX+coppyOffset)]=tmp
                coppyOffset+=diff[stopIDX]
                print(coppyOffset)
                del tmp
        else:
            self.interpolatedDeviationFromNominal=np.copy(self.deviationFromNominal)

        print("Test")

    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import random

    def plotDeviation(self, fig=None, axs=None, lengthInS=None, show=False, lw=PLTSCALFACTOR, correctLinFreqDrift=True,
                      plotInSamples=False, save=False, unit='ns', yLims=None, plotInSamplesAxis=False, alpha=1,
                      color=None, maxSegments=1e4):
        salefactorFromeUnit = {'ns': 1e9, r'\textmu s': 1e6, 'ms': 1e3, 's': 1}
        yScaleFactor = salefactorFromeUnit[unit]

        if fig is None and axs is None:
            fig, ax = plt.subplots()
            axs = [ax]
            if plotInSamples or plotInSamplesAxis:
                ax2 = ax.twinx()  # axis for deviation in samples
                axs = [ax, ax2]
        else:
            if isinstance(axs, list):
                axs = axs
            else:
                if isinstance(axs, plt.Axes):
                    # we have only one axis so we need to create a second one
                    ax2 = axs.twinx()
                    axOld = axs
                    axs = [axOld, ax2]
                else:
                    raise ValueError("No valid axis object")

        segment_length = int(lengthInS * self.fs) if lengthInS is not None else self.expectedTime.size
        total_possible_segments = int(np.ceil(self.expectedTime.size / segment_length))

        # Calculate colors for the maximum possible segments
        cmap = plt.get_cmap('rainbow')
        all_colors = [cmap(i / total_possible_segments) for i in range(total_possible_segments)]

        # Determine the number of segments to plot using equal spacing
        if total_possible_segments > maxSegments:
            step_size = total_possible_segments // maxSegments
            segment_indices = range(0, step_size * maxSegments,
                                    step_size)  # Equal spaced indices, skipping the last one
        else:
            segment_indices = range(total_possible_segments)  # Use all segments if fewer than maxSegments

        slopes = np.zeros(len(segment_indices))
        start_times = []

        for segment_idx, segment_num in enumerate(segment_indices):
            segment_start = segment_num * segment_length
            segment_end = min((segment_num + 1) * segment_length, self.expectedTime.size)
            timeDev = self.deviationFromNominal[segment_start:segment_end]
            times = self.expectedTime[segment_start:segment_end] - self.expectedTime[segment_start]
            start_times.append(self.expectedTime[segment_start])

            current_color = all_colors[segment_num]

            if correctLinFreqDrift:
                slope = (timeDev[-1] - timeDev[0]) / times[-1]
                slopes[segment_idx] = slope
                correctedTimeDev = timeDev - slope * times - timeDev[0]
                axs[0].plot(times, correctedTimeDev * yScaleFactor, label=self.title if segment_idx == 0 else "", lw=lw,
                            color=current_color, alpha=alpha)
                if plotInSamples:
                    axs[1].plot(times, correctedTimeDev / self.deltaT, label=self.title if segment_idx == 0 else "",
                                lw=lw, color=current_color, ls=':', alpha=alpha)
            else:
                axs[0].plot(self.expectedTime[segment_start:segment_end] - self.expectedTime[segment_start],
                            self.deviationFromNominal[segment_start:segment_end] * yScaleFactor,
                            label=self.title if segment_idx == 0 else "", lw=lw, color=current_color, alpha=alpha)
                if plotInSamples:
                    axs[1].plot(self.expectedTime[segment_start:segment_end] - self.expectedTime[segment_start],
                                self.deviationFromNominal[segment_start:segment_end] / self.deltaT,
                                label=self.title if segment_idx == 0 else "", lw=lw, color=current_color, ls=':',
                                alpha=alpha)

        if show or save:
            if LANG == 'EN':
                axs[0].set_xlabel(r"\textbf{Relative time in s}")
                axs[0].set_ylabel(r"\textbf{Time Interval Error (TIE) in " + unit + "}")
            if LANG == 'DE':
                axs[0].set_xlabel(r"\textbf{Relative Zeit in s}")
                axs[0].set_ylabel(r"\textbf{Zeitabweichung (TIE) in " + unit + "}")

            for ax in axs:
                ax.ticklabel_format(axis='both', style='plain')
            axs[0].legend(loc='upper left', ncol=2)

            if plotInSamples or plotInSamplesAxis:
                if LANG == 'EN':
                    axs[1].set_ylabel(r"\textbf{TIE in sampling intervals} $\Delta t=\frac{1}{\overline{f_\text{s}}}$")
                if LANG == 'DE':
                    axs[1].set_ylabel(r"\textbf{Zeitabweichung in} $\Delta t=\frac{1}{\overline{f_\text{s}}}$")

            axs[0].grid()

        if yLims is not None:
            axs[0].set_ylim(yLims)

        if plotInSamplesAxis and not plotInSamples:
            # To get the time deviation in samples we just rescale the limits of the axis and let mpl do the rest
            timeUnitsPerDT = self.deltaT * yScaleFactor
            axs[1].set_ylim(axs[0].get_ylim()[0] / timeUnitsPerDT, axs[0].get_ylim()[1] / timeUnitsPerDT)

        if show:
            fig.show()

        if save:
            fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' + str(
                lengthInS) + 's_' + 'TimeDev_Corr_ ' + trueFalseAnAus[correctLinFreqDrift] + '_TimeDevitions.png'),
                        dpi=300,
                        bbox_inches='tight')
            fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' + str(
                lengthInS) + 's_' + 'TimeDev_Corr_' + trueFalseAnAus[correctLinFreqDrift] + '_uncerComps.pdf'), dpi=300,
                        bbox_inches='tight')
            globals()['figSaveCounter'] += 1
            fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' + str(
                lengthInS) + 's_' + 'TimeDev_Corr_' + trueFalseAnAus[correctLinFreqDrift] + '_uncerComps.svg'), dpi=300,
                        bbox_inches='tight')
            globals()['figSaveCounter'] += 1

        return fig, axs

    def getrandomDeviations(self,length,reytryes=1000):
        isContinousDataSliceRetryCount=0
        while isContinousDataSliceRetryCount<reytryes:
            idx=np.random.randint(self.relSampleNumber.size-(length+1))
            if self.relSampleNumber[idx+length]-self.relSampleNumber[idx]==length:
                break
            else:
                isContinousDataSliceRetryCount+=1
                RuntimeWarning(str(self.title)+" Hit hole in Data")

        return self.deviationFromNominal[idx:idx+length]

    def plotAkf(self,sampleFreq=1000,length=1048576*16):
        fig,ax=plt.subplots()
        tmp=np.zeros(length)
        tmp=self.interpolatedDeviationFromNominal[0:length]
        akf = correlate(tmp, tmp, mode='full')
        akf =akf/np.max(akf)

        gausnoise=np.random.normal(scale=self.std,size=tmp.size)
        gausnoise=gausnoise/np.max(gausnoise)
        akf_gausNoise = correlate (gausnoise,gausnoise, mode='full')
        akf_gausNoise=akf_gausNoise/np.max(akf_gausNoise)
        deltaT=1/sampleFreq
        lag=correlation_lags(tmp.size,tmp.size)*deltaT
        ax.plot(lag,akf,label=r'\textbf{\textbf{Aufgeizeichneter \textit{Jitter}}}')
        ax.plot(lag, akf_gausNoise, label=r'\textbf{Nomalverteilter Jitter} $\sigma = '+str(self.std)+'~\text{ns}$')
        ax.set_xlabel(r'\textbf{Zeitverschiebungen} $\tau$ \textbf{in } s')
        ax.set_ylabel(r'\textbf{Autokorrelations Funktion} $AKF$ \textbf{in } R.U s')
        ax.grid()
        ax.legend(ncol=2)
        fig.show()

    def plotFFT(self,sampleFreq=1000,plotPhase=True,fftlength=1048576*16):
        def abs2(x):
            return x.real ** 2 + x.imag ** 2
        nnumOFSlices=int(np.floor(self.interpolatedDeviationFromNominal.shape[0]/fftlength))
        if plotPhase:
            fig,axs=plt.subplots(2,sharex=True)
        else:
            fig, ax = plt.subplots()
            axs=np.array([ax])
        axs[0].set_yscale('log')
        freqs = np.fft.rfftfreq(fftlength, d=1/sampleFreq)
        scale = 2.0 / (fftlength*fftlength)
        quantNoise=np.random.uniform(size=fftlength)*1/(108e6)*1e9
        quantNoise=quantNoise-(1/54e6)*1e9
        gausnoise=np.random.uniform(size=fftlength)*(self.std)
        gausnoise=gausnoise-(self.std)/2
        fft_gausNoise = np.fft.rfft(gausnoise)
        fft_QuantNoise = np.fft.rfft(quantNoise)
        sliceFFTResultsAbsSqared=np.zeros([nnumOFSlices,freqs.size])
        for i in range(nnumOFSlices):
            print("FFT "+str(i/nnumOFSlices*100)+"% done")
            tmp=copy.deepcopy(self.interpolatedDeviationFromNominal[(fftlength*i):(fftlength+fftlength*i)])*1e9
            fftresult = np.fft.rfft(tmp)
            sliceFFTResultsAbsSqared[i]=abs2(fftresult)*scale
            if i==0:
                axs[0].plot(freqs,abs2(fftresult)* scale,alpha=1/nnumOFSlices,label=r'\textbf{Aufgeizeichneter \textit{Jitter}}',color='tab:blue')
                if plotPhase:
                    axs[1].plot(freqs,np.unwrap(np.angle(fftresult)) / np.pi,label=r'\textbf{\textbf{Aufgeizeichneter \textit{Jitter}}}',color='tab:blue')
            else:
                axs[0].plot(freqs,abs2(fftresult)*scale,alpha=1/nnumOFSlices,color='tab:blue')
                if plotPhase:
                    axs[1].plot(freqs,np.unwrap(np.angle(fftresult)) / np.pi,color='tab:blue')
        axs[0].plot(freqs, np.mean(sliceFFTResultsAbsSqared,axis=0),
                    label=r'\textbf{Mittelwert aufgeizeichneter \textit{Jitter}}', color='tab:blue')
        axs[0].plot(freqs, np.ones(freqs.size) * np.mean(abs2(fft_QuantNoise)*scale),label=r'\textbf{Gleich verteiltes Quantisierungs Rauschen Interval '+"%.2f" % ((1/108e6)*1e9)+' ns }',color='tab:orange')
        axs[0].plot(freqs, np.ones(freqs.size) * np.mean(abs2(fft_gausNoise)*scale),label=r'\textbf{Nomalverteilter Jitter} $\sigma = 43~\text{ns}$',color='tab:red')

        axs[0].set_ylabel(r'\textbf{Jitter~PSD in $\frac{{\text{\textbf{ns}}}^2}{\text{\textbf{Hz}}}$')
        axs[0].grid()
        axs[0].legend(ncol=2)
        if plotPhase:

            axs[1].set_ylabel(r'\textbf{\textit{unwrapped} Phase}\\ $\varphi$ \textbf{in} rad')
            axs[1].plot(freqs,np.unwrap(np.angle(fft_gausNoise)),label=r'\textbf{Gleich verteiltes Quantisierungs Rauschen Interval '+"%.2f" % ((1/108e6)*1e9)+' ns }',color='tab:orange')
            axs[1].plot( freqs,np.unwrap(np.angle(fft_QuantNoise)),
                    label=r'\textbf{Nomalverteiltes Rauschen} $\sigma = 43~\text{ns}$',color='tab:red')
            axs[1].legend(ncol=2)
            axs[1].set_xlabel(r'\textbf{Frequenz $f$ in Hz}')
            axs[1].grid()
        else:
            axs[0].set_xlabel(r'\textbf{Frequenz $f$ in Hz}')
        fig.tight_layout()
        fig.show()

    def plotPhaseNoise(self,sampleFreq=None,samplefreqCorr='local',fftlength=1048576*2,plotRaw=False,fig=None,axs=None,filterWidth=1,show=True,plotTimeDevs=False,lw=PLTSCALFACTOR,signalFreq=None,plotSincSensForLength=None,unit='dBc',save=False,xLims=None,yLims=None):
        fftlength=int(fftlength)
        if sampleFreq==None:
            sampleFreq=self.fs
        if signalFreq==None:
            signalFreq=1.0
        nnumOFSlices=int(np.floor(self.interpolatedDeviationFromNominal.shape[0]/fftlength))
        if fig==None and axs==None:
            fig, axs = plt.subplots()
        if plotTimeDevs:
            figTimeDev,axTimeDev=plt.subplots()

        freqs = np.fft.fftshift(np.fft.fftfreq(fftlength, d=1/sampleFreq))
        sliceFFTResultsAbs=np.zeros([nnumOFSlices,freqs.size])
        correctedFreqs=np.zeros(nnumOFSlices)
        for i in range(nnumOFSlices):
            print("FFT "+str(i/nnumOFSlices*100)+"% done")
            tmp=(copy.deepcopy(self.interpolatedDeviationFromNominal[(fftlength*i):(fftlength+fftlength*i)]))#+np.arange(fftlength)*1/sampleFreq).astype(self.floatType)
            if samplefreqCorr=='local':
                idx = np.arange(tmp.size)*self.deltaT
                slopeDeltaT, offset = np.polyfit(idx, tmp.astype(np.float64), 1)
                DeltaFreq=slopeDeltaT/self.deltaT
                correctedTimes=tmp-(np.arange(tmp.size)*(self.deltaT))*slopeDeltaT-offset
                correcedFreq=sampleFreq + DeltaFreq
                simuSin=np.cos(correctedTimes*(correcedFreq)*2*np.pi)+1j*np.sin(correctedTimes*(correcedFreq)*2*np.pi)
                fftresult = np.fft.fftshift(np.fft.fft(simuSin))
                sliceFFTResultsAbs[i]=abs(fftresult)
                if plotTimeDevs:
                    axTimeDev.plot(correctedTimes,label=r'\textbf{'+self.title+' Slice'+str(i)+'}',lw=lw)
                correctedFreqs[i]=correcedFreq
            else:
                simuSin=np.cos(tmp*sampleFreq*2*np.pi)+1j*np.sin(tmp*sampleFreq*2*np.pi)
                fftresult = np.fft.fftshift(np.fft.fft(simuSin))
                sliceFFTResultsAbs[i]=abs(fftresult)
                if plotTimeDevs:
                    axTimeDev.plot(tmp,lw=lw)
            print('SUM:' + str(np.sum(sliceFFTResultsAbs[i])/fftlength))
        if samplefreqCorr=='local':
            self.fs=np.mean(correctedFreqs)
            self.fs_std=np.std(correctedFreqs)
        sampleFrequFloat=uncertainties.ufloat(self.fs,self.fs_std*2)
        if unit=='dBc':
            psdMean=np.mean((sliceFFTResultsAbs ** 2) * (1 / (sampleFreq * fftlength)),axis=0)
            psdMean=psdMean*2
            p=axs.plot((freqs/(sampleFreq))*signalFreq, 10*np.log10(gaussian_filter(psdMean,filterWidth)),
                    label=r'\textbf{'+self.title,lw=lw)#+ $f_\text{s} = '+' {:.1u}'.format(sampleFrequFloat).replace('+/-',r'\pm')+'$ Hz }'
        else:
            psdMean=np.mean((sliceFFTResultsAbs ** 2) * (1 / (sampleFreq * fftlength)),axis=0)
            psdMean=psdMean*2
            p=axs.plot((freqs/(sampleFreq))*signalFreq, gaussian_filter(np.mean((sliceFFTResultsAbs)/fftlength,axis=0),filterWidth),
                    label=r'\textbf{'+self.title,lw=lw)#+' $f_\text{s} = '+' {:.1u}'.format(sampleFrequFloat).replace('+/-',r'\pm')+'$ Hz }'
        if plotSincSensForLength!=None:
            #create shadow axis to optain second legend
            axSincSens=axs.twinx()
            labelPrefixDict={'EN':'Sine approx. sensitivity ','DE':'Sinus Approximation Sensitivität '}
            if isinstance(plotSincSensForLength, list):
                sincFreqs=np.linspace(-signalFreq*0.5,signalFreq*0.5,num=100000,endpoint=True)
                for length in plotSincSensForLength:
                    WindowAmps = abs(np.sinc(sincFreqs * length))
                    if unit == 'dBc':
                        WindowAmps=10 * np.log10(WindowAmps)
                    line=axSincSens.plot(sincFreqs,WindowAmps,ls='--',label=r'\textbf{'+labelPrefixDict[LANG]+str(length)+' s}',color=axs._get_lines.get_next_color())
                    line[0].set_zorder(-1)
            if isinstance(plotSincSensForLength, dict):
                if 'poles' in plotSincSensForLength.keys():
                    numPoles=plotSincSensForLength['poles']
                    for length in plotSincSensForLength['length']:
                        maxfreq=numPoles*1/length
                        minFreq=-maxfreq
                        sincFreqs=np.linspace(minFreq,maxfreq,num=100000,endpoint=True)
                        WindowAmps = abs(np.sinc(sincFreqs * length))
                        if unit == 'dBc':
                            WindowAmps=10 * np.log10(WindowAmps)
                        line=axSincSens.plot(sincFreqs,WindowAmps,ls='--',label=r'\textbf{'+labelPrefixDict[LANG]+str(length)+' s}',lw=lw*0.5,color=axs._get_lines.get_next_color())
                        line[0].set_zorder(-1)
                else:
                    sincFreqs=np.linspace(-plotSincSensForLength['maxFreq'],plotSincSensForLength['maxFreq'],num=100000,endpoint=True)
                    for length in plotSincSensForLength['length']:
                        WindowAmps = abs(np.sinc(sincFreqs * length))
                        if unit == 'dBc':
                            WindowAmps=10 * np.log10(WindowAmps)
                        line=axSincSens.plot(sincFreqs,WindowAmps,ls='--',label=r'\textbf{'+labelPrefixDict[LANG]+str(length)+' s}',lw=lw*0.5,color=axs._get_lines.get_next_color())
                        line[0].set_zorder(-1)
            axSincSens.set_ylim(axs.get_ylim()) #scale axis like the original
            axSincSens.get_yaxis().set_visible(False)# deactivate gost axis visibility

        if plotRaw:
            for i in range(nnumOFSlices):
                if unit == 'dBc':
                    axs.plot((freqs/sampleFreq)*signalFreq,10*np.log10(sliceFFTResultsAbs[i]),alpha=1/nnumOFSlices,color=p[0].get_color(),lw=lw)#label=r'\textbf{'+self.title+'}'
                if unit == 'A.U':
                    axs.plot((freqs / sampleFreq) * signalFreq, sliceFFTResultsAbs[i], alpha=1 / nnumOFSlices, color=p[0].get_color(), lw=lw)  # label=r'\textbf{'+self.title+'}'

        if show or save:
            if unit !='dBc':
                if LANG=='EN':
                    axs.set_ylabel(r'\textbf{Phase noise amplitude in $\frac{\text{\textbf{A. U.}}^2}{\text{\textbf{Hz}}}$')
                if LANG=='DE':
                    axs.set_ylabel(r'\textbf{Phasenrauschamplitude in $\frac{\text{\textbf{A. U.}}^2}{\text{\textbf{Hz}}}$')
            else:
                if LANG == 'EN':
                    axs.set_ylabel(r'\textbf{Phase noise PSD in} $\frac{\text{\textbf{dBC}}}{\text{\textbf{Hz}}}$')
                if LANG == 'DE':
                    axs.set_ylabel(r'\textbf{Phasenrauschleistungsdichte in} $\frac{\text{\textbf{dBc}}}{\text{\textbf{Hz}}}$')
            if signalFreq!=1.0:
                if LANG == 'EN':
                    axs.set_xlabel(r'\textbf{Offset~frequency to '+str(signalFreq)+' Hz Signal in Hz}')
                if LANG == 'DE':
                    axs.set_xlabel(r'\textbf{Frequenzdifferenz zu einem ' + locale.format_string('%g',signalFreq) + ' Hz Signal in Hz}')
            else:
                if LANG== 'EN':
                    axs.set_xlabel(r'$\frac{{\text{\textbf{Offset~frequency}}}}{\text{\textbf{Signal~frequency}}}$ \textbf{in} $\frac{\text{\textbf{Hz}}}{\text{\textbf{Hz}}}$')
                if LANG== 'DE':
                    axs.set_xlabel(r'$\frac{{\text{\textbf{Frequenzdifferenz}}}}{\text{\textbf{Signalfrequenz}}}$ \textbf{in} $\frac{\text{\textbf{Hz}}}{\text{\textbf{Hz}}}$')
            axs.grid(True, which="both")
            axs.legend(ncol=1,loc='upper left')
            try:
                axSincSens.legend(ncol=1, loc='upper right')
            except:
                pass
            if yLims is not None:
                axs.set_ylim(yLims)
            #fig.tight_layout()
            fig.show()
        if plotTimeDevs:
            axTimeDev.set_ylabel(r'\textbf{Time Deviation from Nominal in ns}')
            axTimeDev.set_xlabel(r'\textbf{Releative time from slice start in s}')
            axTimeDev.legend(ncol=2)
            figTimeDev.show()
        if xLims is not None:
            axs.set_xlim(xLims)
            try:
                axSincSens.set_xlim(xLims)
            except:
                pass
        if save:

            try:
                paramsStr=str(plotSincSensForLength['maxFreq']).replace(' ','_')+'_Hz_'+'_'.join(str(v) for v in plotSincSensForLength['length'])
            except:
                paramsStr = "None"
            fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' +'sincSens_'+paramsStr+'_PhaseNoise.png'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' +'sincSens_'+paramsStr+'_PhaseNoise.pdf'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' + 'sincSens_' + paramsStr + '_PhaseNoise.svg'), dpi=300, bbox_inches='tight')
            globals()['figSaveCounter']+=1
        return fig,axs

    """
    def plotAllanDev(self,fig=None,ax=None,show=False):
        if fig==None and ax==None:
            fig, ax = plt.subplots()
        phaseDevInrad=self.interpolatedDeviationFromNominal/self.deltaT
        observationLength=self.expectedTime[-1]
        taus=np.logspace(-3, np.log10(observationLength/4), 100)#100 points logspaced until observation length
        (t2, ad, ade, adn) = allantools.oadev(phaseDevInrad, rate=self.fs, data_type="phase", taus=taus)
        ax.loglog(t2, ad,label=self.title)
        if show:
            ax.grid()
            ax.legend()
            ax.set_xlabel(r'\textbf{Averaging time $\tau$ in s}')
            ax.set_ylabel(r'\textbf{Relative Allan deviation $\sigma(\tau)$ in cycles}')
            fig.show()
        return fig,ax
    """

def generateFitWithPhaseNoise(freq,fs=1000,t_jitter=100e-9,lengthInS=jitterSimuLengthInS,A0=1,phi0=0,linearFreqCorrection=localFreqqCorr):
    #TODO change interface
    if t_jitter <= 0:
        fs=jitterGensForSimulations[int(-1*t_jitter)].fs
    originalTimpoints=np.linspace(0,lengthInS,num=int(fs*lengthInS))
    Signal=A0*np.sin(originalTimpoints*np.pi*2*freq+phi0)
    if t_jitter >0:
        jitter=np.random.normal(scale=t_jitter, size=Signal.size)
    else:
        jitter=jitterGensForSimulations[int(-1*t_jitter)].getrandomDeviations(Signal.size)
    if linearFreqCorrection:
        idx=np.arange(originalTimpoints.size)
        m, b = np.polyfit(idx, jitter.astype(np.float64), 1)
        timeWJitter=originalTimpoints+jitter-(idx*m+b)
    else:
        timeWJitter = originalTimpoints + jitter
    fitparams=st.threeparsinefit(Signal,timeWJitter,freq)
    del jitter ,timeWJitter,Signal,originalTimpoints
    return st.phase(fitparams)-phi0,st.amplitude(fitparams)/A0

def getmuAndSTdForFreq(testparams,numOfruns=200):
    freq=testparams[0]
    t_jitter=testparams[1]
    length = testparams[2]
    FitPhases=np.zeros(numOfruns)
    FitMags=np.zeros(numOfruns)
    for i in range(numOfruns):
        FitPhases[i],FitMags[i]=generateFitWithPhaseNoise(freq,t_jitter=t_jitter,lengthInS=length)
    ampPercentiles=np.percentile(FitMags, np.array([5,32,50,68,95]))
    phasePercentiles=np.percentile(FitPhases, np.array([5,32,50,68,95]))
    stdPhase=np.std(FitPhases)
    meanPhase=np.mean(FitPhases)
    stdMag=np.std(FitMags)
    meanMag=np.mean(FitMags)
    del FitPhases,FitMags
    return stdPhase,\
           meanPhase,\
           stdMag,\
           meanMag,\
           ampPercentiles[0],\
           ampPercentiles[1],\
           ampPercentiles[2],\
           ampPercentiles[3],\
           ampPercentiles[4],\
           phasePercentiles[0],\
           phasePercentiles[1],\
           phasePercentiles[2],\
           phasePercentiles[3],\
           phasePercentiles[4]



def find_nearest_indices(long_vector, short_vector):
    # Compute the absolute differences using broadcasting
    diffs = np.abs(long_vector[:, np.newaxis] - short_vector)
    # Find the indices of the minimum differences
    nearest_indices = np.argmin(diffs, axis=0)
    return nearest_indices

class SineExcitationExperemeint:
    def __init__(self,dataFile,idx,sensor='0xbccb0000_MPU_9250',qunatity='Acceleration',mainAxis=2,interpolations=[],interpolationFactors=[]):
        self.experimentIDX=idx
        self.dataFile=dataFile
        self.sensorStartIDX=self.dataFile['EXPERIMENTS/Sine excitation']["{:05d}".format(idx)+'Sine_Excitation'][sensor].attrs['Start_index']
        self.sensorStopIDX=self.dataFile['EXPERIMENTS/Sine excitation']["{:05d}".format(idx)+'Sine_Excitation'][sensor].attrs['Stop_index']
        self.data=self.dataFile['RAWDATA'][sensor][qunatity][:,self.sensorStartIDX:self.sensorStopIDX]
        self.reltime=self.dataFile['RAWDATA'][sensor]['Absolutetime'][0,self.sensorStartIDX:self.sensorStopIDX]
        self.reltime=self.reltime-self.reltime[0]
        self.reltime=self.reltime.astype(float)/1e9
        self.fs=self.reltime.size/self.reltime[-1]
        self.deltaT=1/self.fs
        self.freq=self.dataFile['EXPERIMENTS/Sine excitation']["{:05d}".format(idx)+'Sine_Excitation'][sensor][qunatity]['Sin_Fit_freq'][:,0][2]
        abcw=st2.fourparsinefit(self.data[mainAxis,:],self.reltime,self.freq)
        self.actualFreq=abcw[3]
        self.gernerateFFT()
        self.interPolationFactors=interpolationFactors
        self.interpolations=interpolations
        self.generateInterpolatedFFT()#interpolations=[ "linear", "nearest", "slinear", "quadratic", "cubic", "previous", "next"]
        self.generateMultiSineFit()
        self.name='MultiSineFFTComp_'+self.dataFile['RAWDATA'][sensor].attrs['Sensor_name'].replace(' ','_')+'_Exp_'+str(idx)+'_Axis_'+str(mainAxis)+'_freq_'+f'{self.freq:.2f}'+"_Hz"
        #self.getSNR()
        print("INIT DONE")


    def gernerateFFT(self):
        lengthmax=self.data.shape[1]
        lengthToTest=int(lengthmax/2)+np.arange(int(lengthmax/2))
        fftbinwidth=self.fs/lengthToTest
        nonIntPeriodFraction=self.actualFreq % fftbinwidth
        self.numPointsToUse=lengthToTest[np.argmin(nonIntPeriodFraction)]
        self.fftLowLeak=2*np.fft.rfft(self.data[:,:self.numPointsToUse],axis=1)/self.numPointsToUse
        self.fftFreqslowLeak=np.fft.rfftfreq(self.numPointsToUse,d=self.deltaT)
        self.fft=2*np.fft.rfft(self.data[:,:],axis=1)/lengthmax
        self.fftFreqs=np.fft.rfftfreq(lengthmax,d=self.deltaT)

    def generateInterpolatedFFT(self):#interpolations=[ "linear", "nearest", "slinear", "quadratic", "cubic", "previous", "next"]
        self.interpolatedFFTFreqsLowLeak={}
        self.interpolatedFFTFreqs={}
        self.interpolatedFFTLowLeak={}
        self.interpolatedFFT={}
        for interpolationFactor in self.interPolationFactors:
            numAxis=self.data.shape[0]
            aqTimesLowLeak=np.linspace(self.reltime[0],self.reltime[self.numPointsToUse-1],num=self.numPointsToUse*interpolationFactor)
            self.interpolatedFFTFreqsLowLeak[interpolationFactor] = np.fft.rfftfreq(aqTimesLowLeak.size, d=self.deltaT / interpolationFactor)
            self.interpolatedFFTLowLeak[interpolationFactor]={}
            for interpolatorKind in self.interpolations:
                print("Interpolating with "+str(interpolationFactor)+" times "+interpolatorKind)
                result=[]
                for i in range(numAxis):
                    interpolator=sp.interpolate.interp1d(self.reltime,self.data[i,:],kind=interpolatorKind)
                    interpolatedData=interpolator(aqTimesLowLeak)
                    result.append(2*np.fft.rfft(interpolatedData)/interpolatedData.size)
                self.interpolatedFFTLowLeak[interpolationFactor][interpolatorKind]=np.array(result)

            aqTimes = np.linspace(self.reltime[0], self.reltime[-1],num=self.reltime.size * interpolationFactor)
            self.interpolatedFFTFreqs[interpolationFactor] = np.fft.rfftfreq(aqTimes.size, d=self.deltaT / interpolationFactor)
            self.interpolatedFFT[interpolationFactor] = {}
            for interpolatorKind in self.interpolations:
                result=[]
                for i in range(numAxis):
                    interpolator=sp.interpolate.interp1d(self.reltime,self.data[i,:],kind=interpolatorKind)
                    interpolatedData=interpolator(aqTimes)
                    result.append(2*np.fft.rfft(interpolatedData)/interpolatedData.size)
                self.interpolatedFFT[interpolationFactor][interpolatorKind]=np.array(result)

    def generateMultiSineFitOLD(self,numLinesAround=100,numOverTones=5):
        self.numLinesAround=numLinesAround
        self.numOverTones=numOverTones
        self.binwidth=self.fftFreqs[1]-self.fftFreqs[0]
        freqs = []
        freqs.append((np.arange(2*numLinesAround+1)+1)*self.binwidth)
        for k in range(numOverTones):
            freqs.append((np.arange(numLinesAround * 2 + 1) - numLinesAround) * self.binwidth + self.actualFreq * (k + 1))
        self.multisineFitFreqs = np.array(freqs).flatten()
        multiSineParams = []
        multiSineParamsABC = []
        for i in range(self.data.shape[0]):
            abc=st2.multi_threeparsinefit(self.data[i,:],self.reltime,self.multisineFitFreqs)
            multiSineParamsABC.append(abc)
            fitResult = st2.multi_complex(abc)
            multiSineParams.append(fitResult)
        self.multiSineFitresults=np.array(multiSineParams)

    def generateMultiSineFit(self, numLinesAround=100, numOverTones=5):
        self.numLinesAround = numLinesAround
        self.numOverTones = numOverTones
        self.binwidth = self.fftFreqs[1] - self.fftFreqs[0]
        fs = self.fs

        rawfreqs = []
        rawfreqs.append((np.arange(2*numLinesAround+1)+1)*self.binwidth)
        self.startStopFreqs=[(rawfreqs[0][0],rawfreqs[0][-1])]
        self.startStopIDXs=[]
        for k in range(numOverTones):
            rawfreqs.append((np.arange(numLinesAround * 2 + 1) - numLinesAround) * self.binwidth + self.actualFreq * (k + 1))
            self.startStopFreqs.append([self.actualFreq * (k + 1) - numLinesAround * self.binwidth, self.actualFreq * (k + 1) + numLinesAround * self.binwidth])
        basebandFreqregions= []
        def ConverToBasebandFreqs(region):
            nyqistbandStart=np.round(region[0]/((self.fs)/2)-0.5)
            nyqistbandStop=np.round(region[1]/((self.fs)/2)-0.5)
            if nyqistbandStart!=nyqistbandStop:
                print("WAAAA region is in tow nyquist bands using upper band; baseband will be negative")
            nyqistband=np.max([nyqistbandStart,nyqistbandStop])
            basebandFreqs=np.array(region)-(nyqistband*(self.fs/2))
            return basebandFreqs
        def checkRegionOverlap(reg1,reg2):
            baseBandReg1=ConverToBasebandFreqs(reg1)
            baseBandReg2=ConverToBasebandFreqs(reg2)
            width1=(reg1[1]-reg1[0])/2
            witdh2=(reg2[1]-reg2[0])/2
            mindistance=(width1+witdh2)
            dist=np.mean(baseBandReg1)-np.mean(baseBandReg2)
            if abs(dist)<mindistance:
                return True
            else:
                return False

        freqs=[]
        numskippedBands=0 # dirty hack to skip overlapping bands and decrease the number of all bands replace with actual start an stop handling and propper indexing thiw ditcs to contorl the regions
        for i,band in enumerate(self.startStopFreqs):
            if i==0:
                basebandFreqregions.append([band[0], band[1]])
                freqs.append(rawfreqs[i])
            else:
                overLapDetected=False
                for region in basebandFreqregions:
                    if checkRegionOverlap(region,band):
                        overLapDetected=True
                        print("Region overlap detected skipped fresquency band" + str(band) + "Due to overlap with" + str(region))
                if not overLapDetected:
                    basebandFreqregions.append([band[0], band[1]])
                    freqs.append(rawfreqs[i])
                else:
                    numskippedBands += 1

        self.multisineFitFreqs = np.array(freqs).flatten()
        self.numOverTones-=numskippedBands
        multiSineParams = []
        multiSineParamsABC = []
        for i in range(self.data.shape[0]):
            abc = st2.multi_threeparsinefit(self.data[i, :], self.reltime, self.multisineFitFreqs)
            multiSineParamsABC.append(abc)
            fitResult = st2.multi_complex(abc)
            multiSineParams.append(fitResult)
        self.multiSineFitresults = np.array(multiSineParams)


    def getSNR(self,axis=2):
        refIDX=self.numLinesAround * 3 + 1
        referenceAMP=abs(self.multiSineFitresults[axis, refIDX])
        referenceN=(np.sum(abs(self.multiSineFitresults[axis, refIDX-self.numLinesAround:refIDX+self.numLinesAround+1]))-referenceAMP)/(2*self.numLinesAround)# we have the ref include so substract it
        self.referenceFFTFreqsForSNR=self.multisineFitFreqs[refIDX-self.numLinesAround:refIDX+self.numLinesAround+1]
        self.fitSNR=referenceAMP/referenceN
        print("SineFit SNR is "+str(self.fitSNR))
        self.FFTSNR={}
        for interpolationFactor in self.interPolationFactors:
            self.FFTSNR[interpolationFactor]={}
            for interpolatorKind in self.interpolations:
                idxs=find_nearest_indices(self.interpolatedFFTFreqsLowLeak[interpolationFactor],self.referenceFFTFreqsForSNR)
                centerIDX=idxs[self.numLinesAround]
                spectralData=abs(self.interpolatedFFTLowLeak[interpolationFactor][interpolatorKind][axis, centerIDX - self.numLinesAround:centerIDX+self.numLinesAround + 1])
                fftAMP = abs(spectralData[self.numLinesAround])
                fftN = (np.sum(abs(self.interpolatedFFTLowLeak[interpolationFactor][interpolatorKind][axis, centerIDX - self.numLinesAround:centerIDX+self.numLinesAround + 1])) - fftAMP)/(2*self.numLinesAround)
                SNR=fftAMP/fftN
                self.FFTSNR[interpolationFactor][interpolatorKind]=SNR
                print("SNR for "+str(interpolationFactor)+" times "+interpolatorKind+" is "+str(SNR))
        result={'sineSNR':self.fitSNR,'FFTSNR':self.FFTSNR,'freq':self.freq,'actualFreq':self.actualFreq}
        json.dump(result,open('SNRParams/'+self.name+'SNR_params.json','w'))
        return result

    def plotFFTandSineFit(self,axisToPlot=[2],plotQoutient=False,markerSize=1,plotHighLeak=False,filterWidth=0.5):
        fig=plt.figure()
        if plotQoutient:
            gs = gridspec.GridSpec(len(axisToPlot)*3,1)
            ax=[]
            bax=[]
            baxQuatient=[]
            idxOffset=1
        else:
            gs = gridspec.GridSpec(len(axisToPlot)*2,1)
            ax=[]
            bax=[]
            idxOffset=0
        numPlotsPerQuant=2+idxOffset
        baxXlims=[]
        for i in range(self.numOverTones+1):
            start = i * (2 * self.numLinesAround+1)
            stop = (i + 1) * (2 * self.numLinesAround+1)
            baxXlims.append([self.multisineFitFreqs[start],self.multisineFitFreqs[stop-1]])
        for i in range(len(axisToPlot)):
            ax.append(plt.subplot(gs[i*numPlotsPerQuant,0]))
            bax.append(brokenaxes(xlims=baxXlims,subplot_spec=gs[i*numPlotsPerQuant+1,0],fig=fig,d=.005))
            if plotQoutient:
                baxQuatient.append(brokenaxes(xlims=baxXlims, subplot_spec=gs[i * numPlotsPerQuant + 2, 0], fig=fig,d=.005))
        for i,idx in enumerate(axisToPlot):
            ax[i].semilogy(self.fftFreqs[1:],sp.ndimage.gaussian_filter1d(np.abs(self.fft[idx,1:]),filterWidth),label=r'\textbf{DFT }',lw=1)
            bax[i].plot(self.fftFreqs[1:],np.abs(self.fft[idx,1:]),label=r'\textbf{DFT wening Leckeffekt}',lw=1, marker='o',markersize=markerSize)
            ax[i].semilogy(self.fftFreqslowLeak[1:],sp.ndimage.gaussian_filter1d(np.abs(self.fftLowLeak[idx,1:]),filterWidth),label=r'\textbf{DFT wening Leckeffekt}',lw=1)
            bax[i].plot(self.fftFreqslowLeak[1:],np.abs(self.fftLowLeak[idx,1:]),label=r'\textbf{DFT wening Leckeffekt}',lw=1, marker='o',markersize=markerSize)
        minFFT=np.power(10,np.floor(np.log10(np.min(np.abs(self.fft[axisToPlot,1:])))))
        maxFFT=np.power(10,np.ceil(np.log10(np.max(np.abs(self.fft[axisToPlot,1:])))))
        minSine=np.power(10,np.floor(np.log10(np.min(np.abs(self.multiSineFitresults[axisToPlot,:])))))
        maxSine=np.power(10,np.ceil(np.log10(np.max(np.abs(self.multiSineFitresults[axisToPlot,:])))))
        min=np.min([minFFT,minSine])
        max=np.max([maxFFT,maxSine])

        for j,jdx in enumerate(axisToPlot):
            for i in range(self.numOverTones+1):
                start = i * (2 * self.numLinesAround+1)
                stop = (i + 1) * (2 * self.numLinesAround+1)
                if i==0:
                    firstPlot=ax[j].semilogy(self.multisineFitFreqs[start:stop],
                                abs(self.multiSineFitresults[jdx][start:stop]), lw=1,label=r'\textbf{Multisine-Approximation}')
                else:
                    ax[j].semilogy(self.multisineFitFreqs[start:stop],
                                abs(self.multiSineFitresults[jdx][start:stop]), lw=1,color=firstPlot[0].get_color())
            bax[j].plot(self.multisineFitFreqs,abs(self.multiSineFitresults[jdx]), lw=1, marker='o',markersize=markerSize)
            if plotQoutient:
                nearestIDX=find_nearest_indices(self.fftFreqs,self.multisineFitFreqs)
                quotients=self.fft[jdx,nearestIDX]/self.multiSineFitresults[jdx,:]
                baxQuatient[j].plot(self.multisineFitFreqs,abs(quotients),lw=1, marker='o',markersize=markerSize)
        for i, interPolFactor in enumerate(self.interpolatedFFT.keys()):
            for interpolMethod in self.interpolatedFFT[interPolFactor].keys():
                for j, idx in enumerate(axisToPlot):

                    pointsToPlotLowLeak = int(self.interpolatedFFTFreqsLowLeak[interPolFactor].size / interPolFactor) - 1
                    lastNormalPlot = ax[j].semilogy(self.interpolatedFFTFreqsLowLeak[interPolFactor][1:pointsToPlotLowLeak],
                                   sp.ndimage.gaussian_filter1d(abs(self.interpolatedFFTLowLeak[interPolFactor][interpolMethod][idx, 1:pointsToPlotLowLeak]),filterWidth),
                                   label=r'\textbf{FFT '+str(interPolFactor)+' times ' + interpolMethod + ' Interpolation low Leak}', alpha=0.5, lw=1,
                                   ls=lineSyles[1 + (i % len(lineSyles))][1])

                    lastBrokenPlot = bax[j].semilogy(self.interpolatedFFTFreqsLowLeak[interPolFactor][1:pointsToPlotLowLeak],
                                    np.abs(self.interpolatedFFTLowLeak[interPolFactor][interpolMethod][idx, 1:pointsToPlotLowLeak]),
                                    label=r'\textbf{FFT '+str(interPolFactor)+' times ' + interpolMethod + ' Interpolation low Leak}', alpha=0.5, lw=1,
                                    ls=lineSyles[1 + (i % len(lineSyles))][1])
                    if plotHighLeak:
                        pointsToPlot = int(self.interpolatedFFTFreqs[interPolFactor].size / interPolFactor) - 1
                        ax[j].semilogy(self.interpolatedFFTFreqs[interPolFactor][1:pointsToPlot],
                                                        sp.ndimage.gaussian_filter1d(np.abs(self.interpolatedFFT[interPolFactor][interpolMethod][idx, 1:pointsToPlot]),filterWidth),
                                                        label=r'\textbf{FFT '+str(interPolFactor)+' times '+ interpolMethod + ' Interpolation}', alpha=0.5,
                                                        lw=1, ls=lineSyles[1 + (i % len(lineSyles))][1], color=lastNormalPlot[-1].get_color())
                        bax[j].semilogy(self.interpolatedFFTFreqs[interPolFactor][1:pointsToPlot],
                                                         np.abs(self.interpolatedFFT[interPolFactor][interpolMethod][idx, 1:pointsToPlot]),
                                                         label=r'\textbf{FFT '+str(interPolFactor)+' times ' + interpolMethod + ' Interpolation}', alpha=0.5,
                                                         lw=1, ls=lineSyles[1 + (i % len(lineSyles))][1], color=lastBrokenPlot[0][-1].get_color())
        for i,idx in enumerate(axisToPlot):
            ax[i].set_xlim([self.fftFreqs[1],self.fftFreqs[-1]])
            ax[i].grid(True,which="major", axis="both",ls="-",lw=PLTSCALFACTOR)
            ax[i].grid(True, which="minor", axis="both", ls="--", lw=0.25*PLTSCALFACTOR,c='grey')
            ax[i].set_ylim([min,max])
            bax[i].set_ylim([min,max])
        for i, idx in enumerate(axisToPlot):
            for j,axis in enumerate(bax[i].axs):
                axis.set_yscale('log')
                axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(0.25,0.5,0.75,1.0),numticks=20))
                axis.grid(True, which="minor", axis="x", ls=":", lw=0.125*PLTSCALFACTOR,c='grey')
                axis.grid(True, which="major", axis="x",ls=":",lw=0.25*PLTSCALFACTOR)

                axis.grid(True, which="minor", axis="y", ls="--", lw=0.25*PLTSCALFACTOR,c='grey')
                axis.grid(True, which="major", axis="y",ls="-",lw=PLTSCALFACTOR)
                axis.set_yticklabels([], minor=True)# disable minor ticks for all
                if j!=0:
                    axis.set_yticklabels([]) #disable major ticks for all but the first
                    for tick in axis.xaxis.get_minor_ticks():
                        tick.tick1line.set_visible(False)
                        tick.tick2line.set_visible(False)
                        tick.label1.set_visible(False)
                        tick.label2.set_visible(False)

        for axis in ax:
            axis.legend()
            axis.set_ylabel(r"\textbf{Amplitude in} $\frac{\text{m}}{\text{s}^2}$")

        for axis in bax:
            axis.set_ylabel(r"\textbf{Amplitude in} $\frac{\text{m}}{\text{s}^2}$")
        if plotQoutient:
            for j,axis in enumerate(baxQuatient):
                axis.grid(True,which="both", axis="both",ls="-")
                axis.set_ylabel(r"\textbf{Amplitude FFT/Fit in A. U.}")
                if j!=0:
                    axis.set_yticklabels([])
                    axis.set_yticklabels([], minor=True)
                    #axis.set_yticks([])
            baxQuatient[-1].set_xlabel(r"\textbf{Frequenz in Hz }")
        else:
            bax[-1].set_xlabel(r"\textbf{Frequenz in Hz}")
        fig.savefig(os.path.join(SAVEFOLDER,self.name + ".svg"))
        fig.savefig(os.path.join(SAVEFOLDER,self.name+".pdf"))
        fig.savefig(os.path.join(SAVEFOLDER,self.name+".png"))
        fig.show()




if __name__ == "__main__":
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL, "de_DE.utf8")
    manager = mp.Manager()

    #measurmentFIle=h5py.File(r"/run/media/seeger01/fe4ba5c2-817c-48d5-a013-5db4b37930aa/data/MPU9250PTB_v5(2)(copy).hdf5",'r')
    #measurmentFIle = h5py.File('/run/media/seeger01/fe4ba5c2-817c-48d5-a013-5db4b37930aa/data/MPU9250CEM(2)(copy).hdf5','r')
    #leadSensorname='0x1fe40000_MPU_9250'
    pathPrefix = r'/home/seeger01/tmp'
    dataFileEXTREF = h5py.File(os.path.join(pathPrefix, 'extRev_single_GPS_1KHz_Edges.hfd5'), 'r')
    #dataFileLSM6DSRX = h5py.File(os.path.join(pathPrefix, 'ST_sensor_test_1667Hz_noTimeGlittCorr.hfd5'), 'r')
    dataFileINTREF = h5py.File(os.path.join(pathPrefix, 'intRev_multi_GPS_1KHz_Edges.hfd5'), 'r')
    dataFileMPU9250 = h5py.File(os.path.join(pathPrefix, 'MPU9250PTB_v5.hdf5'), 'r')
    dataFileBMA280 = h5py.File(os.path.join(pathPrefix, 'BMA280PTB.hdf5'), 'r')
    dataFileLSM6DSRX1667Hz_9 = h5py.File(os.path.join(pathPrefix, 'LSM6DSRX_1667HZ_09.hdf5'), 'r')
    dataFileLSM6DSRX6667Hz = h5py.File(os.path.join(pathPrefix,'ST_sensor_test_6667Hz_2.hfd5'), 'r')
    dataFileADXL355 = h5py.File(os.path.join(pathPrefix, 'ADXL355_4kHz.hfd5'), 'r')

    """
    jitterGen1 = realWordJitterGen(dataFileINTREF, '0x39f50100_STM32_GPIO_Input',r"\textbf{DAU interner Oszillator}")  # nominalfreq=1000)
    jitterGensForSimulations.append(jitterGen1)
    """
    jitterGenMPU9250 = realWordJitterGen(dataFileMPU9250, '0x1fe40000_MPU_9250',r"\textbf{MEMS with PLL-Oszillator $f_\text{sNom}$ = 1~kHz}")  # $f_s=$ \textbf{1001.0388019191 Hz}")
    jitterGensForSimulations.append(jitterGenMPU9250)

    jitterGenBMA280 = realWordJitterGen(dataFileBMA280, '0x1fe40000_BMA_280',r"\textbf{MEMS with RC-Oszillator $f_\text{sNom}$ = 2~kHz}", offset=[int(1.7e6),2048])  # $f_s=$ \textbf{2064.9499858147 Hz} ",)#offset=[100000,1560000+13440562+20])
    jitterGensForSimulations.append(jitterGenBMA280)

    """
    jitterGen2 = realWordJitterGen(dataFileINTREF, '0x60ad0100_STM32_GPIO_Input',r"\textbf{Board 2 int. clock}",offset=[0,5000000])#nominalfreq=1000)
    jitterGensForSimulations.append(jitterGen2)

    jitterGen3 = realWordJitterGen(dataFileEXTREF, '0x39f50100_STM32_GPIO_Input',r"\textbf{Board 1 ext. clock")# 1000 Hz}")#nominalfreq=1000)
    jitterGensForSimulations.append(jitterGen3)

    jitterGen4 = realWordJitterGen(dataFileEXTREF, '0x60ad0100_STM32_GPIO_Input',r"\textbf{Board 2 ext. clock}")#nominalfreq=1000)
    jitterGensForSimulations.append(jitterGen4)
    """
    """
    jitterGenLSM6DSRX = realWordJitterGen(dataFileLSM6DSRX, '0x60ad0000_LSM6DSRX', r"\textbf{LSM6DSRX $f_s$=1.667~kHz}")
    jitterGensForSimulations.append(jitterGenLSM6DSRX)

    #jitterGenLSM6DSRXPolled2KHz = realWordJitterGen(dataFileLSM6DSRX, '0x60ad0000_LSM6DSRX', r"\textbf{LSM6DSRX polled}", pollingFreq=2000.0)
    #jitterGensForSimulations.append(jitterGenLSM6DSRXPolled2KHz)

    jitterGenLSM6DSRXLongTerm = realWordJitterGen(dataFileLSM6DSRXlongTerm, '0x60ad0000_LSM6DSRX',r"\textbf{LSM6DSRX long observation time}")
    jitterGensForSimulations.append(jitterGenLSM6DSRXLongTerm)
    
    jitterGenLSM6DSRX6667Hz = realWordJitterGen(dataFileLSM6DSRX6667Hz, '0x60ad0000_LSM6DSRX',r"\textbf{LSM6DSRX $f_\text{sNom}$ = 6,667~kHz}")
    jitterGensForSimulations.append(jitterGenLSM6DSRX6667Hz)

    jitterGenLSMDSRX_09 = realWordJitterGen(dataFileLSM6DSRX1667Hz_9, '0x60ad0000_LSM6DSRX',r"\textbf{LSMDSRX $f_\text{sNom}$ = 1,667~kHz}")  # $f_s=$ \textbf{2064.9499858147 Hz} ",)#offset=[100000,1560000+13440562+20])
    jitterGensForSimulations.append(jitterGenLSMDSRX_09)
    
    jitterGenADXL355 = realWordJitterGen(dataFileADXL355, '0x0_ADXL_355',r"\textbf{ADXL 355 $f_\text{sNom}$ = 4~kHz}")  # $f_s=$ \textbf{2064.9499858147 Hz} ",)#offset=[100000,1560000+13440562+20])
    jitterGensForSimulations.append(jitterGenADXL355)
    """
    jitterGenMPU9250.plotDeviation(lengthInS=30.0, show=False, correctLinFreqDrift=True, save=True, unit=r'\textmu s',plotInSamplesAxis=True,alpha=1,color=tubscolors[1],maxSegments=1,yLims=[-40,40])
    jitterGenBMA280.plotDeviation( lengthInS=30.0, show=False, correctLinFreqDrift=True, save=True,unit=r'\textmu s', plotInSamplesAxis=True, alpha=1,color=tubscolors[2],maxSegments=1,yLims=[-2500,2500])
    jitterGenMPU9250.plotDeviation(lengthInS=30.0, show=False, correctLinFreqDrift=True, save=True, unit=r'\textmu s',plotInSamplesAxis=True,alpha=1,color=tubscolors[1],maxSegments=10,yLims=[-40,40])
    jitterGenBMA280.plotDeviation( lengthInS=30.0, show=False, correctLinFreqDrift=True, save=True,unit=r'\textmu s', plotInSamplesAxis=True, alpha=1,color=tubscolors[2],maxSegments=10,yLims=[-2500,2500])
    jitterGenMPU9250.plotDeviation(lengthInS=30.0, show=False, correctLinFreqDrift=True, save=True, unit=r'\textmu s',plotInSamplesAxis=True,alpha=0.33,color=tubscolors[1],yLims=[-40,40])
    jitterGenBMA280.plotDeviation( lengthInS=30.0, show=False, correctLinFreqDrift=True, save=True,unit=r'\textmu s', plotInSamplesAxis=True, alpha=0.2,color=tubscolors[2],yLims=[-2500,2500])

    measurmentFIle=dataFileMPU9250
    leadSensorname = '0x1fe40000_MPU_9250'
    #def processfitCOmparison(idx):
    #    sinEX=SineExcitationExperemeint(measurmentFIle, idx, sensor=leadSensorname)
    #    sinEX.plotFFTandSineFit()
    #    snrParams=sinEX.getSNR()
    #    return snrParams
    #snrParams=process_map(processfitCOmparison, np.array(np.arange(20)), max_workers=3)


    
    WORKER_NUMBER = 12

    """
    timeDiffDF1=dataFile1['RAWDATA/0x39f50100_STM32_GPIO_Input/Absolutetime'][0,9990-14:20000-24].astype(np.int64)-dataFile1['RAWDATA/0x60ad0100_STM32_GPIO_Input/Absolutetime'][0,10000:20000].astype(np.int64)
    ticksDiffDF1=dataFile1['RAWDATA/0x39f50100_STM32_GPIO_Input/Time_Ticks'][0,9990-14:20000-24].astype(np.int64)-dataFile1['RAWDATA/0x60ad0100_STM32_GPIO_Input/Time_Ticks'][0,10000:20000].astype(np.int64)
    ticksDiffDF1=ticksDiffDF1-ticksDiffDF1[0]
    dataFile2 = h5py.File('/home/benedikt/repos/datareceiver/intRev_multi_GPS_1KHz_Edges.hfd5','r')
    timeDiffDF2=dataFile2['RAWDATA/0x39f50100_STM32_GPIO_Input/Absolutetime'][0,10000:20000].astype(np.int64)-dataFile2['RAWDATA/0x60ad0100_STM32_GPIO_Input/Absolutetime'][0,11999:21999].astype(np.int64)
    ticksDiffDF2 = dataFile2['RAWDATA/0x39f50100_STM32_GPIO_Input/Time_Ticks'][0,10000:20000].astype(np.int64)-dataFile2['RAWDATA/0x60ad0100_STM32_GPIO_Input/Time_Ticks'][0,0,11999:21999].astype(np.int64)
    ticksDiffDF2=ticksDiffDF2-np.mean(ticksDiffDF2)
    dataFile3 = h5py.File('/home/benedikt/repos/datareceiver/datalossTest.hfd5','r')
    timeDiffDF3=dataFile3['RAWDATA/0x39f50100_STM32_GPIO_Input/Absolutetime'][0,10013:20013].astype(np.int64)-dataFile3['RAWDATA/0x60ad0100_STM32_GPIO_Input/Absolutetime'][0,12020:22020].astype(np.int64)
    ticksDiffDF3=dataFile3['RAWDATA/0x39f50100_STM32_GPIO_Input/Time_Ticks'][0,10013:20013].astype(np.int64)-dataFile3['RAWDATA/0x60ad0100_STM32_GPIO_Input/Time_Ticks'][0,12020:22020].astype(np.int64)
    ticksDiffDF2=ticksDiffDF2-np.mean(ticksDiffDF3)
    """

    #jitterGensForSimulations[0].plotAllanDev()
    show=True
    def plot_graphs(jitterGensForSimulations, plots_params):
        for params in plots_params:
            fig, axs = None, None
            for idx, jitter_gen in enumerate(jitterGensForSimulations):
                # Exclude 'type' parameter when calling the actual plotting function
                specific_params = {k: v for k, v in params.items() if k != 'type' and k != 'plotSincSensForLength' }

                # Handling the fig and axs objects
                if fig is not None and axs is not None:
                    specific_params['fig'] = fig
                    specific_params['axs'] = axs

                # Set show to True in the last iteration
                if idx == len(jitterGensForSimulations) - 1:
                    specific_params['show'] = show
                    specific_params['save'] = True
                    if 'plotSincSensForLength' in params.keys():
                        specific_params['plotSincSensForLength'] = params['plotSincSensForLength']

                if params['type'] == 'deviation':
                    fig, axs = jitter_gen.plotDeviation(**specific_params)
                elif params['type'] == 'phaseNoise':
                    fig, axs = jitter_gen.plotPhaseNoise(**specific_params)
                # Add more conditions here for other types of plots
            plt.close(fig)
            del(fig)
            del(axs)


    phaseNoiseLW = 2.0
    plots_params = [
        {
            'type': 'deviation',
            'unit': r'\textmu s',
            'lengthInS': 10.0,
            'plotInSamples': True,
            'maxSegments':1
        },
        #{
        #    'type': 'deviation',
        #    'unit': r'\textmu s',
        #    'lengthInS': 10.0,
        #    'maxSegments': 1e4
        #},
        {
            'type': 'deviation',
            'unit': 'ms',
            'lengthInS': 1000.0,
            'plotInSamples': True,
            'maxSegments': 1
        },
        {
            'type': 'deviation',
            'unit': 'ms',
            'lengthInS': 100.0,
            'correctLinFreqDrift': False,
            'plotInSamples': True,
            'maxSegments': 1
        },
        {
            'type': 'deviation',
            'unit': 'ms',
            'lengthInS': 10.0,
            'correctLinFreqDrift': False,
            'plotInSamples': True,
            'maxSegments': 1
        },
        {
            'type': 'phaseNoise',
            'lw':1.0
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'plotSincSensForLength': {'length': [1.0, 10, 100], 'maxFreq': 0.2},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'plotSincSensForLength': {'length': [1.0,10, 30], 'poles':5},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'plotSincSensForLength': {'length': [1.0, 10, 30], 'poles': 5},
            'lw': phaseNoiseLW
        },
        {
        'type': 'phaseNoise',
        'signalFreq': 500,
            'lw':1.0
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [-1.5, 1.5],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [-1.5, 1.5],
            'plotSincSensForLength': {'length':[1.0, 10, 100],'maxFreq':1.5},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [-0.2, 0.2],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'xLims': [-0.2/100, 0.2/100],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [-0.2, 0.2],
            'plotSincSensForLength': {'length':[1.0, 10, 100],'maxFreq':0.2},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [-0.2, 0.2],
            'unit': 'A.U',
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'unit': 'A.U',
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [-0.2, 0.2],
            'unit': 'A.U',
            'plotSincSensForLength': {'length':[1.0, 10, 100],'maxFreq':0.2},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-0.2, 0.2],
            'plotSincSensForLength': {'length': [5.0, 10, 30],'maxFreq': 0.2},
            'lw': phaseNoiseLW
        },
    ]

    plots_params_diss = [
        {
            'type': 'deviation',
            'unit': 'ms',
            'lengthInS': 1000.0,
            'plotInSamples': True,
            'maxSegments': 1
        },
        #{
        #    'type': 'deviation',
        #    'unit': 'ms',
        #    'lengthInS': 10.0,
        #    'maxSegments': 10000
       #},
        {
            'type': 'deviation',
            'unit': r'\textmu s',
            'lengthInS': 10.0,
            'plotInSamples': True,
            'maxSegments':1
        },

        {
            'type': 'deviation',
            'unit': 'ms',
            'lengthInS': 10.0,
            'plotInSamples': True,
            'maxSegments': 1,
            'yLims':[-1.0,2.0]
        },
        {
            'type': 'phaseNoise',
            'lw':1.0
        },
        {
            'type': 'phaseNoise',
            'xLims': [-0.002, 0.002],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'plotSincSensForLength': {'length': [1.0, 10, 30], 'poles': 3},
            'yLims':[-140,40],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [0, 0.2],
            'yLims': [0, 1],
            'unit': 'A.U',
            'plotSincSensForLength': {'length': [1.0, 10, 30], 'poles': 3},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [0, 0.2],
            'yLims': [0, 1],
            'unit': 'A.U',
            'plotSincSensForLength': {'length': [1.0, 10, 30], 'poles': 3},
            'lw': phaseNoiseLW
        }
    ]
    #plot_graphs(jitterGensForSimulations, plots_params_diss)

    """
    jitterGen1.plotDeviation(fig=figDviation,ax=axDeviation,length=150000,lw=1)
    
    shorterDsetLength=150000#np.min([jitterGen1.dataPoints,jitterGen2.dataPoints])-1024

    axDeviation.plot(jitterGen1.expectedTime[:shorterDsetLength], (
                jitterGen1.AbsoluteTime[:shorterDsetLength].astype(np.int64) - jitterGen2.AbsoluteTime[
                                                                               :shorterDsetLength].astype(np.int64)),
                     label="Time difference multi GNSS int. clock",lw=1)
    axDeviation.plot(jitterGen3.expectedTime[:shorterDsetLength], (
                jitterGen3.AbsoluteTime[:shorterDsetLength].astype(np.int64) - jitterGen4.AbsoluteTime[
                                                                               :shorterDsetLength].astype(np.int64)),
                     label="Time difference single GNSS ext. clock",lw=2.5)
    jitterGen3.plotDeviation(fig=figDviation,ax=axDeviation,length=150000,lw=2.5)
    jitterGen4.plotDeviation(fig=figDviation, ax=axDeviation,length=150000,show=True,lw=2.5)
    
    jitterGenMPU9250.plotDeviation(fig=figDviation, axs=axDeviation,lengthInS=deviationPlotlength)
    jitterGenBMA280.plotDeviation(fig=figDviation, axs=axDeviation,lengthInS=deviationPlotlength, show=True)
    #jitterGenLSM6DSRX.plotDeviation(fig=figDviation, axs=axDeviation,lengthInS=deviationPlotlength, show=True)
    jitterGenLSM6DSRX6667Hz.plotDeviation(fig=figDviation, axs=axDeviation,lengthInS=deviationPlotlength, show=True)
    """
    """
    figPhaseNoise,axPhaseNoise=jitterGen1.plotPhaseNoise(plotRaw=False)
    #jitterGen4.plotPhaseNoise(fig=figPhaseNoise,ax=axPhaseNoise,plotRaw=False)
    jitterGenMPU9250.plotPhaseNoise(fig=figPhaseNoise,axs=axPhaseNoise,plotRaw=False)
    jitterGenBMA280.plotPhaseNoise(fig=figPhaseNoise, axs=axPhaseNoise, plotRaw=False)
    jitterGenLSMDSRX_09.plotPhaseNoise(fig=figPhaseNoise, axs=axPhaseNoise, plotRaw=False)
    jitterGenLSM6DSRX6667Hz.plotPhaseNoise(fig=figPhaseNoise, axs=axPhaseNoise, plotRaw=False)
    jitterGenADXL355.plotPhaseNoise(fig=figPhaseNoise, axs=axPhaseNoise, plotRaw=False)
    """
    """
    freqPoints=500
    ampPoints=0
    SimuPoints =     ampPoints+len(jitterGensForSimulations)
    nsPreAmpStep=20
    lengthInS=10
    freqs=np.zeros(freqPoints * SimuPoints)
    noiseLevel=np.zeros(freqPoints * SimuPoints)
    runNoiselevel=np.append(np.flip(np.arange(len(jitterGensForSimulations)) - (len(jitterGensForSimulations)-1)), np.array(np.arange(SimuPoints - 2) + 1) * nsPreAmpStep * 10e-9)
    for i in range(SimuPoints):
        tmpFreqs=np.linspace(0.1,1000,freqPoints)
        freqToNear=(tmpFreqs % 1000) < 5
        freqToNear+=(tmpFreqs % 500) < 5
        freqToAdd=10*freqToNear
        tmpFreqs+=freqToAdd
        tmpNoiseLevel=np.ones(freqPoints)*runNoiselevel[i]
        freqs[i*freqPoints:(i+1)*freqPoints]=tmpFreqs
        noiseLevel[i * freqPoints:(i + 1) * freqPoints] = tmpNoiseLevel
    length=np.ones(freqs.size)*lengthInS
    testparams=np.array([freqs,noiseLevel,length]).transpose()
    results=process_map(getmuAndSTdForFreq, testparams, max_workers=WORKER_NUMBER,chunksize=1)
    #with closing(Pool()) as p:
    #    results=p.map(getmuAndSTdForFreq, tqdm.tqdm(testparams))
    results=np.array(results)
    bw=np.ones(SimuPoints)

    def plotMagDeviations(idxs=np.arange(len(jitterGensForSimulations))):
        fig1, ax = plt.subplots(figsize =(24, 8))
        #fig1.set_figwidth(12)
        #fig1.set_figheight(4)
        #if LANG=='EN':
            #fig1.suptitle(r"\textbf{Simulated time = " + str(lengthInS) + ' s, local frequency correction '+str(localFreqqCorr)+'}')
        #if LANG=='DE':
            #fig1.suptitle(r"\textbf{Simulationsdauer = " + str(lengthInS) + ' s, Lokalefrequenzkorrektur '+trueFalseAnAus[localFreqqCorr]+'}')
        doFit=False
        plotErrors=True
        for i in idxs:
            tmpFreqs=freqs[i * freqPoints: (i + 1) * freqPoints]
            if i<=(len(jitterGensForSimulations)-1):
                label = jitterGensForSimulations[i].title
            else:
                label=r"\textbf{\textit{simu.} $2\sigma= " + str(2*((i-1) * nsPreAmpStep)) + "$ ns}"
            AMPS=results[i * freqPoints: (i + 1) * freqPoints,6]
            AMPSErrorBottom = results[i * freqPoints: (i + 1) * freqPoints,4]
            AMPSErrorTop = results[i * freqPoints: (i + 1) * freqPoints, 8]
            AMPSError25Bottom = results[i * freqPoints: (i + 1) * freqPoints,5]
            AMPSError75Top = results[i * freqPoints: (i + 1) * freqPoints, 7]
            coveragenameDict={'EN':'coverage','DE':'Konfidenzinterval'}
            if plotErrors:
                dataPlot=ax.plot(tmpFreqs,
                       AMPS,
                       label=r"\textbf{Median }"+label,lw=PLTSCALFACTOR*2,color=tubscolors[i])

                errorPlot2 = ax.fill_between(tmpFreqs,
                                   AMPSErrorBottom,
                                    AMPSErrorTop,
                                    #label=r"\textbf{32\% - 68\% "+coveragenameDict[LANG]+" }" + str(label),
                                    color=dataPlot[0].get_color(),
                                    alpha=0.1,
                                    hatch = 'O')
                errorPlot1 = ax.fill_between(tmpFreqs,
                                   AMPSError25Bottom,
                                    AMPSError75Top,
                                    #label=r"\textbf{5\% - 95\% "+coveragenameDict[LANG]+" }"+str(label),
                                    color=dataPlot[0].get_color(),
                                    alpha=0.1,
                                    ls="--",
                                           hatch='o')

            else:
                dataPlot=ax.plot(tmpFreqs,
                       AMPS,
                       label=label)
            if doFit:
                popt, pcov = curve_fit(gaus, tmpFreqs, AMPS, p0=[1, 5e5])
                ax.plot(tmpFreqs,gaus(tmpFreqs,popt[0],popt[1]),label=r"\textbf{Fited bandwidth = "+"{:.2f}".format(abs(popt[1])/1e6)+" MHz }",color=dataPlot[-1].get_color(),ls='--')
                bw[i]=popt[1]
                print('______'+str(i * nsPreAmpStep)+' ns ___________')
                print(popt)
                print(popt[1]/(i * nsPreAmpStep*10e-9)*(i * nsPreAmpStep*10e-9))
                print('_____________________________________________')
        #ax[0].legend()
        #ax[0].legend(ncol=4)
        ax.legend(ncol=2)
        if LANG=='EN':
            ax.set_xlabel(r"\textbf{Simulated signal frequency in Hz}")
            #ax[0].set_ylabel(r"$2\sigma(\hat{A})$ \textbf{in \%}")
            ax.set_ylabel(r"$\frac{\mathbf{\hat{A}}}{\mathbf{A_{nom}}}$")
        if LANG=='DE':
            ax.set_xlabel(r"\textbf{Signalfrequenz in Hz}")
            #ax[0].set_ylabel(r"$2\sigma(\hat{A})$ \textbf{in \%}")
            ax.set_ylabel(r"\textbf{Magnitude} $\frac{\mathbf{\hat{A}}}{\mathbf{A_{nom}}}$")
        #ax[0].grid(True)
        ax.grid(True)
        #fig1.tight_layout()
        fig1.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' +"Magnitude_degradation_simulation"+ str(lengthInS) + "lockalFreqKoor" +trueFalseAnAus[localFreqqCorr]+'.png'), dpi=300, bbox_inches='tight')
        fig1.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' +"Magnitude_degradation_simulation"+ str(lengthInS) + "lockalFreqKoor" +trueFalseAnAus[localFreqqCorr]+ '.pdf') , dpi=300, bbox_inches='tight')
        globals()['figSaveCounter'] += 1
        fig1.show()

    plotMagDeviations()
    plotMagDeviations([0])
    plotMagDeviations([0, 1, 3, 4])
    plotMagDeviations([0, 1, 3, 4, 5])



    freqPoints=500
    ampPoints=0
    SimuPoints =     ampPoints+len(jitterGensForSimulations)
    nsPreAmpStep=20
    lengthInS=100
    freqs=np.zeros(freqPoints * SimuPoints)
    noiseLevel=np.zeros(freqPoints * SimuPoints)
    runNoiselevel=np.append(np.flip(np.arange(len(jitterGensForSimulations)) - (len(jitterGensForSimulations)-1)), np.array(np.arange(SimuPoints - 2) + 1) * nsPreAmpStep * 10e-9)
    for i in range(SimuPoints):
        tmpFreqs=np.linspace(0.1,1000,freqPoints)
        freqToNear=(tmpFreqs % 1000) < 5
        freqToNear+=(tmpFreqs % 500) < 5
        freqToAdd=10*freqToNear
        tmpFreqs+=freqToAdd
        tmpNoiseLevel=np.ones(freqPoints)*runNoiselevel[i]
        freqs[i*freqPoints:(i+1)*freqPoints]=tmpFreqs
        noiseLevel[i * freqPoints:(i + 1) * freqPoints] = tmpNoiseLevel
    length=np.ones(freqs.size)*lengthInS
    testparams=np.array([freqs,noiseLevel,length]).transpose()
    results=process_map(getmuAndSTdForFreq, testparams, max_workers=WORKER_NUMBER,chunksize=1)
    #with closing(Pool()) as p:
    #    results=p.map(getmuAndSTdForFreq, tqdm.tqdm(testparams))
    results=np.array(results)
    bw=np.ones(SimuPoints)

    plotMagDeviations()
    plotMagDeviations([0])
    plotMagDeviations([0, 1, 3, 4])
    plotMagDeviations([0, 1, 3, 4, 5])
    """
    """
    #fig, ax = plt.subplots(2,sharex=True)
    fig4, ax4 = plt.subplots(1)
    fig4.set_figwidth(12)
    fig4.set_figheight(4)
    #if LANG=='EN':
    #    fig4.suptitle(r"\textbf{Simulated time = " + str(lengthInS) + ' s local frequency correction '+str(localFreqqCorr)+'}')
    #if LANG=='DE':
    #    fig4.suptitle(r"\textbf{Simulationsdauer = " + str(lengthInS) + ' s, Lokalefrequenzkorrektur '+trueFalseAnAus[localFreqqCorr]+'}')
    for i in range(SimuPoints):
        if i<=(len(jitterGensForSimulations)-1):
            label = jitterGensForSimulations[i].title
        else:
            label=r"\textbf{\textit{simu.} $2\sigma= " + str(2*((i-1) * nsPreAmpStep)) + "$ ns}"
        tmpFreqs = freqs[i * freqPoints: (i + 1) * freqPoints]
        sigmaPhase = results[i * freqPoints: (i + 1) * freqPoints,0]
        dataPlot = ax4.plot(tmpFreqs,
                              2 * sigmaPhase / np.pi * 180,
                              label=label)
        #ax[1].plot(tmpFreqs,
        #           results[i * freqPoints: (i + 1) * freqPoints, 1] / np.pi * 180,
        #           label=label)
        
        #if i != 0:
        #    popt, pcov = curve_fit(logGisticPICaped, tmpFreqs, sigmaPhase, p0=[0.001, bw[i],np.pi])
        #    ax[0].plot(tmpFreqs, logGisticPICaped(tmpFreqs, popt[0], popt[1], popt[2]) / (np.pi * 180),
        #               label=r"\textbf{ Fit Bandbreite = }" + "{:.2f}".format(abs(popt[1]) / 1e6) + " MHz",
        #               color=dataPlot[-1].get_color(), ls='--')
        #    bw[i] = popt[1]
        #    print('______' + str(i * nsPreAmpStep) + ' ns ___________')
        #    print(popt)
        #    print(popt[1] / (i * nsPreAmpStep * 10e-9) * (i * nsPreAmpStep * 10e-9))
        #    print('_____________________________________________')
        
    ax4.legend(ncol=4)
    #ax[1].legend(ncol=4)
    if LANG=='EN':
        ax4.set_xlabel(r"\textbf{Simulated signal frequency in Hz}")
        ax4.set_ylabel(r"\textbf{Max. phase deviation $2\sigma \varphi$ in} $^\circ$")
    if LANG=='DE':
        ax4.set_xlabel(r"\textbf{Simulierte Signalfrequenz in Hz}")
        ax4.set_ylabel(r"\textbf{Phasenabweichung $2\sigma \varphi$ in} $^\circ$")
    #ax[1].set_ylabel(r"$\overline{\varphi}-\varphi_{soll}$ \textbf{in} $^\circ$")
    ax4.grid(True)
    #ax[1].grid(True)
    fig4.tight_layout()
    fig4.show()
    if askforFigPickelSave:
        saveImagePickle("Monte Carlo Amp from PhaseNoise", ax4, fig4)
        saveImagePickle("Monte Carlo Phase from PhaseNoise", ax4, fig4)
    """
    """
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    surf=ax2.plot_trisurf(freqs, noiseLevel, results[:, 0],cmap=cm.coolwarm)
    fig2.colorbar(surf, shrink=0.5, aspect=5)
    fig2.show()

    
    lengthPoints=15
    StartLength=64
    noiseLevelToUse=100*10e-9
    freqs=np.zeros(freqPoints*lengthPoints)
    noiseLevel=np.zeros(freqPoints*lengthPoints)
    length = np.ones(freqs.size)

    for i in range(lengthPoints):
        tmpFreqs=np.logspace(1.00,7.0,freqPoints)
        freqToNear=(tmpFreqs % 1000) < 5
        freqToNear = (tmpFreqs % 1000) > 995
        freqToNear+=(tmpFreqs % 500) < 5
        freqToNear += (tmpFreqs % 500) > 495
        freqToAdd=10*freqToNear
        tmpFreqs+=freqToAdd
        tmpNoiseLevel=np.ones(freqPoints)*noiseLevelToUse
        freqs[i*freqPoints:(i+1)*freqPoints]=tmpFreqs
        noiseLevel[i * freqPoints:(i + 1) * freqPoints] = tmpNoiseLevel
        length[i * freqPoints:(i + 1) * freqPoints]=StartLength/((i+1)*(i+1))
    testparams=np.array([freqs,noiseLevel,length]).transpose()
    results=process_map(getmuAndSTdForFreq, testparams, max_workers=WORKER_NUMBER)
    results=np.array(results)


    fig2,ax=plt.subplots(2,sharex=True)
    #fig2.set_figwidth(10)
    #fig2.set_figheight(5)
    #fig2.suptitle(r"\textbf{SampleRate = 1 kHz | 100 ns  \textit{Jitter}}")
    for i in range(lengthPoints):
        ax[0].plot(freqs[i * freqPoints: (i + 1) * freqPoints],
                   2*results[i * freqPoints: (i + 1) * freqPoints,0]/np.pi*180,
                   label=r"\textbf{Dauer= "+"{:.2f}".format(StartLength/((i+1)*(i+1)))+" s}")
        ax[1].plot(freqs[i * freqPoints: (i + 1) * freqPoints],
                   results[i * freqPoints: (i + 1) * freqPoints, 1] / np.pi * 180,
                   label=r"\textbf{Dauer = " + "{:.2f}".format(StartLength/((i+1)*(i+1))) + " s}")
    ax[0].legend(ncol=4)
    ax[1].legend(ncol=4)
    ax[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax[0].set_ylabel(r"$2\sigma(\varphi)$ \textbf{in} $^\circ$")
    ax[1].set_ylabel(r"$\overline{\varphi}-\varphi_{soll}$ \textbf{in} $^\circ$")
    ax[0].grid(True)
    ax[1].grid(True)
    fig2.tight_layout()
    fig2.show()
    
    fig3,ax=plt.subplots(2,sharex=True)
    #fig3.set_figwidth(10)
    #fig3.set_figheight(5)
    #fig3.suptitle(r"\textbf{SampleRate = 1 kHz | 100 ns \textit{Jitter}}")
    for i in range(lengthPoints):
        ax[0].plot(freqs[i * freqPoints: (i + 1) * freqPoints],
                   2*results[i * freqPoints: (i + 1) * freqPoints,2]*100,
                   label=r"\textbf{Dauer= "+"{:.2f}".format(StartLength/((i+1)*(i+1)))+" s}")
        ax[1].plot(freqs[i * freqPoints: (i + 1) * freqPoints],
                   results[i * freqPoints: (i + 1) * freqPoints, 3],
                   label=r"\textbf{Dauer = " + "{:.2f}".format(StartLength/((i+1)*(i+1))) + " s}")
    ax[0].legend(ncol=2)
    ax[1].legend(ncol=2)
    ax[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax[0].set_ylabel(r"$2\sigma(\hat{A})$ \textbf{in \%}")
    ax[1].set_ylabel(r"$\frac{\overline{\hat{A}}}{A_{soll}}$ \textbf{in A. U.}")
    ax[0].grid(True)
    ax[1].grid(True)
    fig3.tight_layout()
    fig3.show()
    """

    sineESs=[]
    SNRS=[]
    for i in [11]:
        sineESs.append(SineExcitationExperemeint(measurmentFIle, i, sensor=leadSensorname))
        sineESs[-1].plotFFTandSineFit()
        SNRS.append(sineESs[-1].getSNR())
    print("Debug")

    print("Hello")

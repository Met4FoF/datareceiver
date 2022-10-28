import numpy as np
import matplotlib.pyplot as plt
import sinetools.SineTools as st
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.signal import correlate
from scipy.signal import correlation_lags
from scipy.ndimage import gaussian_filter
import h5py as h5py


# importing copy module
import copy
def gaus(x,a,sigma):
    x0=0
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def logGisticPICaped(x,k,x0,L):
    return L/(1+np.exp(-k*(x-x0)))

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
PLTSCALFACTOR = 1.5
SMALL_SIZE = 12 * PLTSCALFACTOR
MEDIUM_SIZE = 16 * PLTSCALFACTOR
BIGGER_SIZE = 18 * PLTSCALFACTOR

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE/1.5)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title



class realWordJitterGen:
    def __rpr__(self):
        return str(self.title)+' fs= '+str(self.fs)+' Hz'
    def __init__(self,HDfFile,sensorName,title,nominalfreq=0,offset=[40000,10000]):
        self.datafile=HDfFile
        self.title=title
        self.Dataset=self.datafile['RAWDATA/'+sensorName+'/Absolutetime']
        self.dataPoints=self.datafile['RAWDATA/'+sensorName].attrs['Data_point_number']# use only valide points
        self.AbsoluteTime = self.Dataset[0, 0 + offset[0]:self.dataPoints - offset[1]] - self.Dataset[0, offset[0]]
        self.timeData=((self.AbsoluteTime-self.AbsoluteTime[0])/1e9).astype(np.float128)
        #self.timeData=(self.Dataset[0,0+offset[0]:self.dataPoints-offset[1]]-self.Dataset[0,offset[0]])/1e9# substract first point to avoid precisionlos with f64 the divide by 1e9 to ahve seconds
        self.relSampleNumber = (self.datafile['RAWDATA/'+sensorName+'/Sample_number'][0, 0 + offset[0]:self.dataPoints-offset[1]] - self.datafile['RAWDATA/'+sensorName+'/Sample_number'][0, offset[0]])
        if nominalfreq==0:
            self.fs =  (self.relSampleNumber[-1].astype(np.float128))/self.timeData[-1].astype(np.float128)  # calculate smaple freq
        else:
            self.fs = nominalfreq
        #self.deltaT=self.length/(self.dataPoints-1)
        self.deltaT = 1.0/self.fs
        print("Sample frequency is "+str(self.fs)+' Hz')
        self.expectedTime=self.relSampleNumber.astype(np.float128)*self.deltaT
        self.deviationFromNominal=self.timeData-self.expectedTime#calulate deviation from Expected Mean
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



    def plotDeviation(self,fig=None,axs=None,lengthInS=None,show=False,lw=1.5,correctLinFreqDrift=True):
        if fig==None and axs==None:
            fig,ax=plt.subplots()
            fig.set_figwidth(12)
            fig.set_figheight(4)
            ax2 = ax.twinx()# axis for deviation in samples
            axs=[ax,ax2]
        else:
            ax=axs[0]
            ax2=axs[1]
        if lengthInS ==None:
            length=self.timeData.size
        else:
            length=int(lengthInS*self.fs)

        if correctLinFreqDrift:
            tmp=self.deviationFromNominal[:length]
            correctedTimeDev=tmp-np.arange(length)*(tmp[-1]/length)-tmp[0]
            correctedTime=self.timeData[:length]-np.arange(length)*(tmp[-1]/length)
            line=ax.plot(correctedTime, correctedTimeDev * 1e9, label=self.title, lw=lw)
            ax2.plot(correctedTime, correctedTimeDev/self.deltaT, label=self.title, lw=lw,color=line[0].get_color(),ls='--')
        else:
            line=ax.plot(self.timeData[:length], self.deviationFromNominal[:length]*1e9, label=self.title,lw=lw)
            ax2.plot(self.timeData[:length], self.deviationFromNominal[:length] / self.deltaT, label=self.title, lw=lw, color=line[0].get_color(),ls='--')
        #ax.plot(np.arange(self.interpolatedDeviationFromNominal.size)*self.deltaT,self.interpolatedDeviationFromNominal)
        if show:
            ax.set_xlabel(r"\textbf{Relative time in s}")
            ax.set_ylabel(r"\textbf{Time deviation in ns}")
            ax2.set_ylabel(r"\textbf{Time deviation in samples}")
            ax.ticklabel_format(axis='both',style='plain')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid()
            #fig.tight_layout()
            fig.show()
        return fig,axs
    def getrandomDeviations(self,length,reytryes=1000):
        isContinousDataSliceRetryCount=0
        while isContinousDataSliceRetryCount<reytryes:
            idx=np.random.randint(self.deviationFromNominal.size-(length+1))
            if self.relSampleNumber[idx+length]-self.relSampleNumber[idx]==length:
                break
            else:
                isContinousDataSliceRetryCount+=1
                print("Hit hole in Data")

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
        ax.plot(lag, akf_gausNoise, label=r'\textbf{Nomalverteilter Jitter} $\sigma = '+str(self.std)+'~\mathrm{ns}$')
        ax.set_xlabel(r'\textbf{Zeitverschiebungen} $\tau$ \textbf{in } s')
        ax.set_ylabel(r'\textbf{Autokorrelations Funktion} $AKF$ \textbf{in } R.U s')
        ax.grid()
        ax.legend()
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
        axs[0].plot(freqs, np.ones(freqs.size) * np.mean(abs2(fft_gausNoise)*scale),label=r'\textbf{Nomalverteilter Jitter} $\sigma = 43~\mathrm{ns}$',color='tab:red')

        axs[0].set_ylabel(r'\textbf{$Jitter~PSD$ in $\frac{{\mathrm{ns}}^2}{\mathrm{Hz}}$')
        axs[0].grid()
        axs[0].legend()
        if plotPhase:

            axs[1].set_ylabel(r'\textbf{\textit{unwrapped} Phase}\\ $\varphi$ \textbf{in} rad')
            axs[1].plot(freqs,np.unwrap(np.angle(fft_gausNoise)),label=r'\textbf{Gleich verteiltes Quantisierungs Rauschen Interval '+"%.2f" % ((1/108e6)*1e9)+' ns }',color='tab:orange')
            axs[1].plot( freqs,np.unwrap(np.angle(fft_QuantNoise)),
                    label=r'\textbf{Nomalverteiltes Rauschen} $\sigma = 43~\mathrm{ns}$',color='tab:red')
            axs[1].legend()
            axs[1].set_xlabel(r'\textbf{Frequenz $f$ in Hz}')
            axs[1].grid()
        else:
            axs[0].set_xlabel(r'\textbf{Frequenz $f$ in Hz}')
        fig.tight_layout()
        fig.show()

    def plotPhaseNoise(self,sampleFreq=None,fftlength=1048576,plotRaw=True,plotQuanatisationNoise=0,fig=None,ax=None,filterWidth=5):
        if sampleFreq==None:
            sampleFreq=self.fs
        nnumOFSlices=int(np.floor(self.interpolatedDeviationFromNominal.shape[0]/fftlength))
        if fig==None and ax==None:
            fig, ax = plt.subplots()
            fig.set_figwidth(12)
            fig.set_figheight(4)
            #fig.subplots_adjust(bottom=0.15)
        #ax.set_yscale('log')
        freqs = np.fft.rfftfreq(fftlength, d=1/sampleFreq)
        sliceFFTResultsAbs=np.zeros([nnumOFSlices,freqs.size])
        for i in range(nnumOFSlices):
            print("FFT "+str(i/nnumOFSlices*100)+"% done")
            tmp=(copy.deepcopy(self.interpolatedDeviationFromNominal[(fftlength*i):(fftlength+fftlength*i)])+np.arange(fftlength)*1/sampleFreq).astype(np.float128)
            simuSin=np.sin(tmp*sampleFreq*2*np.pi)
            fftresult = np.fft.rfft(simuSin)
            sliceFFTResultsAbs[i]=abs(fftresult)/(2*fftlength)

        p=ax.plot(freqs/(sampleFreq), 20*np.log10(gaussian_filter(np.mean(sliceFFTResultsAbs,axis=0),filterWidth)),
                    label=r'\textbf{'+self.title+'}', )
        if plotRaw:
            for i in range(nnumOFSlices):
                ax.plot(freqs,20*np.log10(sliceFFTResultsAbs[i]),alpha=1/nnumOFSlices,label=r'\textbf{'+self.title+'}',color=p[0].get_color())
        if plotQuanatisationNoise!=0:
            quantNoise = np.random.uniform(size=fftlength) * plotQuanatisationNoise
            quantNoise = quantNoise - plotQuanatisationNoise/2
            fft_QuantNoise = np.fft.rfft(quantNoise)
            p = ax.plot(freqs, 20 * np.log10(gaussian_filter(fft_QuantNoise, filterWidth*10)),
                        label=r'\textbf{Quantisation noise}', )
        ax.set_ylabel(r'\textbf{Phase noise in} $\frac{\mathrm{dBc}}{\mathrm{Hz}}$')
        ax.set_xlabel(r'$\frac{{Offset~frequency}}{Signal~frequency}$ \textbf{in} $\frac{\mathrm{Hz}}{\mathrm{Hz}}$')
        ax.grid(True, which="both")
        ax.legend()
        #ax.set_ylim([-220,0])
        fig.tight_layout()
        fig.show()
        return fig,ax



#dataFile = h5py.File('/home/benedikt/Downloads/jitter_recording.hfd5', 'r')
#sensorName = '0x39f50100_STM32_GPIO_Input'
sensorName = '0x60ad0100_STM32_GPIO_Input'
dataFileEXTREF = h5py.File('/home/benedikt/repos/datareceiver/extRev_single_GPS_1KHz_Edges_copy.hfd5','r')
dataFileINTREF = h5py.File('/home/benedikt/repos/datareceiver/intRev_multi_GPS_1KHz_Edges.hfd5','r')
dataFileMPU9250 = h5py.File('/home/benedikt/tmp/MPU9250PTB_v5.hdf5','r')
dataFileBMA280 = h5py.File('/home/benedikt/tmp/BMA280PTB.hdf5','r')
dataFileLSM6DSRX = h5py.File('/home/benedikt/tmp/ST_sensor_test_1667Hz_noTimeGlittCorr.hfd5','r')
dataFileLSM6DSRX6667Hz= h5py.File('/home/benedikt/tmp/ST_sensor_test_6667Hz_2.hfd5','r')
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
jitterGen1 = realWordJitterGen(dataFileINTREF, '0x39f50100_STM32_GPIO_Input',r"\textbf{Board 1 int. clock}")#nominalfreq=1000)
jitterGen2 = realWordJitterGen(dataFileINTREF, '0x60ad0100_STM32_GPIO_Input',"Board 2 int. clock 1000 Hz")#nominalfreq=1000)
jitterGen3 = realWordJitterGen(dataFileEXTREF, '0x39f50100_STM32_GPIO_Input',r"\textbf{Board 1 ext. clock")# 1000 Hz}")#nominalfreq=1000)
jitterGen4 = realWordJitterGen(dataFileEXTREF, '0x60ad0100_STM32_GPIO_Input',"Board 2 ext. clock")#nominalfreq=1000)
jitterGenMPU9250 = realWordJitterGen(dataFileMPU9250,'0x1fe40000_MPU_9250',r"\textbf{MPU9250}")# $f_s=$ \textbf{1001.0388019191 Hz}")
jitterGenBMA280= realWordJitterGen(dataFileBMA280,'0x1fe40000_BMA_280',     r"\textbf{BMA280}")# $f_s=$ \textbf{2064.9499858147 Hz} ",)#offset=[100000,1560000+13440562+20])
jitterGenLSM6DSRX=realWordJitterGen(dataFileLSM6DSRX,'0x60ad0000_LSM6DSRX',r"\textbf{LSM6DSRX $f_s$=1.667~kHz}")
jitterGenLSM6DSRX6667Hz=realWordJitterGen(dataFileLSM6DSRX6667Hz,'0x60ad0000_LSM6DSRX',r"\textbf{LSM6DSRX $f_s$=6.667~kHz}")

jitterGensForSimulations=[jitterGenBMA280,jitterGenMPU9250,jitterGen3,jitterGenLSM6DSRX,jitterGenLSM6DSRX6667Hz]#jitterGen1,jitterGen3
def generateFitWithPhaseNoise(freq,fs=1000,t_jitter=100e-9,lengthInS=0.1,jitterGens=jitterGensForSimulations,A0=1,phi0=0,linearFreqCorrection=True):
    #TODO change interface
    if t_jitter <= 0:
        fs=jitterGens[int(-1*t_jitter)].fs
    originalTimpoints=np.linspace(0,lengthInS,num=int(fs*lengthInS))
    Signal=A0*np.sin(originalTimpoints*np.pi*2*freq+phi0)
    if t_jitter >0:
        jitter=np.random.normal(scale=t_jitter, size=Signal.size)
    else:
        jitter=jitterGens[int(-1*t_jitter)].getrandomDeviations(Signal.size)
    if linearFreqCorrection:
        idx=np.arange(originalTimpoints.size)
        m, b = np.polyfit(idx, jitter.astype(np.float64), 1)
        timeWJitter=originalTimpoints+jitter-(idx*m+b)
    else:
        timeWJitter = originalTimpoints + jitter
    fitparams=st.threeparsinefit(Signal,timeWJitter,freq)
    return st.phase(fitparams)-phi0,st.amplitude(fitparams)/A0

def getmuAndSTdForFreq(testparams,numOfruns=1000):
    freq=testparams[0]
    t_jitter=testparams[1]
    length = testparams[2]
    FitPhases=np.zeros(numOfruns)
    FitMags=np.zeros(numOfruns)
    for i in range(numOfruns):
        FitPhases[i],FitMags[i]=generateFitWithPhaseNoise(freq,t_jitter=t_jitter,lengthInS=length)
    ampPercentiles=np.percentile(FitMags, np.array([5,32,50,68,95]))
    phasePercentiles=np.percentile(FitPhases, np.array([5,32,50,68,95]))
    return np.std(FitPhases),\
           np.mean(FitPhases),\
           np.std(FitMags),\
           np.mean(FitMags),\
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



if __name__ == "__main__":
    deviationPlotlength=10
    figDviation,axDeviation=jitterGen1.plotDeviation(lengthInS=deviationPlotlength,lw=1)
    """
    jitterGen2.plotDeviation(fig=figDviation,ax=axDeviation,length=150000,lw=1)
    
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
    """
    jitterGenMPU9250.plotDeviation(fig=figDviation, axs=axDeviation,lengthInS=deviationPlotlength)
    jitterGenBMA280.plotDeviation(fig=figDviation, axs=axDeviation,lengthInS=deviationPlotlength, show=True)
    jitterGenLSM6DSRX.plotDeviation(fig=figDviation, axs=axDeviation,lengthInS=deviationPlotlength, show=True)
    jitterGenLSM6DSRX6667Hz.plotDeviation(fig=figDviation, axs=axDeviation,lengthInS=deviationPlotlength, show=True)

    figPhaseNoise,axPhaseNoise=jitterGen1.plotPhaseNoise(plotRaw=False)
    #jitterGen4.plotPhaseNoise(fig=figPhaseNoise,ax=axPhaseNoise,plotRaw=False)
    jitterGenMPU9250.plotPhaseNoise(fig=figPhaseNoise,ax=axPhaseNoise,plotRaw=False)
    jitterGenBMA280.plotPhaseNoise(fig=figPhaseNoise, ax=axPhaseNoise, plotRaw=False)
    jitterGenLSM6DSRX.plotPhaseNoise(fig=figPhaseNoise, ax=axPhaseNoise, plotRaw=False)
    jitterGenLSM6DSRX6667Hz.plotPhaseNoise(fig=figPhaseNoise, ax=axPhaseNoise, plotRaw=False)
    """
    """
    #jitterGen1.plotAkf()
    #jitterGen.plotFFT()
    #jitterGen1.plotFFT(plotPhase=False)

    #jitterGen1.plotPhaseNoise(plotRaw=False)
    freqPoints=50
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

    results=process_map(getmuAndSTdForFreq, testparams, max_workers=15,chunksize=1)
    results=np.array(results)
    bw=np.ones(SimuPoints)


    fig1, ax = plt.subplots()
    fig1.set_figwidth(12)
    fig1.set_figheight(4)
    #fig1.suptitle(r"\textbf{Samplerate = 1 kHz ; duration = " + str(lengthInS) + ' s}')
    doFit=False
    plotErrors=True
    for i in range(SimuPoints):
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
        if plotErrors:
            dataPlot=ax.plot(tmpFreqs,
                   AMPS,
                   label=r"\textbf{Median }"+label,lw=PLTSCALFACTOR*2)
            errorPlot2 = ax.fill_between(tmpFreqs,
                               AMPSErrorBottom,
                                AMPSErrorTop,
                                label=r"\textbf{32\% -68\% coverage }" + str(label),
                                color=dataPlot[0].get_color(),
                                alpha=0.3,
                                hatch = 'O')
            errorPlot1 = ax.fill_between(tmpFreqs,
                               AMPSError25Bottom,
                                AMPSError75Top,
                                label=r"\textbf{5\% -95\% coverage }"+str(label),
                                color=dataPlot[0].get_color(),
                                alpha=0.3,
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
    ax.legend(ncol=3)
    ax.set_xlabel(r"\textbf{Simulated signal frequency in Hz}")
    #ax[0].set_ylabel(r"$2\sigma(\hat{A})$ \textbf{in \%}")
    ax.set_ylabel(r"$\frac{\hat{A}}{A_{nom}}$")
    #ax[0].grid(True)
    ax.grid(True)
    fig1.tight_layout()
    fig1.show()


    #fig, ax = plt.subplots(2,sharex=True)
    fig4, ax4 = plt.subplots(1)
    fig4.set_figwidth(12)
    fig4.set_figheight(4)
    #fig4.suptitle(r"\textbf{Samplerate = 1 kHz ; duration = " + str(lengthInS) + ' s}')
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
    ax4.set_xlabel(r"\textbf{Simulated signal frequency in Hz}")
    ax4.set_ylabel(r"\textbf{Max. phase deviation $2\sigma \varphi$ in} $^\circ$")
    #ax[1].set_ylabel(r"$\overline{\varphi}-\varphi_{soll}$ \textbf{in} $^\circ$")
    ax4.grid(True)
    #ax[1].grid(True)
    fig4.tight_layout()
    fig4.show()

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
    results=process_map(getmuAndSTdForFreq, testparams, max_workers=7)
    results=np.array(results)


    fig2,ax=plt.subplots(2,sharex=True)
    fig2.set_figwidth(10)
    fig2.set_figheight(5)
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
    fig3.set_figwidth(10)
    fig3.set_figheight(5)
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
    ax[1].set_ylabel(r"$\frac{\overline{\hat{A}}}{A_{soll}}$ \textbf{in R.U.}")
    ax[0].grid(True)
    ax[1].grid(True)
    fig3.tight_layout()
    fig3.show()
    """
    print("Hello")
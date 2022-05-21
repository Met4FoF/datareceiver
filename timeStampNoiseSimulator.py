import numpy as np
import matplotlib.pyplot as plt
import sinetools.SineTools as st
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.signal import correlate
from scipy.signal import correlation_lags
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

    def __init__(self,HDfFile,sensorName,title,nominalfreq=1000,offset=[40000,10000]):
        self.datafile=HDfFile
        self.title=title
        self.Dataset=self.datafile['RAWDATA/'+sensorName+'/Absolutetime']
        self.dataPoints=self.datafile['RAWDATA/'+sensorName].attrs['Data_point_number']# use only valide points
        self.AbsoluteTime = self.Dataset[0, 0 + offset[0]:self.dataPoints - offset[1]] - self.Dataset[0, offset[0]]
        self.timeData=(self.AbsoluteTime-self.AbsoluteTime[0])/1e9
        #self.timeData=(self.Dataset[0,0+offset[0]:self.dataPoints-offset[1]]-self.Dataset[0,offset[0]])/1e9# substract first point to avoid precisionlos with f64 the divide by 1e9 to ahve seconds
        self.relSampleNumber = (self.datafile['RAWDATA/'+sensorName+'/Sample_number'][0, 0 + offset[0]:self.dataPoints-offset[1]] - self.datafile['RAWDATA/'+sensorName+'/Sample_number'][0, offset[0]])

        self.length=self.timeData[-1]-self.timeData[0] #calculate length of dataSet
        if nominalfreq==0:
            self.fs = self.dataPoints / (self.length)  # calculate smaple freq
        else:
            self.fs = nominalfreq
        #self.deltaT=self.length/(self.dataPoints-1)
        self.deltaT = 1/self.fs
        print("Sample frequenc is "+str(self.fs)+' Hz')
        self.expectedTime=self.relSampleNumber*self.deltaT
        self.deviationFromNominal=self.timeData-self.expectedTime#calulate deviation from Expected Mean
        self.meandeviationFromNominal=np.mean(self.deviationFromNominal)
        self.std = np.std(self.deviationFromNominal)
        # INTERPOLATE MISSING DATA WITH NEAREST NIGBOUR
        self.interpolatedDeviationFromNominal=np.ones(self.relSampleNumber[-1])*self.meandeviationFromNominal

        diff=np.diff(self.relSampleNumber)
        jumpIDX=np.array(np.argwhere(diff>1))
        #jumpIDX=self.relSampleNumber[jumpIDX]
        jumpIDX = np.insert(jumpIDX, 0, 0)
        offset=0
        for i in range(jumpIDX.size-1):
            startIDX=jumpIDX[i]
            stopIDX=jumpIDX[i+1]
            tmp=np.copy(self.deviationFromNominal[startIDX:stopIDX])
            self.interpolatedDeviationFromNominal[(startIDX+offset):(stopIDX+offset)]=tmp
            offset+=diff[stopIDX]
            print(offset)
            del tmp
        print("Test")



    def plotDeviation(self,fig=None,ax=None,lenghth=None,show=False):
        if fig==None and ax==None:
            fig,ax=plt.subplots()
        if lenghth ==None:
            ax.plot(self.timeData,self.deviationFromNominal*1e9,label=self.title)
        else:
            ax.plot(self.timeData[:lenghth], self.deviationFromNominal[:lenghth]*1e9, label=self.title)
        #ax.plot(np.arange(self.interpolatedDeviationFromNominal.size)*self.deltaT,self.interpolatedDeviationFromNominal)
        if show:
            ax.set_xlabel(r"\textbf{Relative time in s}")
            ax.set_ylabel(r"\textbf{Time deviation in ns}")
            ax.legend()
            ax.grid()
            fig.show()
        return fig,ax
    def getrandomDeviations(self,length,reytryes=1000):
        isContinousDataSliceRetryCount=0
        while isContinousDataSliceRetryCount<reytryes:
            idx=np.random.randint(self.deviationFromNominal.size-(length+1))
            if self.relSampleNumber[idx+length]-self.relSampleNumber[idx]==length:
                break
            else:
                isContinousDataSliceRetryCount+=1

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
        fig.show()

    def plotPhaseNoise(self,sampleFreq=1000,fftlength=1048576*16,plotRaw=True,fig=None,ax=None):
        def abs2(x):
            return x.real ** 2 + x.imag ** 2
        nnumOFSlices=int(np.floor(self.interpolatedDeviationFromNominal.shape[0]/fftlength))
        if fig==None and ax==None:
            fig, ax = plt.subplots()
        ax.set_yscale('log')
        freqs = np.fft.rfftfreq(fftlength, d=1/sampleFreq)
        sliceFFTResultsAbsSqared=np.zeros([nnumOFSlices,freqs.size])
        for i in range(nnumOFSlices):
            print("FFT "+str(i/nnumOFSlices*100)+"% done")
            tmp=copy.deepcopy(self.interpolatedDeviationFromNominal[(fftlength*i):(fftlength+fftlength*i)])+np.arange(fftlength)*1/sampleFreq
            simuSin=np.sin(tmp*sampleFreq*2*np.pi)
            fftresult = np.fft.rfft(simuSin)
            sliceFFTResultsAbsSqared[i]=abs2(fftresult)/fftlength

        p=ax.plot(freqs, np.mean(sliceFFTResultsAbsSqared,axis=0),
                    label=r'\textbf{Mean '+self.title+'}', )
        if plotRaw:
            for i in range(nnumOFSlices):
                ax.plot(freqs,sliceFFTResultsAbsSqared[i],alpha=1/nnumOFSlices,label=r'\textbf{'+self.title+'}',color=p[0].get_color())
        ax.set_ylabel(r'\textbf{Phasenoise in dbc}')
        ax.set_xlabel(r'\textbf{Offset frequency in Hz}')
        ax.grid(True, which="both")
        ax.legend()
        fig.show()
        return fig,ax



#dataFile = h5py.File('/home/benedikt/Downloads/jitter_recording.hfd5', 'r')
#sensorName = '0x39f50100_STM32_GPIO_Input'
sensorName = '0x60ad0100_STM32_GPIO_Input'
dataFileEXTREF = h5py.File('/home/benedikt/repos/datareceiver/extRev_single_GPS_1KHz_Edges_copy.hfd5','r')
dataFileINTREF = h5py.File('/home/benedikt/repos/datareceiver/intRev_multi_GPS_1KHz_Edges.hfd5','r')
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
jitterGen1 = realWordJitterGen(dataFileINTREF, '0x39f50100_STM32_GPIO_Input',"Board 1 multi GNSS")#nominalfreq=1000)
jitterGen2 = realWordJitterGen(dataFileINTREF, '0x60ad0100_STM32_GPIO_Input',"Board 2 multi GNSS")#nominalfreq=1000)
jitterGen3 = realWordJitterGen(dataFileEXTREF, '0x39f50100_STM32_GPIO_Input',"Board 1 single GNSS ext. clock")#nominalfreq=1000)
jitterGen4 = realWordJitterGen(dataFileEXTREF, '0x60ad0100_STM32_GPIO_Input',"Board 2 single GNSS ext. clock")#nominalfreq=1000)
jitterGensForSimulations=[jitterGen1,jitterGen3]
def generateFitWithPhaseNoise(freq,fs=1000,t_jitter=100e-9,lengthInS=0.1,jitterGens=jitterGensForSimulations,A0=1,phi0=0):
    originalTimpoints=np.linspace(0,lengthInS,num=int(fs*lengthInS))
    Signal=A0*np.sin(originalTimpoints*np.pi*2*freq+phi0)
    if t_jitter >0:
        jitter=np.random.normal(scale=t_jitter, size=Signal.size)
    else:
        jitter=jitterGens[int(-1*t_jitter)].getrandomDeviations(Signal.size)
    timeWJitter=originalTimpoints+jitter
    fitparams=st.threeparsinefit(Signal,timeWJitter,freq)
    return st.phase(fitparams)-phi0,st.amplitude(fitparams)/A0

def getmuAndSTdForFreq(testparams,numOfruns=100):
    freq=testparams[0]
    t_jitter=testparams[1]
    length = testparams[2]
    FitPhases=np.zeros(numOfruns)
    FitMags=np.zeros(numOfruns)
    for i in range(numOfruns):
        FitPhases[i],FitMags[i]=generateFitWithPhaseNoise(freq,t_jitter=t_jitter,lengthInS=length)
    return np.std(FitPhases),np.mean(FitPhases),np.std(FitMags),np.mean(FitMags)



if __name__ == "__main__":
    """
    figDviation,axDeviation=jitterGen1.plotDeviation(lenghth=150000)
    jitterGen2.plotDeviation(fig=figDviation,ax=axDeviation,lenghth=150000)
    shorterDsetLength=150000#np.min([jitterGen1.dataPoints,jitterGen2.dataPoints])-1024

    axDeviation.plot(jitterGen1.expectedTime[:shorterDsetLength], (
                jitterGen1.AbsoluteTime[:shorterDsetLength].astype(np.int64) - jitterGen2.AbsoluteTime[
                                                                               :shorterDsetLength].astype(np.int64)),
                     label="Time difference multi GNSS int. clock")
    axDeviation.plot(jitterGen3.expectedTime[:shorterDsetLength], (
                jitterGen3.AbsoluteTime[:shorterDsetLength].astype(np.int64) - jitterGen4.AbsoluteTime[
                                                                               :shorterDsetLength].astype(np.int64)),
                     label="Time difference single GNSS ext. clock")
    jitterGen3.plotDeviation(fig=figDviation,ax=axDeviation,lenghth=150000)
    jitterGen4.plotDeviation(fig=figDviation, ax=axDeviation,lenghth=150000,show=True)
    
    figPhaseNoise,axPhaseNoise=jitterGen1.plotPhaseNoise(plotRaw=False)
    jitterGen4.plotPhaseNoise(fig=figPhaseNoise,ax=axPhaseNoise,plotRaw=False)
    jitterGen1.plotAkf()
    #jitterGen.plotFFT()
    jitterGen1.plotFFT(plotPhase=False)
    """
    freqPoints=100
    ampPoints=8
    nsPreAmpStep=20
    lengthInS=10
    freqs=np.zeros(freqPoints*ampPoints)
    noiseLevel=np.zeros(freqPoints*ampPoints)
    runNoiselevel=np.append(np.array([-1,0]),np.array(np.arange(ampPoints-2)+1)*nsPreAmpStep*10e-9)
    for i in range(ampPoints):
        tmpFreqs=np.logspace(1.00,7.0,freqPoints)
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
    bw=np.ones(ampPoints)


    fig1,ax=plt.subplots(2,sharex=True)
    fig1, ax = plt.subplots()
    fig1.suptitle(r"\textbf{Samplerate = 1 kHz ; duration = " + str(lengthInS) + ' s}')
    for i in range(ampPoints):
        tmpFreqs=freqs[i * freqPoints: (i + 1) * freqPoints]
        if i==0:
            label = r"\textbf{int. clock $2\sigma=" + str(2 * int(jitterGensForSimulations[1].std * 1e9)) + "$ ns}"

        elif i==1:
            label = r"\textbf{ext. clock $2\sigma=" + str(2 * int(jitterGensForSimulations[0].std * 1e9)) + "$ ns}"
        else:
            label=r"\textbf{\textit{simu.} $2\sigma= " + str(2*((i-1) * nsPreAmpStep)) + "$ ns}"
        #ax[0].plot(tmpFreqs,
        #           200*results[i * freqPoints: (i + 1) * freqPoints,2],
        #           label=label)
        AMPS=results[i * freqPoints: (i + 1) * freqPoints, 3]
        dataPlot=ax.plot(tmpFreqs,
                   AMPS,
                   label=label)

        popt, pcov = curve_fit(gaus, tmpFreqs, AMPS, p0=[1, 5e5])
        ax.plot(tmpFreqs,gaus(tmpFreqs,popt[0],popt[1]),label=r"\textbf{Fited bandwidth= }"+"{:.2f}".format(abs(popt[1])/1e6)+" MHz",color=dataPlot[-1].get_color(),ls='--')
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
    ax.set_ylabel(r"$\frac{\overline{\hat{A}}}{A_{nom}}$ \textbf{in R.U.}")
    #ax[0].grid(True)
    ax.grid(True)
    fig1.show()

    """
    #fig, ax = plt.subplots(2,sharex=True)
    fig, ax = plt.subplots(1)
    fig.suptitle(r"\textbf{Samplerate = 1 kHz ; duration = " + str(lengthInS) + ' s}')
    for i in range(ampPoints):
        if i==0:
            label = r"\textbf{int. clock $2\sigma=" + str(2 * int(jitterGensForSimulations[1].std * 1e9)) + "$ ns}"

        elif i==1:
            label = r"\textbf{ext. clock $2\sigma=" + str(2 * int(jitterGensForSimulations[0].std * 1e9)) + "$ ns}"
        else:
            label=r"\textbf{\textit{simu.} $2\sigma= " + str(2*((i-1) * nsPreAmpStep)) + "$ ns}"
        tmpFreqs = freqs[i * freqPoints: (i + 1) * freqPoints]
        sigmaPhase = results[i * freqPoints: (i + 1) * freqPoints, 0]
        dataPlot = ax.plot(tmpFreqs,
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
        
    ax.legend(ncol=4)
    #ax[1].legend(ncol=4)
    ax.set_xlabel(r"\textbf{Simulated signal frequency in Hz}")
    ax.set_ylabel(r"\textbf{Worst case phase deviation $2\sigma \varphi$ in} $^\circ$")
    #ax[1].set_ylabel(r"$\overline{\varphi}-\varphi_{soll}$ \textbf{in} $^\circ$")
    ax.grid(True)
    #ax[1].grid(True)
    fig.show()
    """
    """
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    surf=ax2.plot_trisurf(freqs, noiseLevel, results[:, 0],cmap=cm.coolwarm)
    fig2.colorbar(surf, shrink=0.5, aspect=5)
    fig2.show()
    """
    """
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
    fig2.suptitle(r"\textbf{SampleRate = 1 kHz | 100 ns  \textit{Jitter}}")
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

    fig2.show()

    fig3,ax=plt.subplots(2,sharex=True)
    fig3.suptitle(r"\textbf{SampleRate = 1 kHz | 100 ns \textit{Jitter}}")
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
    fig3.show()
    """
print("Hello")
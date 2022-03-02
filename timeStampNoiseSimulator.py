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
PLTSCALFACTOR = 2
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

    def __init__(self,HDfFile,sensorName):
        self.datafile=HDfFile
        self.Dataset=self.datafile['RAWDATA/'+sensorName+'/Absolutetime']
        self.dataPoints=self.datafile['RAWDATA/'+sensorName].attrs['Data_point_number']# use only valide points
        self.timeData=(self.Dataset[0,0:self.dataPoints]-self.Dataset[0,0])/1e9# substract first point to avoid precisionlos with f64 the divide by 1e9 to ahve seconds
        self.length=self.timeData[-1]-self.timeData[0] #calculate length of dataSet
        self.fs=self.dataPoints/(self.length) #calculate smaple freq
        self.deltaT=self.length/(self.dataPoints-1)
        print("Sample frequenc is "+str(self.fs)+' Hz')
        self.expectedTime=np.arange(self.dataPoints)*self.deltaT
        self.deviationFromNominal=self.timeData-self.expectedTime#calulate deviation from Expected Mean

    def plotDeviation(self):
        fig,ax=plt.subplots()
        ax.plot(self.timeData,self.deviationFromNominal)
        fig.show()

    def getrandomDeviations(self,length):
        idx=np.random.randint(self.dataPoints-(length+1))
        return self.deviationFromNominal[idx:idx+length]

    def plotAkf(self,sampleFreq=1000):
        fig,ax=plt.subplots()
        tmp=copy.deepcopy(self.deviationFromNominal[:4194304])
        akf = correlate(tmp, tmp, mode='full')
        akf =akf/np.max(akf)

        gausnoise=np.random.normal(scale=43e-9,size=tmp.size)
        gausnoise=gausnoise/np.max(gausnoise)
        akf_gausNoise = correlate (gausnoise,gausnoise, mode='full')
        akf_gausNoise=akf_gausNoise/np.max(akf_gausNoise)
        deltaT=1/sampleFreq
        lag=correlation_lags(tmp.size,tmp.size)*deltaT
        ax.plot(lag,akf,label=r'\textbf{\textbf{Aufgeizeichneter \textit{Jitter}}}')
        ax.plot(lag, akf_gausNoise, label=r'\textbf{Nomalverteilter Jitter} $\sigma = 43~\mathrm{ns}$')
        ax.set_xlabel(r'\textbf{Zeitverschiebungen} $\tau$ \textbf{in } s')
        ax.set_ylabel(r'\textbf{Autokorrelations Funktion} $AKF$ \textbf{in } R.U s')
        ax.grid()
        fig.show()

    def plotFFT(self,sampleFreq=1000,plotPhase=True,fftlength=262144):
        def abs2(x):
            return x.real ** 2 + x.imag ** 2
        nnumOFSlices=int(np.floor(self.deviationFromNominal.shape[0]/262144))
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
        gausnoise=np.random.uniform(size=fftlength)*(43)
        gausnoise=gausnoise-(43)/2
        fft_gausNoise = np.fft.rfft(gausnoise)
        fft_QuantNoise = np.fft.rfft(quantNoise)
        sliceFFTResultsAbsSqared=np.zeros([nnumOFSlices,freqs.size])
        for i in range(nnumOFSlices):
            tmp=copy.deepcopy(self.deviationFromNominal[(fftlength*i):(fftlength+fftlength*i)])*1e9
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




dataFile = h5py.File('/home/benedikt/Downloads/jitter_recording.hfd5', 'r')
sensorName = '0xe0040100_BMA_280'
jitterGen = realWordJitterGen(dataFile, sensorName)

def generateFitWithPhaseNoise(freq,fs=1000,t_jitter=100e-9,lengthInS=0.1,jitterGen=jitterGen,A0=1,phi0=0):
    originalTimpoints=np.linspace(0,lengthInS,num=int(fs*lengthInS))
    Signal=A0*np.sin(originalTimpoints*np.pi*2*freq+phi0)
    if t_jitter >=0:
        jitter=np.random.normal(scale=t_jitter, size=Signal.size)
    else:
        jitter=jitterGen.getrandomDeviations(Signal.size)
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

    #jitterGen.plotDeviation()
    jitterGen.plotAkf()
    #jitterGen.plotFFT()
    #jitterGen.plotFFT(plotPhase=False)
    freqPoints=200
    ampPoints=8
    nsPreAmpStep=20
    lengthInS=10
    freqs=np.zeros(freqPoints*ampPoints)
    noiseLevel=np.zeros(freqPoints*ampPoints)
    for i in range(ampPoints):
        tmpFreqs=np.logspace(1.00,4.0,freqPoints)
        freqToNear=(tmpFreqs % 1000) < 5
        freqToNear+=(tmpFreqs % 500) < 5
        freqToAdd=10*freqToNear
        tmpFreqs+=freqToAdd
        tmpNoiseLevel=np.ones(freqPoints)*(i-1)*nsPreAmpStep*10e-9
        freqs[i*freqPoints:(i+1)*freqPoints]=tmpFreqs
        noiseLevel[i * freqPoints:(i + 1) * freqPoints] = tmpNoiseLevel
    length=np.ones(freqs.size)*lengthInS
    testparams=np.array([freqs,noiseLevel,length]).transpose()

    results=process_map(getmuAndSTdForFreq, testparams, max_workers=7)
    results=np.array(results)
    bw=np.ones(ampPoints)
    fig1,ax=plt.subplots(2,sharex=True)
    fig1.suptitle(r"\textbf{SampleRate = 1 kHz Dauer = "+str(lengthInS)+' s}')
    for i in range(ampPoints):
        tmpFreqs=freqs[i * freqPoints: (i + 1) * freqPoints]
        if i==0:
            label=r"\textbf{Realer \textit{Jitter}}"
        else:
            label=r"\textbf{\textit{Jitter} = " + str((i-1) * nsPreAmpStep) + " ns}"
        ax[0].plot(tmpFreqs,
                   200*results[i * freqPoints: (i + 1) * freqPoints,2],
                   label=label)
        AMPS=results[i * freqPoints: (i + 1) * freqPoints, 3]
        dataPlot=ax[1].plot(tmpFreqs,
                   AMPS,
                   label=label)
        if i!=1:
            popt, pcov = curve_fit(gaus, tmpFreqs, AMPS, p0=[1, 5e5])
            ax[1].plot(tmpFreqs,gaus(tmpFreqs,popt[0],popt[1]),label=r"\textbf{ Fit Bandbreite = }"+"{:.2f}".format(abs(popt[1])/1e6)+" MHz",color=dataPlot[-1].get_color(),ls='--')
            bw[i]=popt[1]
            print('______'+str(i * nsPreAmpStep)+' ns ___________')
            print(popt)
            print(popt[1]/(i * nsPreAmpStep*10e-9)*(i * nsPreAmpStep*10e-9))
            print('_____________________________________________')
    ax[0].legend()
    ax[0].legend(ncol=4)
    ax[1].legend(ncol=4)
    ax[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax[0].set_ylabel(r"$2\sigma(\hat{A})$ \textbf{in \%}")
    ax[1].set_ylabel(r"$\frac{\overline{\hat{A}}}{A_{soll}}$ \textbf{in R.U.}")
    ax[0].grid(True)
    ax[1].grid(True)
    fig1.show()

    fig, ax = plt.subplots(2,sharex=True)
    fig.suptitle(r"\textbf{SampleRate = 1 kHz Dauer = " + str(lengthInS) + ' s}')
    for i in range(ampPoints):
        if i==0:
            label=r"\textbf{Realer \textit{Jitter}}"
        else:
            label=r"\textbf{\textit{Jitter} = " + str((i-1) * nsPreAmpStep) + " ns}"
        tmpFreqs = freqs[i * freqPoints: (i + 1) * freqPoints]
        sigmaPhase = results[i * freqPoints: (i + 1) * freqPoints, 0]
        dataPlot = ax[0].plot(tmpFreqs,
                              2 * sigmaPhase / np.pi * 180,
                              label=label)
        ax[1].plot(tmpFreqs,
                   results[i * freqPoints: (i + 1) * freqPoints, 1] / np.pi * 180,
                   label=label)
        
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
        
    ax[0].legend(ncol=4)
    ax[1].legend(ncol=4)
    ax[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax[0].set_ylabel(r"$2\sigma \varphi$ \textbf{in} $^\circ$")
    ax[1].set_ylabel(r"$\overline{\varphi}-\varphi_{soll}$ \textbf{in} $^\circ$")
    ax[0].grid(True)
    ax[1].grid(True)
    fig.show()
    
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
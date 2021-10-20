import numpy as np
import matplotlib.pyplot as plt
import sinetools.SineTools as st
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
from matplotlib import cm
from scipy.optimize import curve_fit

def gaus(x,a,sigma):
    x0=0
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

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

def generateFitWithPhaseNoise(freq,fs=1000,t_jitter=100e-9,lengthInS=0.1):
    originalTimpoints=np.linspace(0,lengthInS,num=int(fs*lengthInS))
    Signal=np.sin(originalTimpoints*np.pi*2*freq)
    timeWJitter=originalTimpoints+np.random.normal(scale=t_jitter,size=Signal.size)
    fitparams=st.threeparsinefit(Signal,timeWJitter,freq)
    return st.phase(fitparams),st.amplitude(fitparams)

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

    freqPoints=200
    ampPoints=12
    nsPreAmpStep=10
    lengthInS=10
    freqs=np.zeros(freqPoints*ampPoints)
    noiseLevel=np.zeros(freqPoints*ampPoints)
    for i in range(ampPoints):
        tmpFreqs=np.logspace(1.00,3.3,freqPoints)
        freqToNear=(tmpFreqs % 1000) < 5
        freqToNear+=(tmpFreqs % 500) < 5
        freqToAdd=10*freqToNear
        tmpFreqs+=freqToAdd
        tmpNoiseLevel=np.ones(freqPoints)*i*nsPreAmpStep*10e-9
        freqs[i*freqPoints:(i+1)*freqPoints]=tmpFreqs
        noiseLevel[i * freqPoints:(i + 1) * freqPoints] = tmpNoiseLevel
    length=np.ones(freqs.size)*lengthInS
    testparams=np.array([freqs,noiseLevel,length]).transpose()
    results=process_map(getmuAndSTdForFreq, testparams, max_workers=7)
    results=np.array(results)
    fig,ax=plt.subplots(2)
    fig.suptitle(r"\textbf{SampleRate = 1 kHz Dauer = "+str(lengthInS)+' s}')
    for i in range(ampPoints):
        ax[0].plot(freqs[i * freqPoints: (i + 1) * freqPoints],
                   2*results[i * freqPoints: (i + 1) * freqPoints,0]/np.pi*180,
                   label=r"\textbf{jitter= "+str(i*nsPreAmpStep)+" ns}")
        ax[1].plot(freqs[i * freqPoints: (i + 1) * freqPoints],
                   results[i * freqPoints: (i + 1) * freqPoints, 1] / np.pi * 180,
                   label=r"\textbf{jitter= " + str(i * nsPreAmpStep) + " ns}")
    ax[0].legend(ncol=4)
    ax[1].legend(ncol=4)
    ax[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax[0].set_ylabel(r"$2\sigma \varphi$ \textbf{in} $^\circ$")
    ax[1].set_ylabel(r"$\overline{\varphi}$ \textbf{in} $^\circ$")
    ax[0].grid(True)
    ax[1].grid(True)

    fig.show()
    bw=np.ones(ampPoints)
    fig1,ax=plt.subplots(2)
    fig1.suptitle(r"\textbf{SampleRate = 1 kHz Dauer = "+str(lengthInS)+' s}')
    for i in range(ampPoints):
        tmpFreqs=freqs[i * freqPoints: (i + 1) * freqPoints]
        ax[0].plot(tmpFreqs,
                   200*results[i * freqPoints: (i + 1) * freqPoints,2],
                   label=r"\textbf{Jitter = "+str(i*nsPreAmpStep)+" ns}")
        AMPS=results[i * freqPoints: (i + 1) * freqPoints, 3]
        dataPlot=ax[1].plot(tmpFreqs,
                   AMPS,
                   label=r"\textbf{Jitter = " + str(i * nsPreAmpStep) + " ns}")
        if i!=0:
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
    ax[1].set_ylabel(r"$\overline{\hat{A}}$ \textbf{in R.U.}")
    ax[0].grid(True)
    ax[1].grid(True)

    fig1.show()

    """
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    surf=ax2.plot_trisurf(freqs, noiseLevel, results[:, 0],cmap=cm.coolwarm)
    fig2.colorbar(surf, shrink=0.5, aspect=5)
    fig2.show()
    """
    lengthPoints=15
    StartLength=64
    noiseLevelToUse=100*10e-9
    freqs=np.zeros(freqPoints*lengthPoints)
    noiseLevel=np.zeros(freqPoints*lengthPoints)
    length = np.ones(freqs.size)

    for i in range(lengthPoints):
        tmpFreqs=np.logspace(1.00,6.0,freqPoints)
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
    fig2,ax=plt.subplots(2)
    fig2.suptitle(r"\textbf{SampleRate = 1 kHz Jitter = 100 ns}")
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
    ax[1].set_ylabel(r"$\overline{\varphi}$ \textbf{in} $^\circ$")
    ax[0].grid(True)
    ax[1].grid(True)

    fig2.show()

    fig3,ax=plt.subplots(2)
    fig3.suptitle(r"\textbf{SampleRate = 1 kHz Jitter = 100 ns}")
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
    ax[1].set_ylabel(r"$\overline{\hat{A}}$ \textbf{in A.U.}")
    ax[0].grid(True)
    ax[1].grid(True)
    fig3.show()
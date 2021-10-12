import numpy as np
import matplotlib.pyplot as plt
import sinetools.SineTools as st
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

def generateFitWithPhaseNoise(freq,fs=1000,t_jitter=100e-9,lengthInS=30):
    originalTimpoints=np.linspace(0,lengthInS,num=fs*lengthInS)
    Signal=np.sin(originalTimpoints*np.pi*2*freq)
    timeWJitter=originalTimpoints+np.random.normal(scale=t_jitter,size=Signal.size)
    fitparams=st.threeparsinefit(Signal,timeWJitter,freq)
    return st.phase(fitparams)

def getmuAndSTdForFreq(testparams,numOfruns=100):
    freq=testparams[0]
    t_jitter=testparams[1]
    FitPhases=np.zeros(numOfruns)
    for i in range(numOfruns):
        FitPhases[i]=generateFitWithPhaseNoise(freq,t_jitter=t_jitter)
    return np.std(FitPhases),np.mean(FitPhases)

freqPoints=100
ampPoints=10
freqs=np.zeros(freqPoints*ampPoints)
noiseLevel=np.zeros(freqPoints*ampPoints)
for i in range(ampPoints):
    tmpFreqs=np.logspace(1.00,5.00,freqPoints)
    freqToNear=(tmpFreqs % 1000) < 5
    freqToNear+=(tmpFreqs % 500) < 5
    freqToAdd=10*freqToNear
    tmpFreqs+=freqToAdd
    tmpNoiseLevel=np.ones(freqPoints)*i*10e-9
    freqs[i*freqPoints:(i+1)*freqPoints]=tmpFreqs
    noiseLevel[i * freqPoints:(i + 1) * freqPoints] = tmpNoiseLevel
testparams=np.array([freqs,noiseLevel]).transpose()
results=process_map(getmuAndSTdForFreq, testparams, max_workers=4)
results=np.array(results)
plt.plot(freqs,results[:,0]/np.pi*180)
plt.plot(freqs,results[:,1]/np.pi*180)

import h5py as h5py
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
PLTSCALFACTOR = 1.5
SMALL_SIZE = 12 * PLTSCALFACTOR
MEDIUM_SIZE = 15 * PLTSCALFACTOR
BIGGER_SIZE = 18 * PLTSCALFACTOR

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plotHist(data,bins):
    mu, std = norm.fit(data)

    # Plot the histogram.
    fig,ax=plt.subplots()
    fig.set_size_inches(11.326, 7, forward=True)
    ax.hist(data, bins=bins, density=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, bins)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k')
    title = r'\textbf{Abweichung zwischen den zwei SmartUp-Units}\\ $\mu$='+"{:2.2f}".format(mu)+'~ns  $\sigma=$'+"{:2.2f}".format(std)+'~ns'
    ax.grid()
    ax.set_xlabel(r"\textbf{Differenz der Zeitstempel in ns}")
    ax.set_ylabel(r"\textbf{Releative Häufigkeit}")
    fig.suptitle(title)
    fig.savefig("timinghistogramm.svg")
    fig.savefig("timinghistogramm.png")
    fig.show()

def plottimediffWithUncerAndZoom(data,uncer,period=(1e8+0.0005950575459905795),zoomArea=(20000,22000)):
    timeMinusExpectedTime1 = data.astype(np.int64)-data[0].astype(np.int64) - (period * np.arange(data.size))+20
    reltime=(data-data[0])/1e9 #since time is in nanoseconds

    UncerMissmatces = np.where(timeMinusExpectedTime1  > uncer, 1, 0)
    numberOfUncerMissmatches = np.sum(UncerMissmatces)
    ratuiOfUncerMissmatches = numberOfUncerMissmatches / data.size
    matchInPercent=100-ratuiOfUncerMissmatches*100
    fig,ax=plt.subplots(2, 1)
    fig.set_size_inches(11.326,7 , forward=True)
    fig.suptitle(r'\textbf{Abweichung der Zeitstempel vom erwarteten Wert.\\\\ Die Unsicherheit überdeck '+"{:2.1f}".format(matchInPercent)+' \% der Messwerte}')
    ax[0].plot(reltime,uncer,label='Unsicherheit von SSU',color=colors[1])
    ax[0].plot(reltime,-1*uncer, color=colors[1])
    ax[0].plot(reltime,timeMinusExpectedTime1,label='Abweichung vom erwartet Wert SSU', color=colors[0])
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_ylabel(r"$\mathbf{\Delta t}$\textbf{ in ns}")
    ax[1].plot(reltime[zoomArea[0]:zoomArea[1]],uncer[zoomArea[0]:zoomArea[1]],label='Unsicherheit von SSU generiert',color=colors[1])
    ax[1].plot(reltime[zoomArea[0]:zoomArea[1]],-1*uncer[zoomArea[0]:zoomArea[1]], color=colors[1])
    ax[1].plot(reltime[zoomArea[0]:zoomArea[1]],timeMinusExpectedTime1[zoomArea[0]:zoomArea[1]],label='Abweichung vom erwartet Wert SSU', color=colors[0])
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_xlabel(r"\textbf{Relative Zeit in s}")
    ax[1].set_ylabel(r"$\mathbf{\Delta t}$\textbf{ in ns}")
    fig.savefig("timingErrorZoom.svg")
    fig.savefig("timingErrorZoom.png")
    fig.show()

hdf=h5py.File('20210721MultiBoard_GPS_test.hfd5', "w")
timediffBetweanBoards=hdf['RAWDATA/0x60ad0100_BMA_280/Absolutetime'][0,12:-998].astype(np.int64)-hdf['RAWDATA/0xf1030100_BMA_280/Absolutetime'][0,11:-999].astype(np.int64)
timeDiffMean=np.mean(timediffBetweanBoards)
timeDiffStd=np.std(timediffBetweanBoards)
uncer1=hdf['RAWDATA/0x60ad0100_BMA_280/Absolutetime_uncertainty'][0,12:-998]
uncer2=hdf['RAWDATA/0xf1030100_BMA_280/Absolutetime_uncertainty'][0,12:-998]
expectedUncer=np.sqrt(uncer1*uncer1+uncer2*uncer2)# aussuming gausian error propagation
UncerMissmatces=np.where(timediffBetweanBoards>expectedUncer,1,0)
numberOfUncerMissmatches=np.sum(UncerMissmatces)
ratuiOfUncerMissmatches=numberOfUncerMissmatches/UncerMissmatces.size
# Fit a normal distribution to the data:

bins=np.max(timediffBetweanBoards)-np.min(timediffBetweanBoards)
plotHist(timediffBetweanBoards,bins)
plottimediffWithUncerAndZoom(hdf['RAWDATA/0xf1030100_BMA_280/Absolutetime'][0,12:-998],uncer2)

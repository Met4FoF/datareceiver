import h5py as h5py
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import os
TUBSNamedColors={'TUBSBlue':(0/255,112/255,155/255),
                 'TUBSOrange':(250/255,110/255,0/255),
                 'TUBSGreen':(109/255,131/255,0/255),
                 'TUBSViolet':(81/255,18/255,70/255),
                 'TUBSLightBlue':(102/255,180/255,211/255),
                 'TUBSLightOrange':(255/255,200/255,41/255),
                 'TUBSLightGreen':(172/255,193/255,58/255),
                 'TUBSLightViolet':(138/255,48/255,127/255)}
tubscolorCycle=list(TUBSNamedColors.values())
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=tubscolorCycle) #TUBS Blue,Orange,Green,Violet,Light Blue,Light Orange,Lieght green,Light Violet
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
LANG='DE'
if LANG=='DE':
    import locale
    locale.setlocale(locale.LC_NUMERIC,"de_DE.utf8")
    locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    plt.rcParams['text.latex.preamble'] = r'\usepackage{icomma}\usepackage{amsmath}\boldmath' # remove nasty Space behind comma in de_DE.utf8 locale https://stackoverflow.com/questions/50657326/matplotlib-locale-de-de-latex-space-btw-decimal-separator-and-number
    plt.rcParams['axes.formatter.use_locale'] = True
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'NexusProSans'
plt.rcParams['mathtext.it'] = 'NexusProSans:italic'
plt.rcParams['mathtext.bf'] = 'NexusProSans:bold'
plt.rcParams['mathtext.tt'] = 'NexusProSans:monospace'
plt.rc('text', usetex=True)
plt.rc("figure", figsize=[16,9])  # fontsize of the figure title
plt.rc("figure", dpi=300)
PLTSCALFACTOR = 1.5
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
figSaveCounter = 0
SAVEFOLDER = './timeDefImages'
SHOW=False

def plotHist(data,bins):
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    mu, std = norm.fit(data)
    # Plot the histogram.
    fig,ax=plt.subplots()
    ax.hist(data, bins=bins, density=True)
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, bins)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p)
    title = r'\begin{center}\textbf{Abweichung zwischen den zwei SmartUp-Units}\\ $\mu='+locale.format_string('%2.2f',mu)+r'$~\textbf{ns}  $\sigma='+locale.format_string('%2.2f',std)+r'$~\textbf{ns}\end{center}'
    ax.grid(linestyle='dashed')
    ax.set_xlabel(r"\textbf{Differenz der Zeitstempel in ns}")
    ax.set_ylabel(r"\textbf{Releative Häufigkeit}")
    xlimMax=np.max(abs(np.array(ax.get_xlim())))
    ax.set_xlim([-xlimMax,xlimMax])#
    fig.suptitle(title)
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2)+'_TimeDiffHist.png'), dpi=300,bbox_inches='tight')
    fig.savefig( os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_TimeDiffHist.pdf'), dpi=300,bbox_inches='tight')
    fig.savefig( os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_TimeDiffHist.svg'), dpi=300,bbox_inches='tight')
    globals()['figSaveCounter']+=1
    if SHOW:
        fig.show()

def plottimediffWithUncerAndZoom(data,uncer,period=(1e8+0.0005950575459905795),zoomArea=(910000,920000)):
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    timeMinusExpectedTime1 = data.astype(np.int64)-data[0].astype(np.int64) - (period * np.arange(data.size))
    reltime=(data-data[0])/1e9 #since time is in nanoseconds
    UncerMissmatcesBool = np.where(abs(timeMinusExpectedTime1)  > uncer, True, False)
    UncerMissmatces = np.where(abs(timeMinusExpectedTime1) > uncer, 1, 0)
    numberOfUncerMissmatches = np.sum(UncerMissmatces)
    ratuiOfUncerMissmatches = numberOfUncerMissmatches / data.size
    matchInPercent=100-ratuiOfUncerMissmatches*100
    reltimeIn=np.copy(timeMinusExpectedTime1)
    reltimeIn[UncerMissmatcesBool]=np.NaN
    invertUncerMissmatcesBool=np.invert(UncerMissmatcesBool)
    reltimeOut=np.copy(timeMinusExpectedTime1)
    reltimeOut[invertUncerMissmatcesBool]=np.NaN
    fig,ax=plt.subplots(2, 1)
    title=r'\begin{center}\textbf{Abweichung der Zeitstempel vom erwarteten Wert.\\ Die Unsicherheit überdeck '+locale.format_string('%2.2f',matchInPercent)+r' \% der Messwerte}\end{center}'
    fig.suptitle(title)
    ax[0].plot(reltime,uncer,label=r'\textbf{Von DAU berechnete Zeitunsicherheit} $u_t(t)$',color=TUBSNamedColors['TUBSOrange'])
    ax[0].plot(reltime,-1*uncer, color=TUBSNamedColors['TUBSOrange'])
    ax[0].plot(reltime,reltimeIn,label=r'\textbf{Zeitabweichung} $\Delta t(t) <u_t(t)$', color=TUBSNamedColors['TUBSBlue'])
    ax[0].plot(reltime,reltimeOut,label=r'\textbf{Zeitabweichung} $\Delta t(t) >u_t(t)$', color=TUBSNamedColors['TUBSViolet'])
    ax[0].axvline(x=reltime[zoomArea[0]], color=TUBSNamedColors['TUBSGreen'], linestyle='-')
    ax[0].axvline(x=reltime[zoomArea[1]], color=TUBSNamedColors['TUBSGreen'], linestyle='-')
    ax[0].grid(linestyle='dashed')
    #ax[0].legend(loc='upper right')
    ax[0].set_ylabel(r"$\mathbf{\Delta t}$\textbf{ in ns}")
    ax[1].plot(reltime[zoomArea[0]:zoomArea[1]],uncer[zoomArea[0]:zoomArea[1]],label=r'\textbf{Von DAU berechnete Zeitunsicherheit} $u_t(t)$',color=TUBSNamedColors['TUBSOrange'])
    ax[1].plot(reltime[zoomArea[0]:zoomArea[1]],-1*uncer[zoomArea[0]:zoomArea[1]], color=TUBSNamedColors['TUBSOrange'])
    ax[1].plot(reltime[zoomArea[0]:zoomArea[1]],reltimeIn[zoomArea[0]:zoomArea[1]],label=r'\textbf{Zeitabweichung} $\Delta t(t) <u_t(t)$', color=TUBSNamedColors['TUBSBlue'])
    ax[1].plot(reltime[zoomArea[0]:zoomArea[1]],reltimeOut[zoomArea[0]:zoomArea[1]],label=r'\textbf{Zeitabweichung} $\Delta t(t) >u_t(t)$', color=TUBSNamedColors['TUBSViolet'])
    ax[1].set_xlabel(r"\textbf{Relative Zeit in s}")
    ax[1].set_ylabel(r"$\mathbf{\Delta t}$\textbf{ in ns}")
    ax[1].grid(linestyle='dashed')
    ax[1].legend(loc='upper left')
    yLowAX0=ax[0].get_ylim()[0]
    yHighAX1 = ax[1].get_ylim()[1]
    con1 = ConnectionPatch(xyA=(reltime[zoomArea[0]],yLowAX0), xyB=(reltime[zoomArea[0]],yHighAX1), coordsA="data", coordsB="data",
                          axesA=ax[0], axesB=ax[1], color=TUBSNamedColors['TUBSGreen'])
    con2 = ConnectionPatch(xyA=(reltime[zoomArea[1]],yLowAX0), xyB=(reltime[zoomArea[1]],yHighAX1), coordsA="data", coordsB="data",
                          axesA=ax[0], axesB=ax[1], color=TUBSNamedColors['TUBSGreen'])
    ax[0].add_artist(con1)
    ax[0].add_artist(con2)
    ax[1].axvline(x=reltime[zoomArea[0]], color=TUBSNamedColors['TUBSGreen'], linestyle='-')
    ax[1].axvline(x=reltime[zoomArea[1]], color=TUBSNamedColors['TUBSGreen'], linestyle='-')
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2)+'_timingError.png'), dpi=300,bbox_inches='tight')
    fig.savefig( os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_timingError.pdf'), dpi=300,bbox_inches='tight')
    fig.savefig( os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_timingError.svg'), dpi=300,bbox_inches='tight')
    globals()['figSaveCounter']+=1
    if SHOW:
        fig.show()

def plotGPSTimeSyncAlgo(hdfDset):
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    #TODO implenet gliding window
    relTicks=hdfDset[0,1:121]-hdfDset[0,1]
    deltaTicks=np.diff(hdfDset[0,:120])
    blockMean=np.zeros([60])
    blockStd=np.zeros([60])
    for i in range(60):
        blockMean[i]=np.mean(deltaTicks[i:i+60])
        blockStd[i] = np.std(deltaTicks[i:i + 60])
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(relTicks,'x',label='Messwerte in Ticks')
    ax[0].grid(linestyle='dashed')
    ax[0].legend()
    ax[0].set_ylabel(r"\textbf{Counterwert in Ticks}")
    ax[1].set_xlabel(r"\textbf{PPS Puls index alias Relative Zeit in s}")
    ax[1].set_ylabel(r"$\mathbf{\Delta t}$\textbf{ in Tickes}\\ \textbf{1 Tick} $\mathbf{\approx}$ \textbf{9.259~ns}")
    ax[1].fill_between(np.arange(60),blockMean-2*blockStd,blockMean+2*blockStd,alpha=0.5,color=colors[1])
    ax[1].plot(np.arange(60), deltaTicks[59:],'x', label=r'Messwerte in Ticks', color=colors[0])
    ax[1].plot(np.arange(60), blockMean-2*blockStd, label=r'$2\cdot$ Standardabweichung $2\sigma$='+"{:3.2f}".format(np.mean(blockStd))+'~Tickes', color=colors[1])
    ax[1].plot(np.arange(60), blockMean+2*blockStd, color=colors[1])
    ax[1].plot(np.arange(60), blockMean, label=r'Mittelwert $\mu$=' + "{:9.3f}".format(np.mean(blockMean)) + '~Ticks',color=colors[3])
    ax[1].grid(linestyle='dashed')
    ax[1].legend()
    fig.savefig("timingTickes.svg",bbox_inches='tight')
    fig.savefig("timingTicks.png",bbox_inches='tight')
    fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2)+'_timingTickes.png'), dpi=300,bbox_inches='tight')
    fig.savefig( os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_timingTickes.pdf'), dpi=300,bbox_inches='tight')
    fig.savefig( os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_timingTickes.svg'), dpi=300,bbox_inches='tight')
    globals()['figSaveCounter']+=1
    if SHOW:
        fig.show()



hdf=h5py.File('20210721MultiBoard_GPS_test.hfd5', "r+")
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
plottimediffWithUncerAndZoom(hdf['RAWDATA/0x60ad0100_BMA_280/Absolutetime'][0,12:-998],uncer1)
plotGPSTimeSyncAlgo(hdf["RAWDATA/0xf1031400_uBlox_NEO-7_GPS/Time_Ticks"])

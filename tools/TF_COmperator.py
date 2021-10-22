# import h5py
# TODO implent proper multi threaded hdf file handling

import h5py as h5py
import numpy as np
import matplotlib.pyplot as plt

from uncertainties import ufloat
from uncertainties.umath import *  # sin(), etc.

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
PLTSCALFACTOR = 3
SMALL_SIZE = 12 * PLTSCALFACTOR
MEDIUM_SIZE = 16 * PLTSCALFACTOR
BIGGER_SIZE = 18 * PLTSCALFACTOR

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title



def plotRAWTFUncerComps(datafile,type='Phase',sensorName='0xbccb0000_MPU_9250',startIDX=0,stopIDX=17,title='Uncertainty of the phases components CEM measurments',zoom=False,lang='EN',zoomPlotPos=[0.3,0.5,0.2,0.2]):
    freqs=datafile['RAWTRANSFERFUNCTION/'+sensorName+'/Acceleration/Acceleration']['Excitation_frequency']['value'][startIDX:stopIDX]
    uncersToPlot={}
    phaseGroupNames=['Phase','SSU_ADC_Phase','REF_Phase','Delta_DUTSNYC_Phase','DUT_SNYNC_Phase','DUT_Phase']#,
    ampGroupNames=['DUT_amplitude','Excitation_amplitude','Magnitude']
    labels={'Delta_DUTSNYC_Phase':r'$2\sigma(\varphi_\mathrm{DUT}(\omega)-\varphi_\mathrm{Sync_{DAU}}(\omega))$',
                'SSU_ADC_Phase':r'$2u(\varphi_{ADC_{DAU}}(\omega))$',
                'REF_Phase':r'$2\sigma(\varphi_\mathrm{ACS}(\omega)-\varphi_\mathrm{Sync_{DAU}}(\omega))$',
                'DUT_Phase':r'$2\sigma(\varphi_{\mathrm{DUT}}(\omega))$',
                'DUT_SNYNC_Phase':r'$2\sigma(\varphi_{\mathrm{Sync_{DAU}}}(\omega))$',
                'Phase':r'$u(\varphi(\omega))$',
                'DUT_amplitude': '$2\sigma(\hat{y}_\mathrm{DUT})$',
                'Excitation_amplitude': '$2\sigma(\hat{a}_\mathrm{ACS})$',
                'Magnitude': '$2\sigma(|S(\omega)|)$'
            }
    alphas={'Delta_DUTSNYC_Phase':1,
                'SSU_ADC_Phase':1,
                'REF_Phase':1,
                'DUT_Phase':0.5,
                'DUT_SNYNC_Phase':0.5,
                'Phase':1,
                'DUT_amplitude':1,
                'Excitation_amplitude':1,
                'Magnitude':1
            }
    hatches={'Delta_DUTSNYC_Phase':"O.",
                'SSU_ADC_Phase':"|",
                'REF_Phase': "/",
                'DUT_Phase':"o",
                'DUT_SNYNC_Phase':"O",
                'Phase':"|/o.",
                'DUT_amplitude': '|',
                'Excitation_amplitude': '-',
                'Magnitude': '+'
             }
    if type=='Phase':
        for pGN in phaseGroupNames:
            phaseUncerData=datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration'][pGN]['uncertainty'][
                startIDX:stopIDX]
            phaseUncerDataUnit=datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration'][pGN].attrs['Unit']
            if phaseUncerDataUnit=='\\radian':
                uncersToPlot[pGN]=phaseUncerData/np.pi*180
            elif phaseUncerDataUnit=='\\degree':
                uncersToPlot[pGN]=phaseUncerData
            else:
                raise RuntimeError(phaseGroupNames+' has unsupported unit '+phaseUncerDataUnit)
    if type=='Mag':
        for ampGN in ampGroupNames:
            ampUncerData=datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration'][ampGN]['uncertainty'][
                startIDX:stopIDX]
            ampValData=datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration'][ampGN]['value'][
                startIDX:stopIDX]
            uncersToPlot[ampGN]=(ampUncerData/ampValData)*100
    idxs=np.arange(freqs.size)
    fig,ax=plt.subplots()
    fig.set_size_inches(20, 10)
    i=0
    for uncerKey in uncersToPlot.keys():
        ax.bar(idxs+(1/(len(uncersToPlot.keys())+1))*i,uncersToPlot[uncerKey],width=1/(len(uncersToPlot.keys())+1),label=labels[uncerKey],alpha=alphas[uncerKey],hatch=hatches[uncerKey])
        i+=+1
    ax.set_xticks(idxs)
    boldFreqlabels = []
    for freq in freqs:
        boldFreqlabels.append(r'\textbf{' + str(freq) + '}')
    ax.set_xticklabels(boldFreqlabels, rotation=0)
    if lang=='EN':
        ax.set_xlabel(r'\textbf{Excitation frequency} \textbf{in Hz}')
    elif lang=='DE':
        ax.set_xlabel(r'\textbf{Anregungsfrequenz in Hz}')
    if type == 'Phase':
        if lang=='EN':
            ax.set_ylabel(r'\textbf{Type A components of}'+'\n' +r'\textbf{phase in $^\circ$}')
        elif lang=='DE':
            ax.set_ylabel(r'\textbf{Phasenkomponenten Typ A in $^\circ$}')
    if type == 'Mag':
        if lang== 'EN':
            ax.set_ylabel(r'\textbf{Type A components of }'+'\n'+r'\textbf{magnitude in \%}')
        elif lang=='DE':
            ax.set_ylabel(r'\textbf{Magnitudenkomponenten Typ A in \%}')
    ax.grid()
    if title!=None and title != '':
            ax.set_title(r'\textbf{'+title+'}')
    if zoom!=False:
        if type != 'Phase':
            raise ValueError("zoom is only usefull for Phase")
        numPlotCOmponents=len(uncersToPlot.keys())
        ax2=fig.add_axes(zoomPlotPos)
        ylim=2*uncersToPlot['SSU_ADC_Phase'][zoom]
        i=0
        ax2.set_ylim(ylim)
        for uncerKey in uncersToPlot.keys():
            ax2.bar((1/numPlotCOmponents)*i, uncersToPlot[uncerKey][zoom],width=(1/(numPlotCOmponents)), label=labels[uncerKey], alpha=alphas[uncerKey],hatch=hatches[uncerKey])
            i=i+1
            ax2.set_ylim([0,ylim])
            ax2.ticklabel_format(axis='y', scilimits=[-2, 2])
            # for major ticks
            ax2.set_xticks([])
            # for minor ticks
            ax2.set_xticks([], minor=True)
            if lang=='EN':
                ax2.set_xlabel(r'\textbf{Frequency '+str(freqs[zoom])+' Hz}')
            elif lang=='DE':
                ax2.set_xlabel(r'\textbf{Frequenz ' + str(freqs[zoom]) + ' Hz}')
            #ax2.set_ylabel(r'$^\circ$')
    ax.legend()
    fig.savefig('tmp.png', dpi=200)
    fig.show()

def plotMeanTfs(datafile,sensorName='0xbccb0000_MPU_9250',numofexpPerLoop=17,loopsPerRepititon=[5,5,5,5,5],repName=['0 mm','1.25 mm','1.5 mm','1.93 mm','1.7 mm']):
    freqs=datafile['RAWTRANSFERFUNCTION/'+sensorName+'/Acceleration/Acceleration']['Excitation_frequency']['value'][0:numofexpPerLoop]
    phaseUncerData = datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Phase']['uncertainty']
    phaseValData = datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Phase']['value']
    ampUncerData = datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Magnitude']['uncertainty']
    ampValData = datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Magnitude']['value']
    loopsProcessed=0
    fig,ax=plt.subplots(2, sharex=True)
    overallAmpMeanData=np.zeros([len(loopsPerRepititon),numofexpPerLoop])
    overallPhaseMeanData = np.zeros([len(loopsPerRepititon), numofexpPerLoop])
    for loopIDX in range(len(loopsPerRepititon)):
        loopsInThisBlock=loopsPerRepititon[loopIDX]
        ampData=np.zeros([loopsInThisBlock,numofexpPerLoop])
        phaseData=np.zeros([loopsInThisBlock,numofexpPerLoop])
        for i in range(loopsInThisBlock):
            startIDX=loopsProcessed*numofexpPerLoop+i*numofexpPerLoop
            stopIDX=startIDX+numofexpPerLoop
            ampData[i,:]=ampValData[startIDX:stopIDX]
            phaseData[i,:]=phaseValData[startIDX:stopIDX]
        ampMean=np.mean(ampData,axis=0)
        ampSTD = np.std(ampData, axis=0)
        phaseMean = np.mean(phaseData, axis=0)
        phaseSTD = np.std(phaseData, axis=0)
        ax[0].errorbar(freqs*(1+0.002*(loopIDX+1)), ampMean,   yerr=2 * ampSTD, label=repName[loopIDX],fmt='o')
        ax[1].errorbar(freqs*(1+0.002*(loopIDX+1)), phaseMean/np.pi*180, yerr=(2 * phaseSTD)/np.pi*180, label=repName[loopIDX],fmt='o')
        loopsProcessed+=loopsInThisBlock
        overallAmpMeanData[loopIDX,:]=ampMean
        overallPhaseMeanData[loopIDX,:]=phaseMean
    ax[0].set_xlabel("Frequency in Hz")
    ax[0].set_xscale('log')
    ax[0].set_ylabel(r"$|S(\omega)|$  in $\frac{\mathrm{m s}^-2}{\mathrm{m s}^-2}$")
    ax[1].set_ylabel(r"$\varphi(\omega)$ in $^\circ$")
    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    fig.show()


    overallAmpMean=np.mean(overallAmpMeanData,axis=0)
    overallPhaseMean = np.mean(overallPhaseMeanData,axis=0)
    fig2,ax2=plt.subplots(2, sharex=True)
    overallAmpMeanData=np.zeros([len(loopsPerRepititon),numofexpPerLoop])
    overallPhaseMeanData = np.zeros([len(loopsPerRepititon), numofexpPerLoop])
    loopsProcessed = 0
    for loopIDX in range(len(loopsPerRepititon)):
        loopsInThisBlock=loopsPerRepititon[loopIDX]
        ampData=np.zeros([loopsInThisBlock,numofexpPerLoop])
        phaseData=np.zeros([loopsInThisBlock,numofexpPerLoop])
        for i in range(loopsInThisBlock):
            startIDX=loopsProcessed*numofexpPerLoop+i*numofexpPerLoop
            stopIDX=startIDX+numofexpPerLoop
            ampData[i,:]=ampValData[startIDX:stopIDX]
            phaseData[i,:]=phaseValData[startIDX:stopIDX]
        ampMean=np.mean(ampData,axis=0)
        ampSTD = np.std(ampData, axis=0)
        phaseMean = np.mean(phaseData, axis=0)
        phaseSTD = np.std(phaseData, axis=0)
        ax2[0].errorbar(freqs*(1+0.002*(loopIDX+1)), (ampMean-overallAmpMean),   yerr=2 * ampSTD, label=repName[loopIDX],fmt='o')
        ax2[1].errorbar(freqs*(1+0.002*(loopIDX+1)), (phaseMean-overallPhaseMean)/np.pi*180, yerr=(2 * phaseSTD)/np.pi*180, label=repName[loopIDX],fmt='o')
        loopsProcessed+=loopsInThisBlock
    ax2[1].set_xlabel("Frequency in Hz")
    ax2[0].set_xscale('log')
    ax2[0].set_ylabel(r"$|S(\omega)|-\overline{|S|}$  in $\frac{\mathrm{m s}^-2}{\mathrm{m s}^-2}$")
    ax2[1].set_ylabel(r"$\varphi(\omega) -\overline{\varphi(\omega) }$ in $^\circ$")
    ax2[0].legend()
    ax2[1].legend()
    ax2[0].grid()
    ax2[1].grid()
    fig2.show()


    print("test")

if __name__ == "__main__":


    hdffilename = r"/home/benedikt/data/MPU9250_PTB_Reproduktion_platten/usedRuns/MPU9250_Platten.hdf5"
    leadSensorname = '0x1fe40000_MPU_9250'
    datafile = h5py.File(hdffilename, "r")
    plotMeanTfs(datafile, sensorName=leadSensorname)


# import h5py
# TODO implent proper multi threaded hdf file handling

import h5py as h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from uncertainties import unumpy, ufloat
from uncertainties.umath import *  # sin(), etc.

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
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def generateWigthedMeanFromArrays(values,uncers):
    mean = np.average(values, weights=1 / (uncers ** 2), axis=0)
    valMinusMean = values - np.tile(mean, (values.shape[0], 1))
    std=np.sqrt(np.average(valMinusMean**2,weights=1 / (uncers ** 2),axis=0))
    return mean,std

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

def plotMeanTfs(datafile,sensorName='0xbccb0000_MPU_9250',numofexpPerLoop=17,loopsPerRepititon=[10,5,5,5,5,5],repName=['Ref','225~$^\circ$ 0.00 mm','~45~~$^\circ$ 1.25 mm','135 $^\circ$ 1.50 mm','290~$^\circ$ 1.93 mm','200~$^\circ$ 1.70 mm'],lang='EN'):
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
    if lang=='EN':
        ax[1].set_xlabel(r"\textbf{Frequency in Hz}")
    if lang=='DE':
        ax[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax[0].set_xscale('log')
    ax[0].set_ylabel(r"$|S(\omega)|$  \textbf{in} $\frac{\mathrm{m s}^-2}{\mathrm{m s}^-2}$")
    ax[1].set_ylabel(r"$\varphi(\omega)$ \textbf{in} $^\circ$")
    ax[0].legend(ncol=3)
    ax[1].legend(ncol=3)
    ax[0].grid()
    ax[1].grid()
    ax[0].grid(axis='x',which = 'minor', linestyle = '--')
    ax[1].grid(axis='x',which='minor', linestyle='--')
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
    if lang=='EN':
        ax2[1].set_xlabel(r"\textbf{Frequency in Hz}")
    if lang=='DE':
        ax2[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax2[0].set_xscale('log')
    ax2[0].set_ylabel(r"$|S(\omega)|-\overline{|S|}$  in $\frac{\mathrm{m s}^-2}{\mathrm{m s}^-2}$")
    ax2[1].set_ylabel(r"$\varphi(\omega) -\overline{\varphi(\omega) }$ in $^\circ$")
    ax2[0].legend(ncol=3)
    ax2[1].legend(ncol=3)
    ax2[0].grid()
    ax2[1].grid()
    fig2.show()
    print("test")



def plotMeanTfsOneFile(datafile,sensorName='0xbccb0000_MPU_9250',numofexpPerLoop=17,loopsPerRepititon=[5,5,5,5,5],repName=['225~$^\circ$ 0.00 mm','~45~~$^\circ$ 1.25 mm','135 $^\circ$ 1.50 mm','290~$^\circ$ 1.93 mm','200~$^\circ$ 1.70 mm'],lang='EN'):
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
    if lang=='EN':
        ax[1].set_xlabel(r"\textbf{Frequency in Hz}")
    if lang=='DE':
        ax[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax[0].set_xscale('log')
    ax[0].set_ylabel(r"$|S(\omega)|$  \textbf{in} $\frac{\mathrm{m s}^-2}{\mathrm{m s}^-2}$")
    ax[1].set_ylabel(r"$\varphi(\omega)$ \textbf{in} $^\circ$")
    ax[0].legend(ncol=3)
    ax[1].legend(ncol=3)
    ax[0].grid()
    ax[1].grid()
    ax[0].grid(axis='x',which = 'minor', linestyle = '--')
    ax[1].grid(axis='x',which='minor', linestyle='--')
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
    if lang=='EN':
        ax2[1].set_xlabel(r"\textbf{Frequency in Hz}")
    if lang=='DE':
        ax2[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax2[0].set_xscale('log')
    ax2[0].set_ylabel(r"$|S(\omega)|-\overline{|S|}$  in $\frac{\mathrm{m s}^-2}{\mathrm{m s}^-2}$")
    ax2[1].set_ylabel(r"$\varphi(\omega) -\overline{\varphi(\omega) }$ in $^\circ$")
    ax2[0].legend(ncol=3)
    ax2[1].legend(ncol=3)
    ax2[0].grid()
    ax2[1].grid()
    fig2.show()
    print("test")

def generateTFFromRawData(datafile,style='PTB',sensorName='0xbccb0000_MPU_9250'):
    uniqueFreqs=np.unique(datafile['RAWTRANSFERFUNCTION/'+sensorName+'/Acceleration/Acceleration']['Excitation_frequency']['value'])
    if style=='PTB':
        freqs = datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Excitation_frequency']['value'][:]
        numOfLoops = int(freqs.size / uniqueFreqs.size)
        phaseValData = np.reshape(datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Phase']['value'][:],[numOfLoops,uniqueFreqs.size])
        phaseUncerData = np.reshape(datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Phase']['uncertainty'][:],[numOfLoops,uniqueFreqs.size])
        ampValData = np.reshape(datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Magnitude']['value'][:],[numOfLoops,uniqueFreqs.size])
        ampUncerData = np.reshape(datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Magnitude']['uncertainty'][:],[numOfLoops,uniqueFreqs.size])
        testfreqs=np.reshape(freqs,[numOfLoops,uniqueFreqs.size])
        testFreqStd=np.std(testfreqs,axis=0)
        if(np.sum(testFreqStd))>10-8:
            raise RuntimeError("Freqs do not macht expected sceme")
    if style == 'CEM':
        freqs = datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Excitation_frequency']['value'][:]
        numOfLoops = 0
        stratIdx=[]
        for i in range(freqs.size-2):
            if (freqs[i-2]==80 and freqs[i-1]==250):
                numOfLoops+=1
                stratIdx.append(i)
        numberOfFreqs=int(np.mean(np.diff(stratIdx)))-2# substract the 250 and 80 Hz test tones

        ampValData = np.zeros([numOfLoops,numberOfFreqs])
        ampUncerData = np.zeros([numOfLoops,numberOfFreqs])
        phaseValData = np.zeros([numOfLoops,numberOfFreqs])
        phaseUncerData = np.zeros([numOfLoops,numberOfFreqs])
        testfreqs = np.zeros([numOfLoops,numberOfFreqs])
        for i in range(numOfLoops):
            start=stratIdx[i]
            stop=start+numberOfFreqs
            phaseValData[i,:] = datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Phase']['value'][start:stop]
            phaseUncerData[i,:] = datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Phase']['uncertainty'][start:stop]
            ampValData[i,:] = datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Magnitude']['value'][start:stop]
            ampUncerData[i,:] = datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Magnitude']['uncertainty'][start:stop]
            testfreqs[i,:] = datafile['RAWTRANSFERFUNCTION/' + sensorName + '/Acceleration/Acceleration']['Excitation_frequency']['value'][start:stop]
        testFreqStd=np.std(testfreqs,axis=0)
        if(np.sum(testFreqStd))>10-8:
            raise RuntimeError("Freqs do not macht expected sceme")
    testFreqsMean=np.mean(testfreqs,axis=0)
    ampMean,stdAmpWigth=generateWigthedMeanFromArrays(ampValData,ampUncerData)
    phaseMean,stdPhaseWight=generateWigthedMeanFromArrays(phaseValData,phaseUncerData)
    return  testFreqsMean,ampMean,stdAmpWigth,phaseMean,stdPhaseWight




def plotTFCOmparison(dict,lang='DE',uncerType='typeA',titleExpansion='Test'):
    numOfTfs=len(dict.keys())
    i=0
    for TfDictKey in dict:
        TfDictEntry=dict[TfDictKey]
        testFreqsMean, ampMean, stdAmpWigth, phaseMean, stdPhaseWight=generateTFFromRawData(TfDictEntry['dataFile'],style=TfDictEntry['style'],sensorName=TfDictEntry['sensorName'])

        freqs=testFreqsMean
        if i ==0:
            labels = []
            ampsArray=np.zeros([numOfTfs,ampMean.size])
            stdAmpWigthArray=np.zeros([numOfTfs,stdAmpWigth.size])
            phaseMeanArray=np.zeros([numOfTfs,phaseMean.size])
            stdPhaseWightArray=np.zeros([numOfTfs,stdPhaseWight.size])
        ampsArray[i, :] = ampMean

        phaseMeanArray[i, :] = phaseMean-TfDictEntry['phaseOffset']
        if uncerType == 'typeA':
            stdAmpWigthArray[i, :] = stdAmpWigth
            stdPhaseWightArray[i, :]=stdPhaseWight
        if uncerType =='CMC':
            if TfDictEntry['style']=='PTB':
                stdPhaseWightArray[i, :] = (np.ones_like(stdPhaseWightArray[i, :])*0.1/180*np.pi)/2
                stdAmpWigthArray[i, :] = (0.1/100*ampMean)/2
            if TfDictEntry['style']=='CEM':
                stdPhaseWightArray[i, :] = (np.ones_like(stdPhaseWightArray[i, :])*0.5/180*np.pi)/2
                stdAmpWigthArray[i, :] = (0.4/100*ampMean)/2

        labels.append(TfDictKey)
        i=i+1
    ampMean, ampUncer = generateWigthedMeanFromArrays(ampsArray[:2,:], stdAmpWigthArray[:2,:])
    phaseMean, phaseUncer = generateWigthedMeanFromArrays(phaseMeanArray[:2,:], stdPhaseWightArray[:2,:])

    fig,ax=plt.subplots(2, sharex=True)
    for TFIDX in range(numOfTfs):
        ax[0].errorbar(freqs*(1+0.002*(TFIDX +1)), ampsArray[TFIDX,:],   yerr=2 * stdAmpWigthArray[TFIDX, :], label=labels[TFIDX],fmt='o',lw=PLTSCALFACTOR*2,ms=PLTSCALFACTOR*6,alpha=0.8)
        ax[1].errorbar(freqs*(1+0.002*(TFIDX +1)), phaseMeanArray[TFIDX,:]/np.pi*180, yerr=(2 * stdPhaseWightArray[TFIDX, :])/np.pi*180, label=labels[TFIDX],fmt='o',lw=PLTSCALFACTOR*2,ms=PLTSCALFACTOR*6,alpha=0.8)
    if lang=='EN':
        ax[1].set_xlabel(r"\textbf{Frequency in Hz}")
    if lang=='DE':
        ax[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax[0].set_xscale('log')
    ax[0].set_ylabel(r"$|S(\omega)|$  \textbf{in} $\frac{\mathrm{m s}^-2}{\mathrm{m s}^-2}$")
    ax[1].set_ylabel(r"$\varphi(\omega)$ \textbf{in} $^\circ$")
    ax[0].legend(ncol=3)
    ax[1].legend(ncol=3)
    ax[0].grid(axis='x',which = 'minor', linestyle = '--')
    ax[1].grid(axis='x',which='minor', linestyle='--')
    ax[0].grid(axis='y',which = 'minor', linestyle = '--')
    ax[1].grid(axis='y',which='minor', linestyle='--')
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(4))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator(4))
    ax[0].grid(lw=PLTSCALFACTOR*0.66)
    ax[1].grid(lw=PLTSCALFACTOR*0.66)
    ax[0].set_title(r"\textbf{Übertragungsfunktion "+titleExpansion+'}')
    fig.show()

    fig2,ax2=plt.subplots(2, sharex=True)
    for TFIDX in range(numOfTfs):
        ax2[0].errorbar(freqs*(1+0.002*(TFIDX +1)), ampsArray[TFIDX,:]-ampMean,   yerr=2 * stdAmpWigthArray[TFIDX, :], label=labels[TFIDX],fmt='o',lw=PLTSCALFACTOR*2,ms=PLTSCALFACTOR*6,alpha=0.8)
        ax2[1].errorbar(freqs*(1+0.002*(TFIDX +1)), (phaseMeanArray[TFIDX,:]-phaseMean)/np.pi*180, yerr=(2 * stdPhaseWightArray[TFIDX, :])/np.pi*180, label=labels[TFIDX],fmt='o',lw=PLTSCALFACTOR*2,ms=PLTSCALFACTOR*6,alpha=0.8)
    if lang=='EN':
        ax2[1].set_xlabel(r"\textbf{Frequency in Hz}")
    if lang=='DE':
        ax2[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax2[0].set_xscale('log')
    ax2[0].set_ylabel(r"$|S(\omega)|-\overline{|S|}$  in $\frac{\mathrm{m s}^-2}{\mathrm{m s}^-2}$")
    ax2[1].set_ylabel(r"$\varphi(\omega) -\overline{\varphi(\omega) }$ in $^\circ$")
    ax2[0].legend(ncol=3)
    ax2[1].legend(ncol=3)
    ax2[0].grid(axis='x',which = 'minor', linestyle = '--')
    ax2[1].grid(axis='x',which='minor', linestyle='--')
    ax2[0].grid(axis='y',which = 'minor', linestyle = '--')
    ax2[1].grid(axis='y',which='minor', linestyle='--')
    ax2[0].yaxis.set_minor_locator(AutoMinorLocator(4))
    ax2[1].yaxis.set_minor_locator(AutoMinorLocator(4))
    ax2[0].grid(lw=PLTSCALFACTOR*0.66)
    ax2[1].grid(lw=PLTSCALFACTOR*0.66)
    ax2[0].set_title(r"\textbf{Abweichung vom gewichteten Mittel der Übertragungsfunktionen " + titleExpansion + '}')
    fig2.show()



if __name__ == "__main__":
    #hdffilename = r"/home/benedikt/data/MPU9250_PTB_Reproduktion_platten/usedRuns/MPU9250_Platten.hdf5"
    leadSensorname = '0x1fe40000_MPU_9250'
    CEMhdffilename = r"/home/benedikt/data/IMUPTBCEM/MPU9250CEM_v5.hdf5"
    CEMSensorname = '0xbccb0000_MPU_9250'
    CEMdatafile = h5py.File(CEMhdffilename, "r")
    CEMTFDIct={'style':'CEM',
               'dataFile':CEMdatafile,
               'sensorName':CEMSensorname}
    PTBhdffilename = r"/home/benedikt/data/IMUPTBCEM/PTB/MPU9250PTB.hdf5"
    PTBPlattendatafilename = r"/home/benedikt/data/IMUPTBCEM/MPU9250_Platten.hdf5"
    PTBSensorname = '0x1fe40000_MPU_9250'
    PTBdatafile = h5py.File(PTBhdffilename, "r")
    PTBPlattendatafile=h5py.File(PTBPlattendatafilename,'r')
    CEMTFDIct={'style':'CEM',
               'dataFile':CEMdatafile,
               'sensorName':CEMSensorname,
               'phaseOffset':np.pi}
    PTBTFDIct={'style':'PTB',
               'dataFile':PTBdatafile,
               'sensorName':PTBSensorname,
               'phaseOffset':0}
    PTBPlattenDIct={'style':'PTB',
               'dataFile':PTBPlattendatafile,
               'sensorName':PTBSensorname,
               'phaseOffset':0}
    
    plotMeanTfsOneFile(PTBPlattendatafile, sensorName=leadSensorname,lang='DE')
    #testFreqsMean,phaseMean, stdPhaseWight,ampMean,stdAmpWigth=generateTFFromRawData(datafile, sensorName=leadSensorname,style='CEM')
    TFDict={'PTB verschiedene Winkel': PTBPlattenDIct,'CEM':CEMTFDIct,'PTB erste Messung':PTBTFDIct,}
    

    plotTFCOmparison(TFDict,uncerType='typeA',titleExpansion='MPU9250 Unsicherheit TypA')
    plotTFCOmparison(TFDict,uncerType='CMC',titleExpansion='MPU9250 Unsicherheit CMC')

    """
    PTBSensorname = '0x1fe40000_BMA_280'
    CEMSensorname = '0xbccb0000_BMA_280'
    PTBhdffilename = r"/home/benedikt/data/BMACEMPTB/BMA280PTB.hdf5"
    CEMhdffilename = r"/home/benedikt/data/BMACEMPTB/BMA280CEM.hdf5"
    PTBdatafile = h5py.File(PTBhdffilename, "r")
    CEMdatafile = h5py.File(CEMhdffilename, "r")
    CEMBMADict={'style':'CEM',
               'dataFile':CEMdatafile,
               'sensorName':CEMSensorname,
               'phaseOffset':0}
    PTBBMADict={'style':'PTB',
               'dataFile':PTBdatafile,
               'sensorName':PTBSensorname,
               'phaseOffset':np.pi}
    TFDict = {'CEM': CEMBMADict, 'PTB': PTBBMADict}
    plotTFCOmparison(TFDict,uncerType='typeA',titleExpansion='BMA280 Unsicherheit TypA')
    plotTFCOmparison(TFDict,uncerType='CMC',titleExpansion='BMA280 Unsicherheit CMC')
    """
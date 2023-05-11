# import h5py
# TODO implent proper multi threaded hdf file handling

import h5py as h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
from uncertainties import unumpy, ufloat
from uncertainties.umath import *  # sin(), etc.
tubscolors=[(0/255,112/255,155/255),(250/255,110/255,0/255), (109/255,131/255,0/255), (81/255,18/255,70/255),(102/255,180/255,211/255),(255/255,200/255,41/255),(172/255,193/255,58/255),(138/255,48/255,127/255)]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=tubscolors) #TUBS Blue,Orange,Green,Violet,Light Blue,Light Orange,Lieght green,Light Violet
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\boldmath'
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
SAVEFOLDER = './tf_images'
SHOW=False
def generateWigthedMeanFromArrays(values,uncers):
    mean = np.average(values, weights=1 / (uncers ** 2), axis=0)
    valMinusMean = values - np.tile(mean, (values.shape[0], 1))
    std=np.sqrt(np.average(valMinusMean**2,weights=1 / (uncers ** 2),axis=0))
    return mean,std

def plotRAWTFUncerComps(datafile,type='Phase',sensorName='0xbccb0000_MPU_9250',startIDX=0,stopIDX=17,title='Uncertainty of the phases components CEM measurments',zoom=False,lang=LANG,zoomPlotPos=[0.2, 0.65, 0.2, 0.2]):
    freqs=datafile['RAWTRANSFERFUNCTION/'+sensorName+'/Acceleration/Acceleration']['Excitation_frequency']['value'][startIDX:stopIDX]
    uncersToPlot={}
    phaseGroupNames=['Phase','SSU_ADC_Phase','REF_Phase','Delta_DUTSNYC_Phase','DUT_SNYNC_Phase','DUT_Phase']
    ampGroupNames=['Magnitude','DUT_amplitude','Excitation_amplitude']
    phaseGroupNamesNestingDict={'Phase':{'REF_Phase':None,'Delta_DUTSNYC_Phase':{'DUT_SNYNC_Phase':None,'DUT_Phase':None}}}#,'SSU_ADC_Phase':None
    magGroupNamesNestingDict={'Magnitude':{'Excitation_amplitude':None,'DUT_amplitude':None}}
    labels={'Delta_DUTSNYC_Phase':r'$2\sigma(\varphi_\mathrm{DUT}(\omega)-\varphi_\mathrm{Sync_{DAU}}(\omega))$',
                'SSU_ADC_Phase':r'$2u(\varphi_{ADC_{DAU}}(\omega))$',
                'REF_Phase':r'$2\sigma(\varphi_\mathrm{ACS}(\omega)-\varphi_\mathrm{Sync_{DAU}}(\omega))$',
                'DUT_Phase':r'$2\sigma(\varphi_{\mathrm{DUT}}(\omega))$',
                'DUT_SNYNC_Phase':r'$2\sigma(\varphi_{\mathrm{Sync_{DAU}}}(\omega))$',
                'Phase':r'$u(\varphi(\omega))$',
                'DUT_amplitude': '$2\sigma(\hat{y}_\mathrm{DUT}(\omega))$',
                'Excitation_amplitude': '$2\sigma(\hat{a}_\mathrm{ACS}(\omega))$',
                'Magnitude': '$2\sigma(|S(\omega)|)$'
            }
    alphas={'Delta_DUTSNYC_Phase':1,
                'SSU_ADC_Phase':1,
                'REF_Phase':1,
                'DUT_Phase':1,
                'DUT_SNYNC_Phase':1,
                'Phase':1,
                'DUT_amplitude':1,
                'Excitation_amplitude':1,
                'Magnitude':1
            }
    colors={'Delta_DUTSNYC_Phase':(0/255,83/255,74/255),
                'SSU_ADC_Phase':tubscolors[3],
                'REF_Phase':tubscolors[1],
                'DUT_Phase':tubscolors[2],
                'DUT_SNYNC_Phase':(172/255,193/255,58/255),
                'Phase':tubscolors[0],
                'DUT_amplitude': (0/255,83/255,74/255),
                'Excitation_amplitude': tubscolors[1],
                'Magnitude': tubscolors[0]
             }
    hatches={'Delta_DUTSNYC_Phase':"",
                'SSU_ADC_Phase':"",
                'REF_Phase': "",
                'DUT_Phase':"",
                'DUT_SNYNC_Phase':"",
                'Phase':"",
                'DUT_amplitude': '',
                'Excitation_amplitude': '',
                'Magnitude': ''
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
    ax.set_axisbelow(True)
    ax.grid(axis='y',linestyle='dashed')
    i=0
    overAllWidth=0.8
    if type=='Mag':
        GroupNamesNestingDict=magGroupNamesNestingDict
    if type=='Phase':
        GroupNamesNestingDict = phaseGroupNamesNestingDict
    for comp,uncerKey0 in enumerate(GroupNamesNestingDict.keys()):
        ax.bar(idxs, uncersToPlot[uncerKey0], align='edge', label=labels[uncerKey0],width=overAllWidth, alpha=alphas[uncerKey0], hatch=hatches[uncerKey0],color=colors[uncerKey0])
        firstlevelKeyNum=len(GroupNamesNestingDict[uncerKey0].keys())
        #TODO remove this dirty hacking and replace with propper recursion !!!!
        if firstlevelKeyNum>0:
            width=1/(firstlevelKeyNum+2)*overAllWidth
            offset=1/(firstlevelKeyNum+1)*overAllWidth
            for comp1,uncerKey1 in enumerate(GroupNamesNestingDict[uncerKey0].keys()):
                pos1=idxs+offset+width*comp1
                ax.bar(pos1,uncersToPlot[uncerKey1], align='edge',width=width, label=labels[uncerKey1], alpha=alphas[uncerKey1], hatch=hatches[uncerKey1],color=colors[uncerKey1])
                try:
                    secondlevelKeyNum = len(GroupNamesNestingDict[uncerKey0][uncerKey1].keys())
                    secondwidth = width / (secondlevelKeyNum+ 1)*overAllWidth
                    secondoffset = 1 / (2*4 * (secondlevelKeyNum+ 1))*overAllWidth
                    for comp2, uncerKey2 in enumerate(GroupNamesNestingDict[uncerKey0][uncerKey1].keys()):
                        ax.bar(pos1+secondoffset+secondwidth*comp2, uncersToPlot[uncerKey2], align='edge', width=secondwidth, label=labels[uncerKey2], alpha=alphas[uncerKey2], hatch=hatches[uncerKey2],color=colors[uncerKey2])
                except AttributeError:
                    pass # None has no keys we dont ne to go an level deeper

        """
        uncerKey =list(uncersToPlot.keys())[0]
        ax.bar(idxs+(1/(len(uncersToPlot.keys())+1))*i,uncersToPlot[uncerKey],align='edge',label=labels[uncerKey],alpha=alphas[uncerKey],hatch=hatches[uncerKey])
        i+=+1
        for uncerKey in list(uncersToPlot.keys())[1:4]:
            print(idxs+(1/(len(uncersToPlot.keys())+1))*i)
            ax.bar(idxs+(1/(len(uncersToPlot.keys())+1-))*i,uncersToPlot[uncerKey],align='edge',width=1/4,label=labels[uncerKey],alpha=alphas[uncerKey],hatch=hatches[uncerKey],linestyle='--')
            i+=+1
        """
    ax.set_xticks(idxs+overAllWidth/2)
    boldFreqlabels = []
    if lang=='DE':
        locale.setlocale(locale.LC_NUMERIC, "de_DE.utf8")
        locale.setlocale(locale.LC_ALL, "de_DE.utf8")
    for freq in freqs:
        boldFreqlabels.append(r'$'+locale.format_string('%g',freq) +'$')
    ax.set_xticklabels(boldFreqlabels, rotation=0)
    if lang=='EN':
        ax.set_xlabel(r'\textbf{Excitation frequency} $\omega$ \textbf{in Hz}')
    elif lang=='DE':
        ax.set_xlabel(r'\textbf{Anregungsfrequenz $\omega$ in Hz}')
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
    if title!=None and title != '':
            ax.set_title(r'\textbf{'+title+'}')
    if zoom!=False:
        if type != 'Phase':
            raise ValueError("zoom is only usefull for Phase")
        numPlotCOmponents=len(uncersToPlot.keys())
        ax2=fig.add_axes(zoomPlotPos)
        ylim=2*uncersToPlot['SSU_ADC_Phase'][zoom]
        i=0
        ax2.set_ylim([0,ylim])
        for comp,uncerKey0 in enumerate(phaseGroupNamesNestingDict.keys()):
            idxs=0
            ax2.bar(idxs, uncersToPlot[uncerKey0][zoom], align='edge', label=labels[uncerKey0],width=1, alpha=alphas[uncerKey0], hatch=hatches[uncerKey0],color=colors[uncerKey0])
            firstlevelKeyNum=len(phaseGroupNamesNestingDict[uncerKey0].keys())
            #TODO remove this dirty hacking and replace with propper recursion !!!!
            if firstlevelKeyNum>0:
                width=1/(firstlevelKeyNum+1)
                offset=1/(2*(firstlevelKeyNum+1))
                for comp1,uncerKey1 in enumerate(phaseGroupNamesNestingDict[uncerKey0].keys()):
                    pos1=idxs+offset+width*comp1
                    ax2.bar(pos1,uncersToPlot[uncerKey1][zoom], align='edge',width=width, label=labels[uncerKey1], alpha=alphas[uncerKey1], hatch=hatches[uncerKey1],color=colors[uncerKey1])
                    try:
                        secondlevelKeyNum = len(phaseGroupNamesNestingDict[uncerKey0][uncerKey1].keys())
                        secondwidth = width / (secondlevelKeyNum+ 1)
                        secondoffset = 1 / (2*4 * (secondlevelKeyNum+ 1))
                        for comp2, uncerKey2 in enumerate(phaseGroupNamesNestingDict[uncerKey0][uncerKey1].keys()):
                            ax2.bar(pos1+secondoffset+secondwidth*comp2, uncersToPlot[uncerKey2][zoom], align='edge', width=secondwidth, label=labels[uncerKey2], alpha=alphas[uncerKey2], hatch=hatches[uncerKey2],color=colors[uncerKey2])
                    except AttributeError:
                        pass # None has no keys we dont ne to go an level deeper
        ax2.ticklabel_format(axis='y', scilimits=[-2, 2])
        # for major ticks
        ax2.set_xticks([])
        # for minor ticks
        ax2.set_xticks([], minor=True)
        if lang == 'EN':
            ax2.set_xlabel(r'\textbf{' + str(freqs[zoom]) + ' Hz}')
        elif lang == 'DE':
            ax2.set_xlabel(r'\textbf{' + str(freqs[zoom]) + ' Hz}')
        # ax2.set_ylabel(r'$^\circ$')

        """
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
        """
    ax.legend(loc='upper right')
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2)+'_'+str(type)+'_uncerComps.png'), dpi=300,bbox_inches='tight')
    fig.savefig( os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_'+str(type)+'_uncerComps.pdf'), dpi=300,bbox_inches='tight')
    globals()['figSaveCounter']+=1
    if SHOW:
        fig.show()

def plotMeanTfs(datafile,sensorName='0xbccb0000_MPU_9250',numofexpPerLoop=17,loopsPerRepititon=[10,5,5,5,5,5],repName=['255$^\circ$','45$^\circ$','135$^\circ$','290$^\circ$','200$^\circ$'],lang=LANG):
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
        ax[0].errorbar(freqs*(1+0.002*(loopIDX+1)), ampMean,   yerr=2 * ampSTD, label=r"\textbf{"+repName[loopIDX]+"}",fmt='o')
        ax[1].errorbar(freqs*(1+0.002*(loopIDX+1)), phaseMean/np.pi*180, yerr=(2 * phaseSTD)/np.pi*180, label=r"\textbf{"+repName[loopIDX]+"}",fmt='o')
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
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    fig.savefig(os.path.join(SAVEFOLDER,str(int(figSaveCounter)).zfill(2)+'_meanTF1.png'), dpi=300,bbox_inches='tight')
    fig.savefig( os.path.join(SAVEFOLDER,str(int(figSaveCounter)).zfill(2)+'_meanTF1.pdf'), dpi=300,bbox_inches='tight')
    figSaveCounter+=1
    if SHOW:
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
        ax2[0].errorbar(freqs*(1+0.002*(loopIDX+1)), (ampMean-overallAmpMean)*100,   yerr=2 * ampSTD*100, label=repName[loopIDX],fmt='o')
        ax2[1].errorbar(freqs*(1+0.002*(loopIDX+1)), (phaseMean-overallPhaseMean)/np.pi*180, yerr=(2 * phaseSTD)/np.pi*180, label=repName[loopIDX],fmt='o')
        loopsProcessed+=loopsInThisBlock
    if lang=='EN':
        ax2[1].set_xlabel(r"\textbf{Frequency in Hz}")
    if lang=='DE':
        ax2[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax2[0].set_xscale('log')
    ax2[0].set_ylabel(r"$|S(\omega)|-\overline{|S|}$ \textbf{in \%}")
    ax2[1].set_ylabel(r"$\varphi(\omega) -\overline{\varphi(\omega) }$ \textbf{in $^\circ$}")
    ax2[0].legend(ncol=3)
    ax2[1].legend(ncol=3)
    ax2[0].grid()
    ax2[1].grid()
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    fig2.savefig(os.path.join(SAVEFOLDER,str(int(figSaveCounter)).zfill(2)+'_meanTF2.png'), dpi=300,bbox_inches='tight')
    fig2.savefig( os.path.join(SAVEFOLDER,str(int(figSaveCounter)).zfill(2)+'_meanTF2.pdf'), dpi=300,bbox_inches='tight')
    figSaveCounter+=1
    if SHOW:
        fig2.show()
    print("test")



def plotMeanTfsOneFile(datafile,sensorName='0xbccb0000_MPU_9250',numofexpPerLoop=17,loopsPerRepititon=[5,5,5,5,5],repName=['255$^\circ$','45$^\circ$','135$^\circ$','290$^\circ$','200$^\circ$'],lang=LANG,difFills=None):
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
        ax[0].errorbar(freqs*(1+0.002*(loopIDX+1)), ampMean,   yerr=2 * ampSTD, label=r"\textbf{"+repName[loopIDX]+r"}",fmt='o',lw=PLTSCALFACTOR*2,ms=PLTSCALFACTOR*6)
        ax[1].errorbar(freqs*(1+0.002*(loopIDX+1)), phaseMean/np.pi*180, yerr=(2 * phaseSTD)/np.pi*180, label=r"\textbf{"+repName[loopIDX]+"}",fmt='o',lw=PLTSCALFACTOR*2,ms=PLTSCALFACTOR*6)
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
    imagename=str(int(globals()['figSaveCounter'])).zfill(2)+'_meanTFOneFile1'
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    fig.savefig(os.path.join(SAVEFOLDER,imagename+'.png'), dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(SAVEFOLDER, imagename + '.pdf'), dpi=300,bbox_inches='tight')
    globals()['figSaveCounter']+=1
    if SHOW:
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
        ax2[0].errorbar(freqs*(1+0.002*(loopIDX+1)), (ampMean-overallAmpMean)*100,   yerr=2 * ampSTD*100, label=r"\textbf{"+repName[loopIDX]+"}",fmt='o',lw=PLTSCALFACTOR*2,ms=PLTSCALFACTOR*6)
        ax2[1].errorbar(freqs*(1+0.002*(loopIDX+1)), (phaseMean-overallPhaseMean)/np.pi*180, yerr=(2 * phaseSTD)/np.pi*180, label=r"\textbf{"+repName[loopIDX]+"}",fmt='o',lw=PLTSCALFACTOR*2,ms=PLTSCALFACTOR*6)
        loopsProcessed+=loopsInThisBlock
    if lang=='EN':
        ax2[1].set_xlabel(r"\textbf{Frequency in Hz}")
    if lang=='DE':
        ax2[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    if difFills!=None:
        if lang == 'EN':
            ax2[0].fill_between(freqs, -difFills[0], difFills[0],label=r"\textbf{CMC uncer.}",alpha=0.2,color='gray')
            ax2[1].fill_between(freqs, -difFills[1], difFills[1],label=r"\textbf{CMC uncer.}",alpha=0.2,color='gray')
        if lang == 'DE':
            ax2[0].fill_between(freqs, -difFills[0], difFills[0],label=r"\textbf{CMC Unsicherheit}",alpha=0.2,color='gray')
            ax2[1].fill_between(freqs, -difFills[1], difFills[1],label=r"\textbf{CMC Unsicherheit}",alpha=0.2,color='gray')
    ax2[0].set_xscale('log')
    ax2[0].set_ylabel(r"$|S(\omega)|-\overline{|S(\omega)|}$  \textbf{in \%}")
    ax2[1].set_ylabel(r"$\varphi(\omega) -\overline{\varphi(\omega) }$ \textbf{in $^\circ$}")
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
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    fig2.savefig(os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_meanTFOneFile2.png'), dpi=300,bbox_inches='tight')
    fig2.savefig( os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_meanTFOneFile2.pdf'), dpi=300,bbox_inches='tight')
    globals()['figSaveCounter']+=1
    if SHOW:
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




def plotTFCOmparison(dict,lang='DE',uncerType='typeA',titleExpansion='Test',excludeFromMean=[]):
    numOfTfs=len(dict.keys())
    i=0
    for TfDictKey in dict:
        TfDictEntry=dict[TfDictKey]
        testFreqsMean, ampMean, stdAmpWigth, phaseMean, stdPhaseWight=generateTFFromRawData(TfDictEntry['dataFile'],style=TfDictEntry['style'],sensorName=TfDictEntry['sensorName'])

        freqs=testFreqsMean
        if i ==0:
            labels = []
            colors= []
            usedTFSIDX=[]
            labelExpansion = []
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
        colors.append(TfDictEntry['color'])
        if not TfDictKey in excludeFromMean:
            usedTFSIDX.append(True)
            labelExpansion.append('')
        else:
            usedTFSIDX.append(False)
            if lang =='DE':
                labelExpansion.append(r'$^\ast$')
            if lang =='EN':
                labelExpansion.append(r'$^\ast$')
        i=i+1
    ampMean, ampUncer = generateWigthedMeanFromArrays(ampsArray[usedTFSIDX,:], stdAmpWigthArray[usedTFSIDX,:])
    phaseMean, phaseUncer = generateWigthedMeanFromArrays(phaseMeanArray[usedTFSIDX,:], stdPhaseWightArray[usedTFSIDX,:])

    fig,ax=plt.subplots(2, sharex=True)
    for TFIDX in range(numOfTfs):
        ax[0].errorbar(freqs*(1+0.002*(TFIDX +1)), ampsArray[TFIDX,:],   yerr=2 * stdAmpWigthArray[TFIDX, :], label=labels[TFIDX],fmt='o',lw=PLTSCALFACTOR*2,ms=PLTSCALFACTOR*6,alpha=0.8,color=colors[TFIDX])
        ax[1].errorbar(freqs*(1+0.002*(TFIDX +1)), phaseMeanArray[TFIDX,:]/np.pi*180, yerr=(2 * stdPhaseWightArray[TFIDX, :])/np.pi*180, label=labels[TFIDX],fmt='o',lw=PLTSCALFACTOR*2,ms=PLTSCALFACTOR*6,alpha=0.8,color=colors[TFIDX])
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
    ax[0].set_title(r"\textbf{ÜF "+titleExpansion+'}')
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    fig.savefig(os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_TFComparison1.png'), dpi=300,bbox_inches='tight')
    fig.savefig( os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_TFComparison1.pdf'), dpi=300,bbox_inches='tight')
    globals()['figSaveCounter']+=1
    if SHOW:
        fig.show()

    fig2,ax2=plt.subplots(2, sharex=True)
    for TFIDX in range(numOfTfs):
        ax2[0].errorbar(freqs*(1+0.002*(TFIDX +1)), (ampsArray[TFIDX,:]-ampMean)*100,   yerr=2 * stdAmpWigthArray[TFIDX, :]*100, label=labels[TFIDX]+labelExpansion[TFIDX],fmt='o',lw=PLTSCALFACTOR*2,ms=PLTSCALFACTOR*6,alpha=0.8,color=colors[TFIDX])
        ax2[1].errorbar(freqs*(1+0.002*(TFIDX +1)), (phaseMeanArray[TFIDX,:]-phaseMean)/np.pi*180, yerr=(2 * stdPhaseWightArray[TFIDX, :])/np.pi*180, label=labels[TFIDX]+labelExpansion[TFIDX],fmt='o',lw=PLTSCALFACTOR*2,ms=PLTSCALFACTOR*6,alpha=0.8,color=colors[TFIDX])
    if lang=='EN':
        ax2[1].set_xlabel(r"\textbf{Frequency in Hz}")
    if lang=='DE':
        ax2[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax2[0].set_xscale('log')
    ax2[0].set_ylabel(r"$|S(\omega)|-\overline{|S(\omega)|}$  \textbf{in \%}")
    ax2[1].set_ylabel(r"$\varphi(\omega) -\overline{\varphi(\omega) }$ \textbf{in $^\circ$}")
    if not all(usedTFSIDX):
        if lang=='DE':
            ax2[0].set_xlabel(r"$^\ast$ Nicht für das gewichtet Mittel verwendet")
        if lang=='EN':
            ax2[0].set_xlabel(r"$^\ast$ Not used for weighted mean")
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
    ax2[0].set_title(r"\textbf{Abw. vom gewichteten Mittel der ÜF " + titleExpansion + '}')
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    fig2.savefig(os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_TFComparison2.png'), dpi=300,bbox_inches='tight')
    fig2.savefig( os.path.join(SAVEFOLDER,str(int(globals()['figSaveCounter'])).zfill(2)+'_TFComparison2.pdf'), dpi=300,bbox_inches='tight')
    globals()['figSaveCounter']+=1
    if SHOW:
        fig2.show()



if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL,"de_DE.utf8")

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
               'phaseOffset':np.pi,
               'color':tubscolors[1]}
    PTBTFDIct={'style':'PTB',
               'dataFile':PTBdatafile,
               'sensorName':PTBSensorname,
               'phaseOffset':0,
               'color':tubscolors[0]}
    PTBPlattenDIct={'style':'PTB',
               'dataFile':PTBPlattendatafile,
               'sensorName':PTBSensorname,
               'phaseOffset':0,
                'color':tubscolors[4]}

    plotMeanTfsOneFile(PTBPlattendatafile, sensorName=leadSensorname, difFills=[0.1,0.2])
    #testFreqsMean,phaseMean, stdPhaseWight,ampMean,stdAmpWigth=generateTFFromRawData(datafile, sensorName=leadSensorname,style='CEM')

    TFDict={r'\textbf{CEM}':CEMTFDIct,r'\textbf{PTB, erste Messung}':PTBTFDIct}#r'\textbf{PTB, verschiedene Winkel}': PTBPlattenDIct
    

    plotTFCOmparison(TFDict,uncerType='typeA',titleExpansion='MPU9250 Unsicherheit, TypA')
    plotTFCOmparison(TFDict,uncerType='CMC',titleExpansion='MPU9250 Unsicherheit, CMC')

    # testFreqsMean,phaseMean, stdPhaseWight,ampMean,stdAmpWigth=generateTFFromRawData(datafile, sensorName=leadSensorname,style='CEM')
    TFDict = {r'\textbf{CEM}': CEMTFDIct, r'\textbf{PTB, verschiedene Winkel}': PTBPlattenDIct}  #

    plotTFCOmparison(TFDict, uncerType='typeA', titleExpansion='MPU9250 Unsicherheit, TypA')
    plotTFCOmparison(TFDict, uncerType='CMC', titleExpansion='MPU9250 Unsicherheit, CMC')
    # testFreqsMean,phaseMean, stdPhaseWight,ampMean,stdAmpWigth=generateTFFromRawData(datafile, sensorName=leadSensorname,style='CEM')
    TFDict = {r'\textbf{CEM}': CEMTFDIct,r'\textbf{PTB, erste Messung}':PTBTFDIct, r'\textbf{PTB, verschiedene Winkel}': PTBPlattenDIct}  #

    plotTFCOmparison(TFDict, uncerType='typeA', titleExpansion='MPU9250 Unsicherheit, TypA',excludeFromMean=[r'\textbf{PTB, verschiedene Winkel}'])
    plotTFCOmparison(TFDict, uncerType='CMC', titleExpansion='MPU9250 Unsicherheit, CMC',excludeFromMean=[r'\textbf{PTB, verschiedene Winkel}'])

    plotTFCOmparison(TFDict, uncerType='typeA', titleExpansion='MPU9250 Unsicherheit, TypA',excludeFromMean=[r'\textbf{PTB, erste Messung}'])
    plotTFCOmparison(TFDict, uncerType='CMC', titleExpansion='MPU9250 Unsicherheit, CMC',excludeFromMean=[r'\textbf{PTB, erste Messung}'])

    plotRAWTFUncerComps(PTBdatafile, type='Phase', sensorName=PTBSensorname, startIDX=0, stopIDX=17, title='Typ A Unsicherheitskomponenten PTB Messungen, Phase', zoom=False, lang=LANG)
    plotRAWTFUncerComps(PTBdatafile, type='Mag', sensorName=PTBSensorname, startIDX=0, stopIDX=17, title='Typ A Unsicherheitskomponenten PTB Messungen, Magnitude', zoom=False, lang=LANG)
    plotRAWTFUncerComps(CEMdatafile, type='Phase', sensorName=CEMSensorname, startIDX=2, stopIDX=19, title='Typ A Unsicherheitskomponenten CEM Messungen, Phase', zoom=False, lang=LANG)
    plotRAWTFUncerComps(CEMdatafile, type='Mag', sensorName=CEMSensorname, startIDX=2, stopIDX=19, title='Typ A Unsicherheitskomponenten CEM Messungen, Magnitude', zoom=False, lang=LANG)
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
import json
from scipy import stats  ## for Student T Coverragefactor
from scipy.optimize import curve_fit  # for fiting of Groupdelay
from scipy import interpolate  # for 1D amplitude estimation
import numpy as np
import matplotlib.pyplot as plt
import logging


DPI=160
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
def findNearestIDX(array,value):
     idx = (np.abs(array-value)).argmin()
     return idx

class Met4FOFADCCall:
    def __init__(self, Filenames=[None]):

        if Filenames != [None]:
            i = 0
            for CalFile in Filenames:
                print(CalFile)
                if i == 0:
                    with open(CalFile) as json_file:
                        tmp = json.load(json_file)
                    self.metadata = tmp["MeataData"]
                    self.fitResults = tmp["FitResults"]
                    i = i + 1
                    json_file.close()
                else:
                    with open(CalFile) as json_file:
                        tmp = json.load(json_file)
                    if self.metadata["BordID"] == tmp["MeataData"]["BordID"]:
                        for Channel in tmp["FitResults"]:
                            for Freqs in tmp["FitResults"][Channel]:
                                self.fitResults[Channel][Freqs] += tmp["FitResults"][
                                    Channel
                                ][Freqs]
                    else:
                        raise RuntimeWarning(
                            "BoardIDs"
                            + self.metadata["BordID"]
                            + "and"
                            + tmp.metadata["BordID"]
                            + "DO Not Match ignoring File"
                            + CalFile
                        )
                    i = i + 1
                    json_file.close()
            self.TransferFunctions = {}
            for Channels in self.fitResults:
                self.GetTransferFunction(Channels)

    def GetTransferFunction(self, Channel):
        FreqNum = len(self.fitResults[Channel].keys())
        Transferfunction = {
            "Frequencys": np.zeros(FreqNum),
            "AmplitudeCoefficent": np.zeros(FreqNum),
            "AmplitudeCoefficentUncer": np.zeros(FreqNum),
            "Phase": np.zeros(FreqNum),
            "PhaseUncer": np.zeros(FreqNum),
            "N": np.zeros(FreqNum),
        }
        i = 0
        for freq in self.fitResults[Channel].keys():
            Transferfunction["Frequencys"][i] = freq
            Transferfunction["N"][i] = N = len(
                [d["Phase"] for d in self.fitResults[Channel][freq]]
            )
            StudentTCoverageFactor95 = stats.t.ppf(1 - 0.025, N)
            Transferfunction["AmplitudeCoefficent"][i] = np.mean(
                [d["Amplitude"] for d in self.fitResults[Channel][freq]]
            )
            Transferfunction["AmplitudeCoefficentUncer"][i] = (
                np.std([d["Amplitude"] for d in self.fitResults[Channel][freq]])
                * StudentTCoverageFactor95
            )
            Transferfunction["Phase"][i] = np.mean(
                [d["Phase"] for d in self.fitResults[Channel][freq]]
            )
            Transferfunction["PhaseUncer"][i] = (
                np.std([d["Phase"] for d in self.fitResults[Channel][freq]])
                * StudentTCoverageFactor95
            )

            i = i + 1
        self.TransferFunctions[Channel] = Transferfunction
        return Transferfunction

    def PlotTransferfunction(
        self,
        Channel,
        PlotType="lin",
        interpolSteps=1000,
        fig=None,
        ax=[None, None],
        LabelExtension="",
        TitleExtension="",
        saveFigName=None,
        startStopFreq=None,
        lang='EN'#'EN' or 'DE'
    ):

        BoardID = self.metadata["BordID"]
        tf = self.GetTransferFunction(Channel)
        if startStopFreq==None:
            plotStartFreq =np.min(tf["Frequencys"])
            plotStopFreq =np.max(tf["Frequencys"])
            plotStatIDX=0
            plotStopIDX=tf["Frequencys"].size
        else:
            plotStartFreq =np.min(startStopFreq[0])
            plotStopFreq =np.max(startStopFreq[1])
            plotStatIDX =findNearestIDX(tf["Frequencys"],startStopFreq[0])
            plotStopIDX =findNearestIDX(tf["Frequencys"],startStopFreq[1])
        if PlotType == "log":
            XInterPol = np.power(
                10,
                np.linspace(
                    np.log10(plotStartFreq),
                    np.log10(plotStopFreq),
                    interpolSteps,
                ),
            )
        else:
            XInterPol = np.linspace(
                np.min(plotStartFreq), np.max(plotStopFreq), interpolSteps
            )
        interPolAmp = np.zeros(interpolSteps)
        interPolAmpErrMin = np.zeros(interpolSteps)
        interPolAmpErrMax = np.zeros(interpolSteps)
        interPolPhase = np.zeros(interpolSteps)
        interPolPhaseErrMin = np.zeros(interpolSteps)
        interPolPhaseErrMax = np.zeros(interpolSteps)
        for i in range(interpolSteps):
            tmp = self.getNearestTF(Channel, XInterPol[i])
            interPolAmp[i] = tmp["AmplitudeCoefficent"]
            interPolAmpErrMin[i] = interPolAmp[i] - tmp["AmplitudeCoefficentUncer"]
            interPolAmpErrMax[i] = interPolAmp[i] + tmp["AmplitudeCoefficentUncer"]
            interPolPhase[i] = tmp["Phase"]
            interPolPhaseErrMin[i] = interPolPhase[i] - tmp["PhaseUncer"]
            interPolPhaseErrMax[i] = interPolPhase[i] + tmp["PhaseUncer"]
        if fig == None and ax == [None, None]:
            Fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True,dpi=DPI)
            Fig.set_size_inches(14, 7, forward=True)
        else:
            Fig = fig
            ax1 = ax[0]
            ax2 = ax[1]
        if PlotType == "log":
            ax1.set_xscale("log")
            ax2.set_xscale("log")
        if lang=='EN':
            labelInterpol=r"\textbf{Interpolated}"
            lableMeasVals=r"\textbf{Mesured Values"
            axisCapRelMag=r"Relative magnitude $|S|$"
            labelFreq=r"\textbf{Frequency $f$ in Hz"
            title="\textbf{Transfer function of "+ str(Channel)+ " of Board with ID"+ hex(int(BoardID/65536))+ TitleExtension
        elif 'DE':
            labelInterpol = r"\textbf{Interpoliert}"
            lableMeasVals = r"\textbf{Messwerte}"
            axisCapRelMag = r"\textbf{Relative Magnitude} $|S|$"
            labelFreq = r"\textbf{Frequenz }$f$ \textbf{in Hz}"
            labelPhase = r"\textbf{Phase $\varphi$ in °"
            title=r"\textbf{Transferfunction  "+ str(Channel)+ " des Boards mit der ID "+ hex(int(BoardID/65536))+'}'+ TitleExtension
        Fig.suptitle(title)
        ax1.plot(XInterPol, interPolAmp,ls='dotted',label=labelInterpol + LabelExtension)
        lastcolor = ax1.get_lines()[-1].get_color()
        ax1.fill_between(
            XInterPol, interPolAmpErrMin, interPolAmpErrMax, alpha=0.3, color=lastcolor
        )
        ax1.errorbar(
            tf["Frequencys"][plotStatIDX:plotStopIDX],
            tf["AmplitudeCoefficent"][plotStatIDX:plotStopIDX],
            yerr=tf["AmplitudeCoefficentUncer"][plotStatIDX:plotStopIDX],
            ls='none',
            #markersize=4,
            label=lableMeasVals + LabelExtension,
            #uplims=True,
            #lolims=True,
            color=lastcolor,
        )
        ax1.set_ylabel(axisCapRelMag)
        ax1.grid(True)
        ax2.plot(
            XInterPol,
            interPolPhase / np.pi * 180,
            ls='dotted',
            label=labelInterpol + LabelExtension,
        )
        ax2.fill_between(
            XInterPol,
            interPolPhaseErrMin / np.pi * 180,
            interPolPhaseErrMax / np.pi * 180,
            alpha=0.3,
            color=lastcolor,
        )
        ax2.errorbar(
            tf["Frequencys"][plotStatIDX:plotStopIDX],
            tf["Phase"][plotStatIDX:plotStopIDX] / np.pi * 180,
            yerr=tf["PhaseUncer"][plotStatIDX:plotStopIDX] / np.pi * 180,
            ls='none',
            #markersize=3,
            label=lableMeasVals + LabelExtension,
            #uplims=True,
            #lolims=True,
            color=lastcolor,
        )
        ax2.set_xlabel(labelFreq)
        ax2.set_ylabel(labelPhase)
        ax2.grid(True)
        ax1.legend(numpoints=1, ncol=3)
        ax2.legend(numpoints=1, ncol=3)
        if saveFigName!=None:
            Fig.savefig(saveFigName+".svg",dpi=Fig.dpi, bbox_inches='tight', pad_inches=0.5)
            Fig.savefig(saveFigName + ".png",dpi=Fig.dpi, bbox_inches='tight', pad_inches=0.5)
        Fig.show()
        return Fig, [ax1, ax2]

    def getNearestTF(self, Channel, freq):
        Freqs = self.TransferFunctions[Channel]["Frequencys"]
        testFreqIDX = np.argmin(abs(Freqs - freq))
        if (
            Freqs[testFreqIDX] - freq == 0
        ):  # ok we hit an calibrated point no need to interpolate
            return {
                "Frequency": self.TransferFunctions[Channel]["Frequencys"][testFreqIDX],
                "AmplitudeCoefficent": self.TransferFunctions[Channel][
                    "AmplitudeCoefficent"
                ][testFreqIDX],
                "AmplitudeCoefficentUncer": self.TransferFunctions[Channel][
                    "AmplitudeCoefficentUncer"
                ][testFreqIDX],
                "Phase": self.TransferFunctions[Channel]["Phase"][testFreqIDX],
                "PhaseUncer": self.TransferFunctions[Channel]["PhaseUncer"][
                    testFreqIDX
                ],
                "N": self.TransferFunctions[Channel]["N"][testFreqIDX],
            }
        else:
            # interpolate
            A, AErr = self.getInterPolatedAmplitude(Channel, freq)
            P, PErr = self.getInterPolatedPhase(Channel, freq)
            return {
                "Frequency": freq,
                "AmplitudeCoefficent": np.asscalar(A),
                "AmplitudeCoefficentUncer": np.asscalar(AErr),
                "Phase": np.asscalar(P),
                "PhaseUncer": np.asscalar(PErr),
                "N": 0,
            }

    def __getitem__(self, key):
        if len(key) == 4:
            return self.TransferFunctions[key]
        if len(key) == 2:
            return self.getNearestTF(key[0], key[1])
        else:
            raise ValueError(
                "Invalide Key:  > "
                + str(key)
                + " <Use either [Channel] eg ['ADC1] or [Channel,Frequency] eg ['ADC1',1000]  as key "
            )

    def getGroupDelay(self, Channel):
        freqs = self.TransferFunctions[Channel]["Frequencys"]
        phases = self.TransferFunctions[Channel]["Phase"]
        phaseUncer = self.TransferFunctions[Channel]["PhaseUncer"]
        popt, pcov = curve_fit(
            PhaseFunc, freqs, phases, sigma=phaseUncer, absolute_sigma=True
        )
        return [popt, pcov]

    def getInterPolatedAmplitude(self, Channel, freq):
        Freqs = self.TransferFunctions[Channel]["Frequencys"]
        Ampls = self.TransferFunctions[Channel]["AmplitudeCoefficent"]
        AmplErr = self.TransferFunctions[Channel]["AmplitudeCoefficentUncer"]
        testFreqIDX = np.argmin(abs(Freqs - freq))
        DeltaInterpolIDX = 0
        if freq - Freqs[testFreqIDX] < 0:
            DeltaInterpolIDX = -1
        if freq - Freqs[testFreqIDX] > 0:
            DeltaInterpolIDX = 1
        if testFreqIDX + DeltaInterpolIDX < 0:
            assert RuntimeWarning(
                str(freq)
                + " is to SMALL->Extrapolation is not recomended! minimal Frequency is "
                + str(Freqs[0])
            )
            return [Ampls[0], AmplErr[0]]
        if testFreqIDX + DeltaInterpolIDX >= Freqs.size:
            raise ValueError(
                str(freq)
                + " is to BIG->Extrapolation not supported! maximal Frequency is "
                + str(Freqs[-1])
            )
        if DeltaInterpolIDX == 0:
            return [
                self.TransferFunctions[Channel]["AmplitudeCoefficent"][testFreqIDX],
                self.TransferFunctions[Channel]["AmplitudeCoefficentUncer"][
                    testFreqIDX
                ],
            ]
        elif DeltaInterpolIDX == -1:
            IDX = [testFreqIDX - 1, testFreqIDX]
            x = Freqs[IDX]
            A = Ampls[IDX]
            AErr = AmplErr[IDX]
        elif DeltaInterpolIDX == 1:
            IDX = [testFreqIDX, testFreqIDX + 1]
            x = Freqs[IDX]
            A = Ampls[IDX]
            AErr = AmplErr[IDX]
        fA = interpolate.interp1d(x, A)
        fAErr = interpolate.interp1d(x, AErr)
        print(
            "Interpolateded transferfunction for Channel "
            + str(Channel)
            + "at Freq "
            + str(freq)
        )  # will not print anything
        return [fA(freq), fAErr(freq)]

    def getInterPolatedPhase(self, Channel, freq):
        Freqs = self.TransferFunctions[Channel]["Frequencys"]
        Phases = self.TransferFunctions[Channel]["Phase"]
        PhasesErr = self.TransferFunctions[Channel]["PhaseUncer"]
        testFreqIDX = np.argmin(abs(Freqs - freq))
        DeltaInterpolIDX = 0
        if freq - Freqs[testFreqIDX] < 0:
            DeltaInterpolIDX = -1
        if freq - Freqs[testFreqIDX] > 0:
            DeltaInterpolIDX = 1
        if testFreqIDX + DeltaInterpolIDX < 0:
            assert RuntimeWarning(
                str(freq)
                + " is to SMALL->Extrapolation is not recomended! minimal Frequency is "
                + str(Freqs[0])
            )
            return [Phases[0], PhasesErr[0]]
        if testFreqIDX + DeltaInterpolIDX >= Freqs.size:
            raise ValueError(
                "Extrapolation not supported! maximal Frequency is" + Freqs[-1]
            )
        if DeltaInterpolIDX == 0:
            return [
                self.TransferFunctions[Channel]["AmplitudeCoefficent"][testFreqIDX],
                self.TransferFunctions[Channel]["AmplitudeCoefficentUncer"][
                    testFreqIDX
                ],
            ]
        elif DeltaInterpolIDX == -1:
            IDX = [testFreqIDX - 1, testFreqIDX]
        elif DeltaInterpolIDX == 1:
            IDX = [testFreqIDX, testFreqIDX + 1]
        x = Freqs[IDX]
        P = Phases[IDX]
        PErr = PhasesErr[IDX]
        fP = interpolate.interp1d(x, P)
        fPErr = interpolate.interp1d(x, PErr)
        logging.info(
            "Interpolateded transferfunction for Channel "
            + str(Channel)
            + "at Freq "
            + str(freq)
        )  # will not print anything
        return [fP(freq), fPErr(freq)]

def jsonsplitterFortestVoltages(jsonFile):
    ouputDicts=[{},{},{}]
    outIDXFromVoltage={19.5:0,1.95:1,0.195:2}
    with open(jsonFile) as json_file:
        tmp = json.load(json_file)
    metadata = tmp["MeataData"]
    fitResults = tmp["FitResults"]
    ouputDicts[0]["MeataData"]=metadata
    ouputDicts[1]["MeataData"] = metadata
    ouputDicts[2]["MeataData"] = metadata
    ouputDicts[0]["FitResults"]={'ADC1':{},'ADC2':{},'ADC3':{}}
    ouputDicts[1]["FitResults"]={'ADC1':{},'ADC2':{},'ADC3':{}}
    ouputDicts[2]["FitResults"]={'ADC1':{},'ADC2':{},'ADC3':{}}
    for ADCname in fitResults.keys():
        for freq in fitResults[ADCname].keys():
            for datapointdict in fitResults[ADCname][freq]:
                examp=datapointdict['TestAmplVPP.']
                try:
                    ouputDicts[outIDXFromVoltage[examp]]["FitResults"][ADCname][freq].append(datapointdict)
                except:
                    ouputDicts[outIDXFromVoltage[examp]]["FitResults"][ADCname][freq]=[datapointdict]
    #caldata['FitResults']['ADC1']['1.0'][0]
    #{'Freq': 1.0, 'Amplitude': 1.0029198553691523, 'Phase': 8.555608677440536e-06, 'TestAmplVPP.': 19.5}
    return ouputDicts





if __name__ == "__main__":
    ADCTF19V5 =  Met4FOFADCCall(['../cal_data/1FE4_AC_CAL/200320_1FE4_ADC123_3CYCLES_19V5_1HZ_1MHZ.json'])
    ADCTF1V95 =  Met4FOFADCCall(['../cal_data/1FE4_AC_CAL/200320_1FE4_ADC123_3CYCLES_1V95_1HZ_1MHZ.json'])
    ADCTF0V195 = Met4FOFADCCall(['../cal_data/1FE4_AC_CAL/200320_1FE4_ADC123_3CYCLES_V195_1HZ_1MHZ.json'])
    Fig, axs=ADCTF0V195.PlotTransferfunction('ADC1',interpolSteps=100,PlotType="log",
                                            LabelExtension=r'~0.195~V',lang='DE')
    ADCTF1V95.PlotTransferfunction('ADC1',fig=Fig,ax=axs, interpolSteps=100, PlotType="log",
                                   LabelExtension=r'~1.95~V',lang='DE',saveFigName='ADCTF')
    ADCTF19V5.PlotTransferfunction('ADC1', fig=Fig, ax=axs, interpolSteps=100, PlotType="log",
                                   LabelExtension=r'~19.5~V', lang='DE', saveFigName='ADCTF')

    #Fig2, axs2=ADCTFFull.PlotTransferfunction('ADC1',interpolSteps=100,PlotType="log",LabelExtension=r'~19.5~V \& 1.95~V \& 0.195~V',lang='DE',startStopFreq=[1,10000])
    #ADCTF.PlotTransferfunction('ADC1',fig=Fig2,ax=axs2, interpolSteps=100, PlotType="log",LabelExtension=r'~19.5~V' ,lang='DE',saveFigName='ADCTFZoom',startStopFreq=[1,10000])

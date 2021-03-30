import json
from scipy import stats  ## for Student T Coverragefactor
from scipy.optimize import curve_fit  # for fiting of Groupdelay
from scipy import interpolate  # for 1D amplitude estimation
import numpy as np
import matplotlib.pyplot as plt


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
    ):
        BoardID = self.metadata["BordID"]
        tf = self.GetTransferFunction(Channel)
        if PlotType == "log":
            XInterPol = np.power(
                10,
                np.linspace(
                    np.log10(np.min(tf["Frequencys"])),
                    np.log10(np.max(tf["Frequencys"]) - 1),
                    interpolSteps,
                ),
            )
        else:
            XInterPol = np.logspace(
                np.min(tf["Frequencys"]), np.max(tf["Frequencys"]), interpolSteps
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
            Fig, (ax1, ax2) = plt.subplots(2, 1)
        else:
            Fig = fig
            ax1 = ax[0]
            ax2 = ax[1]
        if PlotType == "log":
            ax1.set_xscale("log")
            ax2.set_xscale("log")
        ax1.plot(XInterPol, interPolAmp, label="Interpolated" + LabelExtension)
        lastcolor = ax1.get_lines()[-1].get_color()
        ax1.fill_between(
            XInterPol, interPolAmpErrMin, interPolAmpErrMax, alpha=0.3, color=lastcolor
        )
        ax1.errorbar(
            tf["Frequencys"],
            tf["AmplitudeCoefficent"],
            yerr=tf["AmplitudeCoefficentUncer"],
            fmt="o",
            markersize=4,
            label="Mesured Values" + LabelExtension,
            uplims=True,
            lolims=True,
            color=lastcolor,
        )
        Fig.suptitle(
            "Transfer function of "
            + str(Channel)
            + " of Board with ID"
            + hex(BoardID)
            + TitleExtension
        )
        ax1.set_ylabel("Relative magnitude $|S|$")
        ax1.grid(True)
        ax2.plot(
            XInterPol,
            interPolPhase / np.pi * 180,
            label="Interpolated" + LabelExtension,
        )
        ax2.fill_between(
            XInterPol,
            interPolPhaseErrMin / np.pi * 180,
            interPolPhaseErrMax / np.pi * 180,
            alpha=0.3,
            color=lastcolor,
        )
        ax2.errorbar(
            tf["Frequencys"],
            tf["Phase"] / np.pi * 180,
            yerr=tf["PhaseUncer"] / np.pi * 180,
            fmt="o",
            markersize=3,
            label="Mesured Values" + LabelExtension,
            uplims=True,
            lolims=True,
            color=lastcolor,
        )
        ax2.set_xlabel(r"Frequency $f$ in Hz")
        ax2.set_ylabel(r"Phase $\Delta\varphi$ in °")
        ax2.grid(True)
        ax1.legend(numpoints=1, fontsize=8, ncol=3)
        ax2.legend(numpoints=1, fontsize=8, ncol=3)
        plt.show()
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

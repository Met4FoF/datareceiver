# import h5py
# TODO implent proper multi threaded hdf file handling
import h5pickle as h5py  # be carefull opining more than 100 files in a row kills first file handle !!!
import h5py as h5py_plain
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import time
import multiprocessing
import sys
import time
import sinetools.SineTools as st

# import yappi
import warnings

import os
import shutil
from pathlib import Path
from adccaldata import Met4FOFADCCall
from scipy.optimize import curve_fit  # for fiting of Groupdelay
from scipy import interpolate  # for 1D amplitude estimation

import deepdish as dd  # for dict to hdf serialisation

# used only in transfercalculation because of performance reasonsfrom uncertainties import ufloat
# >>> from uncertainties.umath import *  # sin(), etc.
from uncertainties import ufloat
from uncertainties.umath import *  # sin(), etc.
from met4fofhdftools import add1dsinereferencedatatohdffile

# from met4fofhdftools import addadctransferfunctiontodset
from met4fofhdftools import (
    uncerval,getRAWTFFromExperiemnts
)  # uncerval = np.dtype([("value", np.float), ("uncertainty", np.float)])


def ufloattouncerval(ufloat):
    result = np.empty([1], dtype=uncerval)
    result["value"] = ufloat.n
    result["uncertainty"] = ufloat.s
    return result


# plt.rc('font', family='serif')
# plt.rc('text', usetex=True)
PLTSCALFACTOR = 2
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


def getplotableunitstring(unitstr, Latex=False):
    if not Latex:
        convDict = {
            "\\degreecelsius": "°C",
            "\\degree": "°",
            "\\micro\\tesla": "µT",
            "\\radian\\second\\tothe{-1}": "rad/s",
            "\\metre\\second\\tothe{-2}": "m/s^2",
            "\\metre\\second\\tothe{-1}": "m/s",
            "\\volt": "v",
            "\\hertz": "Hz",
        }
    else:
        convDict = {
            "\\degreecelsius": "$^\circ C$",
            "\\degree": "$^\circ$",
            "\\micro\\tesla": "$\micro T$",
            "\\radian\\second\\tothe{-1}": "$\\frac{rad}{s}$",
            "\\metre\\second\\tothe{-2}": "$\\frac{m}{s^2}",
            "\\metre\\second\\tothe{-1}": "$\\frac{m}{s}",
            "\\volt": "v",
            "\\hertz": "Hz",
        }
    try:
        result = convDict[unitstr]
    except KeyError:
        result = unitstr
    return result


# code from https://stackoverflow.com/questions/23681948/get-index-of-closest-value-with-binary-search  answered Dec 6 '14 at 23:51 Yaguang
def binarySearch(data, val):
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break

        # check if data[mid] is closer to val than data[best_ind]
        dtype = data.dtype
        if data.dtype == np.uint64:
            absmidint = abs(data[mid].astype(np.int64) - val.astype(np.int64))
            absbestind = abs(data[best_ind].astype(np.int64) - val.astype(np.int64))
        else:
            absmidint = abs(data[mid] - val)
            absbestind = abs(data[best_ind] - val)
        if absmidint < absbestind:
            best_ind = mid

    return best_ind


class hdfmet4fofdatafile:
    def __init__(self, hdffile):
        self.hdffile = hdffile
        self.senorsnames = list(self.hdffile["RAWDATA"].keys())
        self.sensordatasets = {}
        for name in self.senorsnames:
            datasets = list(self.hdffile["RAWDATA/" + name].keys())
            keystocheckandremove = [
                "Absolutetime",
                "Absolutetime_uncertainty",
                "Sample_number",
            ]
            for key in keystocheckandremove:
                try:
                    datasets.remove(key)
                except ValueError:
                    raise RuntimeWarning(
                        str(name)
                        + " doese not contain "
                        + str(key)
                        + " dataset is maybe corrupted!"
                    )
            self.sensordatasets[name] = datasets
        print("INIT DONE")
        print("RAW DataGroups are " + str(self.senorsnames))
        print("RAW Datasets are " + str(self.sensordatasets))

    def calcblockwiesestd(self, dataset, blocksize=100):
        # start = time.time()
        blockcount = int(np.floor(dataset.size / blocksize))
        std = np.zeros(blockcount)
        split = np.split(dataset[: blocksize * blockcount], blockcount, axis=0)
        std = np.std(split, axis=1)
        # end = time.time()
        # print("bwstd for dataset " + str(dataset) + "took " + str(end - start) + " secs")
        return std

    def detectnomovment(
        self, datahdfpath, timehdfpath, treshold=0.05, blocksinrow=5, blocksize=100
    ):
        tmpData = np.squeeze(self.hdffile[datahdfpath])
        tmpTime = np.squeeze(self.hdffile[timehdfpath])  # fetch data from hdffile
        mag = np.linalg.norm(tmpData, axis=0)
        std = self.calcblockwiesestd(mag, blocksize=blocksize)
        wasvalide = 0
        nomovementtidx = []
        nomovementtimes = []
        for i in range(std.size):
            if std[i] < treshold:
                wasvalide = wasvalide + 1
            else:
                if wasvalide > blocksinrow:
                    startidx = (i - wasvalide) * blocksize
                    stopidx = (i) * blocksize
                    if stopidx - startidx < 0:
                        print("index error")
                    if tmpTime[stopidx] - tmpTime[startidx] < 0:
                        print("timing error")
                    nomovementtidx.append([startidx, stopidx])
                    nomovementtimes.append([tmpTime[startidx], tmpTime[stopidx]])
                wasvalide = 0
        nomovementidx = np.array(nomovementtidx)
        return nomovementidx, np.array(nomovementtimes)

    def detectmovment(
        self,
        datahdfpath,
        timehdfpath,
        treshold=0.5,
        blocksinrow=5,
        blocksize=100,
        plot=False,
    ):
        tmpData = np.squeeze(self.hdffile[datahdfpath])
        tmpTime = np.squeeze(self.hdffile[timehdfpath])  # fetch data from hdffile
        mag = np.linalg.norm(tmpData, axis=0)
        std = self.calcblockwiesestd(mag, blocksize=blocksize)
        wasvalide = 0
        movementtidx = []
        movementtimes = []
        for i in range(std.size):
            if std[i] > treshold:
                wasvalide = wasvalide + 1
            else:
                if wasvalide > blocksinrow:
                    startidx = (i - wasvalide) * blocksize
                    stopidx = (i) * blocksize
                    movementtidx.append([startidx, stopidx])
                    movementtimes.append([tmpTime[startidx], tmpTime[stopidx]])
                wasvalide = 0
        movementidx = np.array(movementtidx)
        if plot:
            fig, ax = plt.subplots()
            reltime = (tmpTime - tmpTime[0]) / 1e9
            blocktime = reltime[
                0::blocksize
            ]  # cut one block out--> garden fence problem
            ax.plot(blocktime[: std.size], std, label="Data")
            for i in np.arange(len(movementtimes)):
                relmovementimes = (movementtimes[i] - tmpTime[0]) / 1e9
                ax.plot(relmovementimes, np.array([treshold, treshold]), label=str(i))
            fig.show()
        return movementidx, np.array(movementtimes)

    def getnearestidxs(self, sensorname, time):
        absolutimegroupname = "RAWDATA/" + sensorname + "/" + "Absolutetime"
        absolutetimes = np.squeeze(self.hdffile[absolutimegroupname])
        idxs = np.copy(time)
        with np.nditer(idxs, op_flags=["readwrite"]) as it:
            for x in it:
                x[...] = binarySearch(absolutetimes, x)
        return idxs

    # TODO move to hdf datafile class
    def addrawtftohdffromexpreiments(self, experimentgroup,
                                     sensor,
                                     numeratorQuantity="Acceleration",
                                     denominatorQuantity="Acceleration",
                                     type="1D_Z",
                                     miscdict={'scale':'Excitation_frequency'},
                                     attrsdict={
            'Phase': {'Unit': '\\radian',
                      'Physical_quantity': ['Phase response'],
                      'Uncertainty_type': "95% coverage gausian"},
            'Magnitude': {'Unit': '\\one',
                          'Unit_numerator': '\\metre\\second\\tothe{-2}',
                          'Unit_denominator': '\\metre\\second\\tothe{-2}',
                          'Physical_quantity': ['Magnitude response'],
                          'Uncertainty_type': "95% coverage gausian"},
            'Frequency': {'Unit': '\\hertz',
                          'Physical_quantity': ['Frequency used by sine aproximation'],
                          'Uncertainty_type': "95% coverage gausian"},
            'Excitation_frequency': {'Unit': '\\hertz',
                                     'Physical_quantity': ['Nominal frequency'],
                                     'Uncertainty_type': "95% coverage gausian"},
            'Excitation_amplitude': {'Unit': '\\metre\\second\\tothe{-2}',
                                     'Physical_quantity': ['Excitation Amplitude'],
                                     'Uncertainty_type': "95% coverage gausian"}}):
        rawtf = getRAWTFFromExperiemnts(experimentgroup,
                                        sensor,
                                        numeratorQuantity,
                                        denominatorQuantity,
                                        type)
        try:
            RAWTRANSFERFUNCTIONGROUP = self.hdffile["RAWTRANSFERFUNCTION"]
        except KeyError:
            RAWTRANSFERFUNCTIONGROUP = self.hdffile.create_group("RAWTRANSFERFUNCTION")
        try:
            SensorRAWTFGROUP = RAWTRANSFERFUNCTIONGROUP[sensor]#->RAWTRANSFERFUNCTION/0x1FE40000_MPU9250
        except KeyError:
            SensorRAWTFGROUP = RAWTRANSFERFUNCTIONGROUP.create_group(sensor)
        try:
            SensorRAWTFGROUPNUM = SensorRAWTFGROUP[numeratorQuantity]#->RAWTRANSFERFUNCTION/0x1FE40000_MPU9250/Acceleration
        except KeyError:
            SensorRAWTFGROUPNUM = SensorRAWTFGROUP.create_group(numeratorQuantity)
        try:
            SensorRAWTFGROUPNUM = SensorRAWTFGROUP[numeratorQuantity]#->RAWTRANSFERFUNCTION/0x1FE40000_MPU9250/Acceleration
        except KeyError:
            SensorRAWTFGROUPNUM = SensorRAWTFGROUP.create_group(numeratorQuantity)
        try:
            RAWTFGROUP = SensorRAWTFGROUPNUM[denominatorQuantity]#->RAWTRANSFERFUNCTION/0x1FE40000_MPU9250/Acceleration/Acceleration
        except KeyError:
            RAWTFGROUP = SensorRAWTFGROUPNUM.create_group(denominatorQuantity)
        print(attrsdict)
        for tfcomponentkey in rawtf.keys():
            group=RAWTFGROUP.create_group(tfcomponentkey)
            try:
                attrs=attrsdict[tfcomponentkey]
                for attrkeys in attrs.keys():
                    group.attrs[attrkeys]=attrs[attrkeys]
            except KeyError:
                warnings.warn("for Dataset "+str(tfcomponentkey)+" no attrs are given in attrsdict")
            for vukey in rawtf[tfcomponentkey].keys():# loop over value and uncertanies
                dset=group.create_dataset(vukey,rawtf[tfcomponentkey][vukey].shape, dtype="float64")
                dset[:]=rawtf[tfcomponentkey][vukey]
        scaledataset=RAWTFGROUP[miscdict['scale']]['value']#the values are the scale
        scaledataset.make_scale("Frequency")
        for tfcomponentkey in rawtf.keys():
            if tfcomponentkey!=miscdict['scale']:
                group=RAWTFGROUP[tfcomponentkey]
                group['value'].dims[0].label = "Frequency"
                group['value'].dims[0].attach_scale(scaledataset)
                group['uncertainty'].dims[0].label = "Frequency"
                group['uncertainty'].dims[0].attach_scale(scaledataset)
        print("Done")



class transferfunktion:
    def __init__(self, tfgroup):
        self.group = tfgroup

    def getNearestTF(self, channel, freq):
        Freqs = self.group["Frequency"]['value']
        FreqsUncer = self.group["Frequency"]['uncertainty']
        testFreqIDX = np.argmin(abs(Freqs - freq))
        if (
            Freqs[testFreqIDX] - freq == 0
        ):  # ok we hit an calibrated point no need to interpolate
            return {
                "frequency": ufloat(Freqs[testFreqIDX],FreqsUncer[testFreqIDX]),
                "Magnitude": ufloat(self.group["Magnitude"]['value'][channel][testFreqIDX],self.group["Magnitude"]['uncertainty'][channel][testFreqIDX]),
                "Phase": ufloat(self.group["Phase"]['value'][channel][testFreqIDX],self.group["Phase"]['uncertainty'][channel][testFreqIDX]),
                "N": ufloat(self.group["Phase"]['value'][channel][testFreqIDX],self.group["Phase"]['uncertainty'][channel][testFreqIDX])
            }
        else:
            # interpolate
            A = self.getInterPolated(channel, freq,'Magnitude')
            P = self.getInterPolated(channel, freq,'Phase')
            #TODO add interpolation with frequency 'uncertainty'
            return {"Frequency": ufloat(freq,np.NaN), "Magnitude": A, "Phase": P, "N": ufloat(np.NaN,np.NaN)}

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

    # def getGroupDelay(self, Channel):
    #    freqs = self.TransferFunctions[Channel]['Frequencys']
    #    phases = self.TransferFunctions[Channel]['Phase']
    #    phaseUncer = self.TransferFunctions[Channel]['PhaseUncer']
    #    popt, pcov = curve_fit(PhaseFunc, freqs, phases, sigma=phaseUncer, absolute_sigma=True)
    #    return [popt, pcov]

    def getInterPolated(self, channel, freq,key):
        Freqs = self.group["Frequency"]['value'][:]
        FreqsUncer = self.group["Frequency"]['uncertainty'][:]
        vals = self.group[key]['value'][channel, :]
        valsUncer = self.group[key]['uncertainty'][channel, :]
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
            return vals[0]
        if testFreqIDX + DeltaInterpolIDX >= Freqs.size:
            raise ValueError(
                str(freq)
                + " is to BIG->Extrapolation not supported! maximal Frequency is "
                + str(Freqs[-1])
            )
        if DeltaInterpolIDX == 0:
            return vals[testFreqIDX]
        elif DeltaInterpolIDX == -1:
            IDX = [testFreqIDX - 1, testFreqIDX]
        elif DeltaInterpolIDX == 1:
            IDX = [testFreqIDX, testFreqIDX + 1]
        x = Freqs[IDX]
        if not (np.isnan(FreqsUncer[IDX])).all():
                warnings.warn(RuntimeWarning("Interpolation with frequency uncertantiy not supported by getInterPolatedAmplitude Fix this"))
        A = vals[IDX]
        AErr = valsUncer[IDX]
        fA = interpolate.interp1d(x, A)
        fAErr = interpolate.interp1d(x, AErr)
        print(
            "Interpolateded transferfunction for Channel "
            + str(channel)
            + "at Freq "
            + str(freq)
        )  # will not print anything
        return ufloat(fA(freq), fAErr(freq))


class experiment:
    def __init__(self, hdfmet4fofdatafile, times, experemientTypeName, experiementID):
        self.params = {
            "experemientTypeName": experemientTypeName,
        }
        self.met4fofdatafile = hdfmet4fofdatafile
        self.datafile = self.met4fofdatafile.hdffile
        self.experiemntID = experiementID
        self.timepoints = times
        self.idxs = {}
        self.data = (
            {}
        )  # all elements in this dict are new an will be saved in the hdf file
        self.runtimeData = {}  # all data here will NOT saved into the hdf file
        self.flags = {"saved_to_disk":False}

        for name in self.met4fofdatafile.senorsnames:
            self.idxs[name] = self.met4fofdatafile.getnearestidxs(name, self.timepoints)
            if self.idxs[name][1] - self.idxs[name][0] == 0:
                raise ValueError("EMPTY DATA SET")
            self.data[name] = {}
            self.runtimeData[name] = {}
            for dataset in self.met4fofdatafile.sensordatasets[name]:
                self.data[name][dataset] = {}
                self.runtimeData[name][dataset] = {}
        print("EX base class Init done")

    def plotall(self, absolutetime=False):
        cols = len(self.met4fofdatafile.sensordatasets)  # one colum for every sensor
        datasetspersensor = []
        for sensors in self.met4fofdatafile.sensordatasets:
            datasetspersensor.append(len(self.met4fofdatafile.sensordatasets[sensors]))
        rows = np.max(datasetspersensor)
        fig, axs = plt.subplots(cols, rows, sharex="all")
        icol = 0
        for sensor in self.met4fofdatafile.sensordatasets:
            irow = 0
            idxs = self.idxs[sensor]
            axs[icol, 0].annotate(
                sensor.replace("_", " "),
                xy=(0, 0.5),
                xytext=(-axs[icol, 0].yaxis.labelpad - 5, 0),
                xycoords=axs[icol, 0].yaxis.label,
                textcoords="offset points",
                size="large",
                ha="right",
                va="center",
                rotation=90,
            )

            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                dsetattrs = self.datafile["RAWDATA/" + sensor + "/" + dataset].attrs
                time = self.datafile["RAWDATA/" + sensor + "/" + "Absolutetime"][
                    0, idxs[0] : idxs[1]
                ]
                if not absolutetime:
                    time = time.astype("int64") - self.timepoints[0].astype("int64")
                time = time / 1e9
                # print(dsetattrs.keys())
                axs[icol, irow].set_ylabel(getplotableunitstring(dsetattrs["Unit"]))
                if not absolutetime:
                    axs[icol, irow].set_xlabel("Relative time in s")
                else:
                    axs[icol, irow].set_xlabel("Unixtime in s")
                axs[icol, irow].set_title(dataset.replace("_", " "))
                for i in np.arange(
                    self.datafile["RAWDATA/" + sensor + "/" + dataset].shape[0]
                ):
                    label = dsetattrs["Physical_quantity"][i]
                    data = self.datafile["RAWDATA/" + sensor + "/" + dataset][
                        i, idxs[0] : idxs[1]
                    ]
                    axs[icol, irow].plot(time, data, label=label)
                    axs[icol, irow].legend()
                irow = irow + 1
            icol = icol + 1
        fig.show()

    def createHDFGroup(self):
        try:
            EXPERIMENTS = self.datafile["EXPERIMENTS"]
        except KeyError:
            EXPERIMENTS = self.datafile.create_group("EXPERIMENTS")
        try:
            SINEEX = EXPERIMENTS[self.params["experemientTypeName"]]
        except KeyError:
            SINEEX = EXPERIMENTS.create_group(self.params["experemientTypeName"])
        try:
            EXPGROUP = SINEEX[self.experiemntID]
            print(str(EXPGROUP.name) + "existed allready returning groupname")
        except KeyError:
            EXPGROUP = SINEEX.create_group(self.experiemntID)
        self.datafile.flush()
        return EXPGROUP


class sineexcitation(experiment):
    def __init__(self, hdfmet4fofdatafile, times, experiementID):
        super().__init__(hdfmet4fofdatafile, times, "Sine excitation", experiementID)

    def dofft(self):
        for sensor in self.met4fofdatafile.sensordatasets:
            idxs = self.idxs[sensor]
            points = idxs[1] - idxs[0]
            time = self.datafile["RAWDATA/" + sensor + "/" + "Absolutetime"][
                0, idxs[0] : idxs[1]
            ]
            reltime = time.astype("int64") - self.timepoints[0].astype("int64")
            self.runtimeData[sensor]["Mean Delta T"] = np.mean(np.diff(reltime / 1e9))
            self.runtimeData[sensor]["RFFT Frequencys"] = np.fft.rfftfreq(
                points, self.runtimeData[sensor]["Mean Delta T"]
            )
            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                data = self.datafile["RAWDATA/" + sensor + "/" + dataset][
                    :, idxs[0] : idxs[1]
                ]
                self.runtimeData[sensor][dataset]["RFFT"] = np.fft.rfft(data, axis=1)
                self.runtimeData[sensor][dataset]["FFT_max_freq"] = self.runtimeData[
                    sensor
                ]["RFFT Frequencys"][
                    np.argmax(
                        abs(
                            np.sum(
                                self.runtimeData[sensor][dataset]["RFFT"][:, 1:], axis=0
                            )
                        )
                    )
                    + 1
                ]
                # print(self.Data[sensor][dataset]['FFT_max_freq'])
        self.flags["FFT Calculated"] = True

    def do3paramsinefits(self, freqs, periods=10):
        if not self.flags["FFT Calculated"]:
            self.dofft()
        for sensor in self.met4fofdatafile.sensordatasets:
            idxs = self.idxs[sensor]
            points = idxs[1] - idxs[0]
            time = self.datafile["RAWDATA/" + sensor + "/" + "Absolutetime"][
                0, idxs[0] : idxs[1]
            ]
            reltime = time.astype("int64") - self.timepoints[0].astype("int64")
            reltime = reltime / 1e9
            excitationfreqs = freqs
            uniquexfreqs = np.sort(np.unique(excitationfreqs))
            idxs = self.idxs[sensor]
            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                try:
                    fftmaxfreq = self.runtimeData[sensor][dataset]["FFT_max_freq"]
                except NameError:
                    self.dofft()
                freqidx = binarySearch(uniquexfreqs, fftmaxfreq)
                datasetrows = self.datafile["RAWDATA/" + sensor + "/" + dataset].shape[
                    0
                ]
                f0 = uniquexfreqs[freqidx]
                self.data[sensor][dataset]["Sin_Fit_freq"] = f0 * np.ones(
                    [datasetrows, 1]
                )  # we doing an singe frequency fit
                # calc first row and create output array[:,idxs[0]:idxs[1]]
                sineparams = st.seq_threeparsinefit(
                    self.datafile["RAWDATA/" + sensor + "/" + dataset][
                        0, idxs[0] : idxs[1]
                    ],
                    reltime,
                    f0,
                    periods=10,
                )
                self.data[sensor][dataset]["SinPOpt"] = np.zeros([datasetrows, 4])
                self.data[sensor][dataset]["SinPCov"] = np.zeros([datasetrows, 4, 4])
                self.data[sensor][dataset]["SinParams"] = np.zeros(
                    [datasetrows, sineparams.shape[0], 3]
                )
                self.data[sensor][dataset]["SinParams"][0] = sineparams
                for i in np.arange(1, datasetrows):
                    sineparams = st.seq_threeparsinefit(
                        self.datafile["RAWDATA/" + sensor + "/" + dataset][
                            i, idxs[0] : idxs[1]
                        ],
                        reltime,
                        f0,
                        periods=10,
                    )
                    self.data[sensor][dataset]["SinParams"][i] = sineparams
                for i in np.arange(datasetrows):
                    sineparams = self.data[sensor][dataset]["SinParams"][i]
                    Complex = sineparams[:, 1] + 1j * sineparams[:, 0]
                    DC = sineparams[:, 2]
                    Freq = np.ones(sineparams.shape[0]) * f0
                    self.data[sensor][dataset]["SinPOpt"][i, :] = [
                        np.mean(abs(Complex)),
                        np.mean(DC),
                        np.mean(Freq),
                        np.mean(np.unwrap(np.angle(Complex))),
                    ]
                    CoVarData = np.stack(
                        (abs(Complex), DC, Freq, np.unwrap(np.angle(Complex))), axis=0
                    )
                    self.data[sensor][dataset]["SinPCov"][i, :] = np.cov(
                        CoVarData, bias=True
                    )  # bias=True Nomation With N like np.std
        self.flags["Sine fit calculated"] = True
        return

    def plotXYsine(
        self, sensor, dataset, axis, fig=None, ax=None, mode="XY", alpha=0.05
    ):
        dsetattrs = self.datafile["RAWDATA/" + sensor + "/" + dataset].attrs
        idxs = self.idxs[sensor]
        time = self.datafile["RAWDATA/" + sensor + "/" + "Absolutetime"][
            0, idxs[0] : idxs[1]
        ]
        reltime = time.astype("int64") - self.timepoints[0].astype("int64")
        reltime = reltime / 1e9
        sinparams = self.data[sensor][dataset]["SinPOpt"]
        f0 = sinparams[axis, 2]
        dc = sinparams[axis, 1]
        amp = sinparams[axis, 0]
        phi = sinparams[axis, 3]
        undisturbedsine = np.sin(2 * np.pi * f0 * reltime + phi) * amp + dc
        sinedata = self.datafile["RAWDATA/" + sensor + "/" + dataset][
            axis, idxs[0] : idxs[1]
        ]
        if fig == None and ax == None:
            fig, ax = plt.subplots()
            ax.set_xlabel(
                "Nonminal "
                + dsetattrs["Physical_quantity"][axis]
                + " calculated from sine in "
                + getplotableunitstring(dsetattrs["Unit"])
            )
        if mode == "XY":
            data = sinedata
            ax.set_ylabel(
                dsetattrs["Physical_quantity"][axis]
                + "in "
                + getplotableunitstring(dsetattrs["Unit"])
            )
        if mode == "diff":
            data = sinedata - undisturbedsine
            ax.set_ylabel(
                dsetattrs["Physical_quantity"][axis]
                + "in "
                + getplotableunitstring(dsetattrs["Unit"])
            )
        if mode == "XY+fit":
            ax.set_ylabel(
                dsetattrs["Physical_quantity"][axis]
                + "in "
                + getplotableunitstring(dsetattrs["Unit"])
            )
            data = sinedata
            polycoevs = np.polyfit(undisturbedsine, data, 2)
            print(polycoevs)

        ax.scatter(undisturbedsine, data, alpha=alpha, s=1, label="Raw data")
        if mode == "XY+fit":
            max = np.max(undisturbedsine)
            min = np.min(undisturbedsine)
            delta = max - min
            polyx = np.arange(min, max, delta / 2e3)
            poly = np.poly1d(polycoevs)
            ax.plot(polyx, poly(polyx), label="Fit")
            # ax.plot(polyx,polyx,label='Prefect signal')
            ax.legend()
        if mode == "XY+fit":
            return polycoevs
        else:
            return

    def plotsinefit(self, absolutetime=False):
        cols = len(self.met4fofdatafile.sensordatasets)  # one colum for every sensor
        datasetspersensor = []
        for sensors in self.met4fofdatafile.sensordatasets:
            datasetspersensor.append(len(self.met4fofdatafile.sensordatasets[sensors]))
        rows = np.max(datasetspersensor)
        fig, axs = plt.subplots(cols, rows, sharex="all")
        icol = 0
        for sensor in self.met4fofdatafile.sensordatasets:
            irow = 0
            idxs = self.idxs[sensor]
            axs[icol, 0].annotate(
                sensor.replace("_", " "),
                xy=(0, 0.5),
                xytext=(-axs[icol, 0].yaxis.labelpad - 5, 0),
                xycoords=axs[icol, 0].yaxis.label,
                textcoords="offset points",
                size="large",
                ha="right",
                va="center",
                rotation=90,
            )

            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                dsetattrs = self.datafile["RAWDATA/" + sensor + "/" + dataset].attrs
                time = self.datafile["RAWDATA/" + sensor + "/" + "Absolutetime"][
                    0, idxs[0] : idxs[1]
                ]
                if not absolutetime:
                    time = time.astype("int64") - self.timepoints[0].astype("int64")
                time = time / 1e9
                # print(dsetattrs.keys())
                axs[icol, irow].set_ylabel(getplotableunitstring(dsetattrs["Unit"]))
                if not absolutetime:
                    axs[icol, irow].set_xlabel("Relative time in s")
                else:
                    axs[icol, irow].set_xlabel("Unixtime in s")
                axs[icol, irow].set_title(dataset.replace("_", " "))
                for i in np.arange(
                    self.datafile["RAWDATA/" + sensor + "/" + dataset].shape[0]
                ):
                    label = dsetattrs["Physical_quantity"][i]
                    data = self.datafile["RAWDATA/" + sensor + "/" + dataset][
                        i, idxs[0] : idxs[1]
                    ]
                    p = axs[icol, irow].plot(time, data, label=label)
                    sinparams = self.data[sensor][dataset]["SinPOpt"]
                    f0 = sinparams[i, 2]
                    dc = sinparams[i, 1]
                    amp = sinparams[i, 0]
                    phi = sinparams[i, 3]
                    sinelabel = label + " Sine Fit"
                    undisturbedsine = np.sin(2 * np.pi * f0 * time + phi) * amp + dc
                    axs[icol, irow].plot(
                        time,
                        undisturbedsine,
                        label=sinelabel,
                        color=p[0].get_color(),
                        ls="dotted",
                    )
                    axs[icol, irow].legend()
                irow = irow + 1
            icol = icol + 1
        fig.show()

    def calculatetanloguephaseref1freq(
        self,
        refdatagroupname,
        refdataidx,
        analogrefchannelname,
        analogrefchannelidx,
        analogchannelquantity="Voltage",
    ):
        adcreftfname = analogrefchannelname
        adcreftfname = adcreftfname.replace("RAWDATA", "REFERENCEDATA")
        ADCTF = transferfunktion(self.datafile[adcreftfname]["Transferfunction"])
        for sensor in self.met4fofdatafile.sensordatasets:
            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                datasetrows = self.datafile["RAWDATA/" + sensor + "/" + dataset].shape[
                    0
                ]
                self.data[sensor][dataset]["Transfer_coefficients"] = {}
                TC = self.data[sensor][dataset]["Transfer_coefficients"][
                    self.datafile[refdatagroupname].attrs["Refference_Qauntitiy"]
                ] = {}
                TC["Magnitude"] = {}
                TC["Magnitude"]["value"] = np.zeros([datasetrows, datasetrows])
                TC["Magnitude"]["value"][:] = np.NAN
                TC["Magnitude"]["uncertainty"] = np.zeros([datasetrows, datasetrows])
                TC["Magnitude"]["uncertainty"][:] = np.NAN

                TC["Excitation_amplitude"] = {}
                TC["Excitation_amplitude"]["value"] = np.zeros(
                    [datasetrows, datasetrows]
                )
                TC["Excitation_amplitude"]["value"][:] = np.NAN
                TC["Excitation_amplitude"]["uncertainty"] = np.zeros(
                    [datasetrows, datasetrows]
                )
                TC["Excitation_amplitude"]["uncertainty"][:] = np.NAN
                TC["Phase"] = {}
                TC["Phase"]["value"] = np.zeros([datasetrows, datasetrows])
                TC["Phase"]["value"][:] = np.NAN
                TC["Phase"]["uncertainty"] = np.zeros([datasetrows, datasetrows])
                TC["Phase"]["uncertainty"][:] = np.NAN
                TC["Frequency"] = {}
                TC["Frequency"]["value"] = np.zeros([datasetrows])
                TC["Frequency"]["value"][:] = np.NAN
                TC["Frequency"]["uncertainty"] = np.zeros([datasetrows])
                TC["Frequency"]["uncertainty"][:] = np.NAN
                TC["Excitation_frequency"] = {}
                TC["Excitation_frequency"]["value"] = np.zeros([datasetrows])
                TC["Excitation_frequency"]["value"][:] = np.NAN
                TC["Excitation_frequency"]["uncertainty"] = np.zeros([datasetrows])
                TC["Excitation_frequency"]["uncertainty"][:] = np.NAN
                for j in np.arange(0, datasetrows):
                    for i in np.arange(0, datasetrows):
                        if j == 0:
                            TC["Frequency"]["value"][i] = self.data[sensor][dataset][
                                "Sin_Fit_freq"
                            ][i]
                            TC["Excitation_frequency"]["value"][i] = self.datafile[
                                refdatagroupname
                            ]["Frequency"]["value"][i][refdataidx]
                        fitfreq = self.data[sensor][dataset]["Sin_Fit_freq"][j]
                        print(refdataidx)
                        reffreq = self.datafile[refdatagroupname]["Frequency"]['value'][
                            i
                        ][refdataidx]
                        if fitfreq != reffreq:
                            warinigstr = (
                                "Frequency mismatach in Sesnor"
                                + sensor
                                + " "
                                + dataset
                                + " fit["
                                + str(i)
                                + "]= "
                                + str(fitfreq)
                                + " ref["
                                + str(refdataidx)
                                + "]= "
                                + str(reffreq)
                                + " Transferfunction will be invaladie !!"
                            )
                            warnings.warn(warinigstr, RuntimeWarning)
                        else:

                            # calculate magnitude response
                            TC["Excitation_amplitude"]["value"][j, i]       = self.datafile[refdatagroupname]["Excitation_amplitude"]["value"][j][refdataidx]
                            TC["Excitation_amplitude"]["uncertainty"][j, i] = self.datafile[refdatagroupname]["Excitation_amplitude"][ "uncertainty"][j][refdataidx]
                            ufexamp = ufloat(TC["Excitation_amplitude"]["value"][j, i],TC["Excitation_amplitude"]["uncertainty"][j, i])
                            if ufexamp == 0:
                                ufexamp = np.NaN
                            ufmeasamp = ufloat(
                                self.data[sensor][dataset]["SinPOpt"][i][0],
                                self.data[sensor][dataset]["SinPCov"][i][0, 0],
                            )
                            mag = ufmeasamp / ufexamp
                            TC["Magnitude"]["value"][j, i] = mag.n
                            TC["Magnitude"]["uncertainty"][j, i] = mag.s
                            # calculate phase
                            adcname = analogrefchannelname.replace("RAWDATA/", "")

                            ufdutphase = ufloat(
                                self.data[sensor][dataset]["SinPOpt"][j][3],
                                self.data[sensor][dataset]["SinPCov"][j][3, 3],
                            )

                            ufanalogrefphase = ufloat(
                                self.data[adcname][analogchannelquantity]["SinPOpt"][
                                    analogrefchannelidx
                                ][3],
                                self.data[adcname][analogchannelquantity]["SinPCov"][
                                    analogrefchannelidx
                                ][3, 3],
                            )

                            ufADCTFphase = ADCTF.getNearestTF(analogrefchannelidx, fitfreq)['Phase']

                            ufrefphase = ufloat(
                                self.datafile[refdatagroupname]["Phase"]['value'][j][refdataidx],self.datafile[refdatagroupname]["Phase"]["uncertainty"][j][refdataidx]
                            )  # in rad
                            phase = (
                                ufdutphase
                                - (ufanalogrefphase + ufADCTFphase)
                                + ufrefphase
                            )
                            if phase.n < -np.pi:
                                phase += ufloat(2 * np.pi, 0)
                            elif phase.n > np.pi:
                                phase -= ufloat(2 * np.pi, 0)
                            TC["Phase"]["value"][j, i] = phase.n
                            TC["Phase"]["uncertainty"][j, i] = phase.s
        pass

    def saveToHdf(self):
        if not self.flags["saved_to_disk"]:
            experimentGroup = self.createHDFGroup()
            experimentGroup.attrs["Start_time"] = self.timepoints[0]
            experimentGroup.attrs["End_time"] = self.timepoints[1]
            experimentGroup.attrs["ID"] = self.experiemntID
            self.datafile.flush()
            Path("tmp").mkdir(parents=True, exist_ok=True)
            dd.io.save("tmp/" + self.experiemntID + ".hdf5", self.data)
            h5df = h5py_plain.File("tmp/" + self.experiemntID + ".hdf5", "r")
            for key in h5df.keys():
                self.datafile.copy(h5df[key], experimentGroup)
                experimentGroup[key].attrs["Start_index"] = self.idxs[key][0]
                experimentGroup[key].attrs["Stop_index"] = self.idxs[key][1]
                self.datafile.flush()
            self.flags["saved_to_disk"] = True
        else:
            raise RuntimeWarning("Data already written to hdf file. Skipping")


def processdata(i):
    sys.stdout.flush()
    times = mpdata["movementtimes"][i]
    refidx = int(mpdata["refidx"][i])
    print("DONE i=" + str(i) + "refidx=" + str(refidx))
    times[0] += 1e9
    times[1] -= 1e9
    if times[1].astype(np.int64) - times[0].astype(np.int64) < 0:
        raise ValueError("time after cutting is <0")
    experiment = sineexcitation(
        mpdata["hdfinstance"], times, "{:05d}".format(i) + "_Sine_Excitation"
    )
    sys.stdout.flush()
    # print(experiment)
    sys.stdout.flush()
    start = time.time()
    experiment.dofft()
    end = time.time()
    # print("FFT Time "+str(end - start))
    start = time.time()

    # axisfreqs=mpdata['hdfinstance'].hdffile['REFERENCEDATA/Acceleration_refference']['Frequency'][:, refidx]['value']
    # axisfreqs=axisfreqs[axisfreqs != 0]#remove zero elements
    axisfreqs = mpdata["uniquexfreqs"]
    experiment.do3paramsinefits(axisfreqs, periods=10)
    end = time.time()
    # print("Sin Fit Time "+str(end - start))
    sys.stdout.flush()
    experiment.calculatetanloguephaseref1freq(
        "REFERENCEDATA/Acceleration_refference",
        refidx,
        "RAWDATA/0x1fe40a00_STM32_Internal_ADC",
        0,
    )
    print("DONE i=" + str(i) + "refidx=" + str(refidx))
    return experiment


def generateCEMrefIDXfromfreqs(freqs, removefreqs=np.array([2000.0])):
    refidx = np.empty(0)
    for i in np.arange(freqs.size):
        if not freqs[i] in removefreqs:
            refidx = np.append(refidx, i)
            i = i + 1
    return refidx

if __name__ == "__main__":
    start = time.time()

    # hdffilename = r"D:\data\IMUPTBCEM\Messungen_CEM\MPU9250CEM.hdf5"
    # hdffilename = r"D:\data\MessdatenTeraCube\Test2_XY 10_4Hz\Test2 XY 10_4Hz.hdf5"
    ##revcsv = r"/media/benedikt/nvme/data/2020-09-07_Messungen_MPU9250_SN31_Zweikanalig/WDH3/20200907160043_MPU_9250_0x1fe40000_metallhalter_sensor_sensor_SN31_WDH3_Ref_TF.csv"
    sensorname = "0x1fe40000_MPU_9250"
    hdffilename = r"/media/benedikt/nvme/data/IMUPTBCEM/WDH3/MPU9250PTB.hdf5"
    try:
        os.remove(hdffilename)
    except FileNotFoundError:
        pass
    shutil.copyfile(hdffilename.replace(".hdf5","(copy).hdf5"), hdffilename)


    # revcsv = r"/media/benedikt/nvme/data/2020-09-07_Messungen_MPU9250_SN31_Zweikanalig/Messungen_CEM/m1/20201023130103_MPU_9250_0xbccb0000_00000_Ref_TF.csv"
    datafile = h5py.File(hdffilename, "r+")

    test = hdfmet4fofdatafile(datafile)
    #getRAWTFFromExperiemnts(datafile['EXPERIMENTS/Sine excitation'], '0x1fe40000_MPU_9250')
    #add1dsinereferencedatatohdffile(revcsv, datafile)
    # adc_tf_goup=datafile.create_group("REFENCEDATA/0x1fe40a00_STM32_Internal_ADC")
    # addadctransferfunctiontodset(adc_tf_goup,datafile["RAWDATA/0x1fe40a00_STM32_Internal_ADC"], [r"/media/benedikt/nvme/data/201118_BMA280_amplitude_frequency/200318_1FE4_ADC123_19V5_1HZ_1MHZ.json"])
    # datafile.flush()

    # add1dsinereferencedatatohdffile(revcsv, datafile)
    # adc_tf_goup=datafile.create_group("REFENCEDATA/0xbccb0a00_STM32_Internal_ADC")
    # addadctransferfunctiontodset(adc_tf_goup,datafile["RAWDATA/0xbccb0a00_STM32_Internal_ADC"], [r"/home/benedikt/datareceiver/cal_data/BCCB_AC_CAL/201006_BCCB_ADC123_3CLCES_19V5_1HZ_1MHZ.json"])
    # datafile.flush()

    # nomovementidx,nomovementtimes=test.detectnomovment('0x1fe40000_MPU_9250', 'Acceleration')
    # REFmovementidx, REFmovementtimes = test.detectmovment('RAWREFERENCEDATA/0x00000000_PTB_3_Component/Velocity', 'RAWREFERENCEDATA/0x00000000_PTB_3_Component/Releativetime', treshold=0.004,
    #                                                blocksinrow=100, blocksize=10000, plot=True)

    movementidx, movementtimes = test.detectmovment('RAWDATA/0x1fe40000_MPU_9250/Acceleration', 'RAWDATA/0x1fe40000_MPU_9250/Absolutetime', treshold=0.6,
                                                    blocksinrow=100, blocksize=100, plot=True)
    manager = multiprocessing.Manager()
    mpdata = manager.dict()
    mpdata['hdfinstance'] = test
    mpdata['movementtimes'] = movementtimes
    mpdata['lock'] = manager.Lock()
    # PTB Data CALCULATE REFERENCE data index skipping one data set at the end of evry loop
    mpdata['refidx'] = np.zeros([16 * 10])
    refidx = np.zeros([17 * 10])
    for i in np.arange(10):
        refidx[i * 17:(i + 1) * 17] = np.arange(17) + i * 18
    mpdata['refidx'] = refidx
    #refidx = np.array([0,0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33])

    freqs = test.hdffile['REFERENCEDATA/Acceleration_refference/Frequency']['value'][2, :]
    #refidx = generateCEMrefIDXfromfreqs(freqs)
    mpdata['refidx'] = refidx

    unicefreqs = np.unique(freqs, axis=0)
    mpdata['uniquexfreqs'] = unicefreqs
    i = np.arange(refidx.size)
    #i = np.arange(10)

    #i=np.arange(4)
    with multiprocessing.Pool(14) as p:
        results = p.map(processdata, i)
    end = time.time()
    print(end - start)
    #i = 0
    #processdata(1)

    freqs = np.zeros(movementtimes.shape[0])
    mag = np.zeros(movementtimes.shape[0])
    maguncer = np.zeros(movementtimes.shape[0])
    examp = np.zeros(movementtimes.shape[0])
    rawamp = np.zeros(movementtimes.shape[0])
    phase = np.zeros(movementtimes.shape[0])
    phaseuncer = np.zeros(movementtimes.shape[0])
    for i in range(len(results)):
        ex=results[i]
    #    if(i==99):
    #        print("DEBUG")
        mag[i] = ex.data[sensorname]['Acceleration']['Transfer_coefficients']['Acceleration']['Magnitude']['value'][2,2]
        maguncer[i] = ex.data[sensorname]['Acceleration']['Transfer_coefficients']['Acceleration']['Magnitude']['uncertainty'][2,2]
        examp[i] = ex.data[sensorname]['Acceleration']['Transfer_coefficients']['Acceleration']['Excitation_amplitude']['value'][2,2]
        freqs[i] = ex.data[sensorname]['Acceleration']['SinPOpt'][2][2]
        rawamp[i] = ex.data[sensorname]['Acceleration']['SinPOpt'][2][0]
        phase[i] = ex.data[sensorname]['Acceleration']['Transfer_coefficients']['Acceleration']['Phase']['value'][2,2]
        phaseuncer[i] = ex.data[sensorname]['Acceleration']['Transfer_coefficients']['Acceleration']['Phase']['uncertainty'][2,2]
        ex.saveToHdf()
    output = {'freqs': freqs,'mag': mag, 'maguncer': maguncer, 'examp': examp,  'phase': phase,
             'phaseuncer': phaseuncer}
    df = pd.DataFrame(output)
    # for ex in results:
    #      mag[i] = ex.Data['0xbccb0000_MPU_9250']['Acceleration']['TF']['Magnitude'][2]['value']
    #      examp[i] = ex.Data['0xbccb0000_MPU_9250']['Acceleration']['TF']['ExAmp'][2]['value']
    #      freqs[i] = ex.Data['0xbccb0000_MPU_9250']['Acceleration']['SinPOpt'][2][2]
    #      rawamp[i] = ex.Data['0xbccb0000_MPU_9250']['Acceleration']['SinPOpt'][2][0]
    #      phase[i] = ex.Data['0xbccb0000_MPU_9250']['Acceleration']['TF']['Phase'][2]['value']
    #      i = i + 1
    # DC = np.zeros(movementtimes.shape[0])
    # AC = np.zeros(movementtimes.shape[0])
    # ACNominal = test.hdffile['REFENCEDATA/Acceleration_refference/Excitation amplitude'][2,:,'value']
    # F = np.zeros(movementtimes.shape[0])
    # for ex in results:
    #      DC[i] = ex.Data['0x1fe40000_MPU_9250']['Acceleration']['SinPOpt'][2][1]
    #      AC[i] = ex.Data['0x1fe40000_MPU_9250']['Acceleration']['SinPOpt'][2][0]
    #      F[i] = ex.Data['0x1fe40000_MPU_9250']['Acceleration']['SinPOpt'][2][2]
    #      i = i+1
    # color = iter(cm.rainbow(np.linspace(0, 1, np.unique(F).size)))
    # colordict={}
    # for i in range(np.unique(F).size):
    #     colordict[np.unique(F)[i]]=next(color)
    # freqcolors=[]
    # for i in range(F.size):
    #     freqcolors.append(colordict[F[i]])
    # fig,ax=plt.subplots()
    # labelplotet=[]
    # for i in range(len(AC)):
    #     if F[i] not in labelplotet:
    #         ax.scatter(ACNominal[i], DC[i], color=freqcolors[i],Label="{:.1f}".format(F[i]))
    #         labelplotet.append(F[i])
    #     else:
    #          ax.scatter(ACNominal[i], DC[i], color=freqcolors[i])
    # ax.set_xlabel('Nominal amplitude in m/s^2')
    # ax.set_ylabel('DC in m/s^2')
    # ax.legend()
    # fig.show()
    #
    # results[0].plotXYsine('0x1fe40000_MPU_9250', 'Acceleration', 2)
    # fig,ax=plt.subplots()
    # coefs = np.empty([len(results), 3])
    # for ex in results:
    #     coefs[i]=ex.plotXYsine('0x1fe40000_MPU_9250', 'Acceleration',2,fig=fig,ax=ax,mode='XY+fit')

    TF=getRAWTFFromExperiemnts(
        datafile["EXPERIMENTS/Sine excitation"], "0x1fe40000_MPU_9250"
    )

    test.addrawtftohdffromexpreiments(datafile["EXPERIMENTS/Sine excitation"], "0x1fe40000_MPU_9250")
    test.hdffile.flush()
    test.hdffile.close()
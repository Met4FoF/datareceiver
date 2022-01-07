import json
import h5py
import numpy as np
import csv
from MET4FOFDataReceiver import HDF5Dumper
from MET4FOFDataReceiver import SensorDescription
import messages_pb2
import threading
import pandas as pd
import os
import warnings
from tools.adccaldata import Met4FOFADCCall as Met4FOFADCCall

uncerval = np.dtype([("value", np.float), ("uncertainty", np.float)])

def copyHFDatrrs(source, dest):
    for key in list(source.attrs.keys()):
        dest.attrs[key] = source.attrs[key]

def findfilesmatchingstr(folder, pattern):
    matchingfiles = []
    for (dirpath, dirnames, filenames) in os.walk(folder):
        filenames = [os.path.join(dirpath, file) for file in filenames]
        for filename in filenames:
            if filename.find(pattern) > -1:
                matchingfiles.append(filename)
    return sorted(matchingfiles)


def readspektraprpasdf(filename):
    df = pd.read_csv(
        filename,
        decimal=",",
        sep="\t",
        encoding="ISO-8859-1",
        header=17,
        na_values=[
            "Hz",
            "m/s² Peak",
            "mV/(m/s²)",
            "%",
            "dB",
            "Degree",
            "%",
            "(Ref. value)",
        ],
    )
    df.apply(lambda x: x.replace(",", "."))  # remove nasty commas
    return df.iloc[1:]

def spektraprptohdfref(filenamelist):
    df = readspektraprpasdf(filenamelist[0])
    df["loop"] = np.zeros(df.shape[0])
    filecount = len(filenamelist)
    for i in range(filecount - 1):
        nextdf = readspektraprpasdf(filenamelist[i + 1])
        nextdf["loop"] = np.ones(nextdf.shape[0]) * (i + 1)
        df = df.append(nextdf)
    resultdf = pd.DataFrame(df["loop"])
    # loop;frequency;ex_amp;ex_amp_std;phase;phase_std<
    resultdf["frequency"] = df["Frequency,"]
    resultdf["frequency"] = df["Frequency,"]
    resultdf["ex_amp"] = df["Accel.,"]
    # resultdf['ex_amp_std']=df['Accel.,']*(df['Stdrd. dev.,']/df['S,'])*2
    resultdf["ex_amp_std"] = df["Accel.,"] * 0.001  # aussimming 0.1% uncertanty
    resultdf["phase"] = df["Phase,"]
    resultdf["phase_std"] = 0.1
    return resultdf

def readspektracsvFromxls(filename):
    df = pd.read_csv(
        filename,
        decimal=",",
        sep="\t",
        encoding="ISO-8859-1",
        header=17,
        na_values=[
            "Hz",
            "m/s² Peak",
            "mV/(m/s²)",
            "%",
            "dB",
            "Degree",
            "%",
            "(Ref. value)",
        ],
    )
    df.apply(lambda x: x.replace(",", "."))  # remove nasty commas
    return df.iloc[1:]

def readspektraCSVasdf(filename):
    df=pd.read_csv(filename, sep=';', skiprows=[1])
    df.loc[-1] = {'Number': 0, 'Frequency': 80, 'Acceleration': np.NaN, 'S': np.NaN,# add row with index -1
                  'Std.dev. S': np.NaN,'Dev.': np.NaN, 'Phase': np.NaN, 'Std.Dev. Phase': np.NaN,
                  'Distortion. Ref': np.NaN,'Distortion. DUT.': np.NaN, 'Gain. Ref': np.NaN, 'Gain DUT': np.NaN}
    df.index = df.index + 1  # shifting index
    df = df.sort_index()
    return df

def spektraCSVtohdfref(filenamelist):
    df = readspektraCSVasdf(filenamelist[0])
    df["loop"] = np.zeros(df.shape[0])
    filecount = len(filenamelist)
    for i in range(filecount - 1):
        nextdf = readspektraCSVasdf(filenamelist[i + 1])
        nextdf["loop"] = np.ones(nextdf.shape[0]) * (i + 1)
        df = df.append(nextdf)
    resultdf = pd.DataFrame(df["loop"])
    # loop;frequency;ex_amp;ex_amp_std;phase;phase_std<
    resultdf["frequency"] = df["Frequency"]
    resultdf["frequency"] = df["Frequency"]
    resultdf["ex_amp"] = df['Acceleration']
    # resultdf['ex_amp_std']=df['Accel.,']*(df['Stdrd. dev.,']/df['S,'])*2
    resultdf["ex_amp_std"] = df['Acceleration']*2*0.01*df['Std.dev. S']#  df['Std.dev. S'] is relative uncer in % times to because 95% coverage 0.01*df['Acceleration'] to have absolute not relative value
    resultdf["phase"] = df['Phase']
    resultdf["phase_std"] = 2*df['Std.Dev. Phase']
    return resultdf


def adddumptohdf(
    dumpfilename,
    hdffilename,
    hdfdumplock=threading.Lock(),
    adcbaseid=10,
    extractadcdata=False,
    correcttimeglitches=False,
    chunksize=2048
):
    # lock use for multi threading lock in met4FOF hdf dumper implementation
    # adcbaseid=10
    # extractadcdata = False #legacy mode for data where channel 11,12 and 13 contain STM32 internal adc data
    hdfdumpfile = h5py.File(hdffilename, "a")  # open the hdf file

    with open(dumpfilename) as dumpfile:
        reader = csv.reader(dumpfile, delimiter=";")
        descpparsed = False
        skiprowcount = 0
        while not descpparsed:
            row = next(reader)
            try:
                paramsdictjson = json.loads(row[0])
                if isinstance(paramsdictjson, dict):
                    print(paramsdictjson)
                    descpparsed = True
            except json.decoder.JSONDecodeError:
                skiprowcount = skiprowcount + 1
                print("skipped " + str(skiprowcount) + " rows")
                pass

        if paramsdictjson["Name"] == "MPU 9250":
            print("MPU9250 description found adding hieracey")
            if (not"HIERARCHY" in paramsdictjson["1"])or (paramsdictjson["1"]["HIERARCHY"]==None):
                print("HIERARCHY not found adding hieracey")
                paramsdictjson["1"]["HIERARCHY"] = "Acceleration/0"
                paramsdictjson["2"]["HIERARCHY"] = "Acceleration/1"
                paramsdictjson["3"]["HIERARCHY"] = "Acceleration/2"

                paramsdictjson["4"]["HIERARCHY"] = "Angular_velocity/0"
                paramsdictjson["5"]["HIERARCHY"] = "Angular_velocity/1"
                paramsdictjson["6"]["HIERARCHY"] = "Angular_velocity/2"

                paramsdictjson["7"]["HIERARCHY"] = "Magnetic_flux_density/0"
                paramsdictjson["8"]["HIERARCHY"] = "Magnetic_flux_density/1"
                paramsdictjson["9"]["HIERARCHY"] = "Magnetic_flux_density/2"

                paramsdictjson["10"]["HIERARCHY"] = "Temperature/0"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        elif paramsdictjson["Name"] == "BMA 280":
            print("BMA description found adding hieracey")
            if (not ("HIERARCHY" in paramsdictjson["1"])) or (paramsdictjson["1"]["HIERARCHY"]==None):
                print("HIERARCHY not found or NONE adding hieracey")
                paramsdictjson["1"]["HIERARCHY"] = "Acceleration/0"
                paramsdictjson["2"]["HIERARCHY"] = "Acceleration/1"
                paramsdictjson["3"]["HIERARCHY"] = "Acceleration/2"

                paramsdictjson["10"]["HIERARCHY"] = "Temperature/0"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        elif paramsdictjson["Name"] == "STM32 Internal ADC":
            print("STM32 Internal ADC description found")
            if (not "HIERARCHY" in paramsdictjson["1"]) or (paramsdictjson["1"]["HIERARCHY"]==None):
                print("HIERARCHY not found adding hieracey")
                paramsdictjson["1"]["HIERARCHY"] = "Voltage/0"
                paramsdictjson["2"]["HIERARCHY"] = "Voltage/1"
                paramsdictjson["3"]["HIERARCHY"] = "Voltage/2"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        elif paramsdictjson["Name"] == "MS5837_02BA":
            print("MS5837_02BA description found adding hieracey")
            if (not "HIERARCHY" in paramsdictjson["1"]) or (paramsdictjson["1"]["HIERARCHY"]==None):
                paramsdictjson["1"]["HIERARCHY"] = "Temeprature/0"
                paramsdictjson["2"]["HIERARCHY"] = "Releative humidity/0"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        else:
            if (not "HIERARCHY" in paramsdictjson["1"]) or (paramsdictjson["1"]["HIERARCHY"]==None):
                print("sensor " + str(paramsdictjson["Name"]) + " with out HIERARCHY not supported exiting")
                exit()
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        baseid = int(np.floor(paramsdictjson["ID"] / 65536))
        # descriptions are now ready start the hdf dumpers
        sensordumper = HDF5Dumper(sensordscp, hdfdumpfile, hdfdumplock,correcttimeglitches=correcttimeglitches,chunksize=chunksize)
        if extractadcdata:
            adcid = int(baseid * 65536 + 256 * adcbaseid)
            print("ADC ID " + hex(adcid))
            adcparamsdict = {
                "ID": int(adcid),
                "Name": "STM32 Internal ADC",
                "1": {
                    "CHID": 1,
                    "PHYSICAL_QUANTITY": "Voltage Ch 1",
                    "UNIT": "\\volt",
                    "RESOLUTION": 4096.0,
                    "MIN_SCALE": -10,
                    "MAX_SCALE": 10,
                    "HIERARCHY": "Voltage/0",
                },
                "2": {
                    "CHID": 2,
                    "PHYSICAL_QUANTITY": "Voltage Ch 2",
                    "UNIT": "\\volt",
                    "RESOLUTION": 4096.0,
                    "MIN_SCALE": -10,
                    "MAX_SCALE": 10,
                    "HIERARCHY": "Voltage/1",
                },
                "3": {
                    "CHID": 3,
                    "PHYSICAL_QUANTITY": "Voltage Ch 3",
                    "UNIT": "\\volt",
                    "RESOLUTION": 4096.0,
                    "MIN_SCALE": -10,
                    "MAX_SCALE": 10,
                    "HIERARCHY": "Voltage/2",
                },
            }
            adcdscp = SensorDescription(fromDict=adcparamsdict, ID=adcid)
            adcdumper = HDF5Dumper(adcdscp, hdfdumpfile, hdfdumplock)
        cloumnames = next(reader)
        # loop over the remaining file content
        for row in reader:
            sensormsg = messages_pb2.DataMessage()
            try:
                id = int(row[0])
                if paramsdictjson["ID"] == id:
                    sensormsg.id = id
                    sensormsg.sample_number = int(row[1])
                    sensormsg.unix_time = int(row[2])
                    sensormsg.unix_time_nsecs = int(row[3])
                    sensormsg.time_uncertainty = int(row[4])
                    sensormsg.Data_01 = float(row[5])
                    sensormsg.Data_02 = float(row[6])
                    sensormsg.Data_03 = float(row[7])
                    sensormsg.Data_04 = float(row[8])
                    sensormsg.Data_05 = float(row[9])
                    sensormsg.Data_06 = float(row[10])
                    sensormsg.Data_07 = float(row[11])
                    sensormsg.Data_08 = float(row[12])
                    sensormsg.Data_09 = float(row[13])
                    sensormsg.Data_10 = float(row[14])
                    sensordumper.pushmsg(sensormsg, sensordscp)
                    if extractadcdata:
                        adcmsg = messages_pb2.DataMessage()
                        adcmsg.id = adcid
                        adcmsg.sample_number = int(row[1])
                        adcmsg.unix_time = int(row[2])
                        adcmsg.unix_time_nsecs = int(row[3])
                        adcmsg.time_uncertainty = int(row[4])
                        adcmsg.Data_01 = float(row[15])
                        adcmsg.Data_02 = float(row[16])
                        adcmsg.Data_03 = float(row[17])
                        adcdumper.pushmsg(adcmsg, adcdscp)
                else:
                    warnings.warn(
                        "Sensor ID in line mismatach! Line ignored",
                        category=RuntimeWarning,
                    )
            except ValueError as VE:
                print(VE)
                print(row)
                warnings.warn(
                    "line could not converted to values!Lione ignored",
                    category=RuntimeWarning,
                )
        hdfdumpfile.flush()
        hdfdumpfile.close()


def add1dsinereferencedatatohdffile(dataframeOrFilename, hdffile, refference_name, axis, isdeg=True,overWrite=False):
    if isinstance(dataframeOrFilename, pd.DataFrame):
        refcsv = dataframeOrFilename
        isaccelerationreference1d = True
        # todo add aditional check for correct dataframe names
    else:
        saccelerationreference1d = False
        refcsv = pd.read_csv(dataframeOrFilename, delimiter=";", comment="#")
        hdffile = hdffile
        with open(dataframeOrFilename, "r") as file:
            first_line = file.readline()
            second_line = file.readline()
            third_line = file.readline()
            if (
                r"loop;frequency;ex_amp;ex_amp_std;phase;phase_std" in first_line
                and r"#Number;Hz;m/s^2;m/s^2;deg;deg" in third_line
            ):
                isaccelerationreference1d = True
                print("1D Accelerationrefference fiele given creating hdf5 data set")
            else:
                if (
                    not r"loop;frequency;ex_amp;ex_amp_std;phase;phase_std"
                    in first_line
                ):
                    raise RuntimeError(
                        "Looking for >>>loop;frequency;ex_amp;ex_amp_std;phase;phase_std<<< in csvfile first row got"
                        + first_line
                    )
                if not r"#Number;Hz;m/s^2;m/s^2;deg;deg" in third_line:
                    raise RuntimeError(
                        "Looking for >>>loop;frequency;ex_amp;ex_amp_std;phase;phase_std<<< in csvfile first row got"
                        + third_line
                    )
    if isaccelerationreference1d:
        DSGroups = {}
        try:
            REFDATA = hdffile["REFERENCEDATA"]
        except KeyError:
            REFDATA = hdffile.create_group("REFERENCEDATA")
        try:
            group = REFDATA.create_group("Acceleration_refference")
        except ValueError as ve:
            if overWrite:
                group = REFDATA["Acceleration_refference"]
            else:
                raise ve
        group.attrs["Refference_name"] = refference_name
        group.attrs["Sensor_name"] = group.attrs["Refference_name"]
        group.attrs["Refference_type"] = "1D Acceleration"
        group.attrs["Refference_Qauntitiy"] = "Acceleration"
        try:
            DSGroups["Frequency"] = group.create_group("Frequency")
            FreqVal = DSGroups["Frequency"].create_dataset("value", ([3, refcsv.shape[0]]), dtype="float64")
            FreqUncer = DSGroups["Frequency"].create_dataset("uncertainty", ([3, refcsv.shape[0]]), dtype="float64")
        except ValueError as ve:
            if overWrite:
                DSGroups["Frequency"]=group["Frequency"]
            else:
                raise ve
        DSGroups["Frequency"]['value'].make_scale("Frequency")
        DSGroups["Frequency"].attrs["Unit"] = "\\hertz"
        DSGroups["Frequency"].attrs["Physical_quantity"] = "Excitation frequency"
        DSGroups["Frequency"]['value'][axis, :] = refcsv["frequency"].to_numpy()
        DSGroups["Frequency"]['uncertainty'][axis, :] = refcsv["frequency"].to_numpy()*np.NaN
        try:
            DSGroups["Cycle_count"] = group.create_group("Cycle_count")
            NVal = DSGroups["Cycle_count"].create_dataset("value", ([refcsv.shape[0]]), dtype="int32")
            NUncer = DSGroups["Cycle_count"].create_dataset("uncertainty", ([refcsv.shape[0]]), dtype="int32")
        except ValueError as ve:
            if overWrite:
                DSGroups["Cycle_count"] = group["Cycle_count"]
            else:
                raise ve
        DSGroups["Cycle_count"].attrs["Unit"] = "\\one"
        DSGroups["Cycle_count"].attrs["Physical_quantity"] = "Cycle_count"
        DSGroups["Cycle_count"].attrs['Uncertainty_type'] = "Errorless integer number"
        DSGroups["Cycle_count"]['value'][:] = refcsv["loop"].to_numpy()
        DSGroups["Cycle_count"]["uncertainty"][:] = refcsv["loop"].to_numpy()*0
        DSGroups["Cycle_count"]['value'].dims[0].label = "Frequency"
        DSGroups["Cycle_count"]['value'].dims[0].attach_scale(DSGroups["Frequency"]['value'])
        try:
            DSGroups["Excitation_amplitude"] = group.create_group("Excitation_amplitude")
            ExAmpval = DSGroups["Excitation_amplitude"].create_dataset("value", ([3, refcsv.shape[0]]), dtype=float)
            ExAmpuncer= DSGroups["Excitation_amplitude"].create_dataset("uncertainty", ([3, refcsv.shape[0]]), dtype=float)
        except ValueError as ve:
            if overWrite:
                DSGroups["Excitation_amplitude"] = group["Excitation_amplitude"]
            else:
                raise ve
        DSGroups["Excitation_amplitude"].attrs["Unit"] = "\\metre\\second\\tothe{-2}"
        DSGroups["Excitation_amplitude"].attrs["Physical_quantity"] = [
            "X Acceleration Excitation_amplitude",
            "Y Acceleration Excitation_amplitude",
            "Z Acceleration Excitation_amplitude",
        ]
        DSGroups["Excitation_amplitude"].attrs[
            "UNCERTAINTY_TYPE"
        ] = "95% coverage gausian"
        DSGroups["Excitation_amplitude"]["value"][:] = np.NaN
        DSGroups["Excitation_amplitude"]["value"][axis, :] = refcsv["ex_amp"].to_numpy()
        DSGroups["Excitation_amplitude"]["uncertainty"][:] = np.NaN
        DSGroups["Excitation_amplitude"]["uncertainty"][axis, :] = refcsv[
            "ex_amp_std"
        ].to_numpy()
        DSGroups["Excitation_amplitude"]["value"].dims[0].label = "Frequency"
        DSGroups["Excitation_amplitude"]["value"].dims[0].attach_scale(
            DSGroups["Frequency"]["value"]
        )
        try:
            DSGroups["Phase"] = group.create_group("Phase")
            PhaseVal=DSGroups["Phase"].create_dataset("value", ([3, refcsv.shape[0]]), dtype=float)
            PhaseUcer=DSGroups["Phase"].create_dataset("uncertainty", ([3, refcsv.shape[0]]), dtype=float)
        except ValueError as ve:
            if overWrite:
                DSGroups["Phase"] = group["Phase"]
            else:
                raise ve
        DSGroups["Phase"]["value"][:] = np.NaN
        DSGroups["Phase"]["uncertainty"][:] = np.NaN
        DSGroups["Phase"].attrs["Unit"] = "\\radian"
        DSGroups["Phase"].attrs["Physical_quantity"] = [
            "X Inertial phase",
            "Y Inertial phase",
            "Z Inertial phase",
        ]
        DSGroups["Phase"].attrs['Uncertainty_type'] = "95% coverage gausian"
        DSGroups["Phase"]["value"][axis, :] = refcsv["phase"].to_numpy()
        DSGroups["Phase"]["uncertainty"][axis, :] = refcsv["phase_std"].to_numpy()
        if isdeg:
            DSGroups["Phase"]["value"][axis, :] = (
                DSGroups["Phase"]["value"][axis, :] / 180 * np.pi
            )
            DSGroups["Phase"]["uncertainty"][axis, :] = (
                DSGroups["Phase"]["uncertainty"][axis, :] / 180 * np.pi
            )
        DSGroups["Phase"]["value"].dims[0].label = "Frequency"
        DSGroups["Phase"]["value"].dims[0].attach_scale(DSGroups["Frequency"]["value"])
        hdffile.flush()


def addadctransferfunctiontodset(hdffile, adcname, jsonfilelist, isdeg=True,overwrite=False):
    ADCCal = Met4FOFADCCall(Filenames=jsonfilelist)
    TFs = {}
    for channel in ADCCal.fitResults.keys():
        TFs[channel] = ADCCal.GetTransferFunction(channel)
    channelcount = len(TFs.keys())
    freqpoints = np.empty(channelcount)
    i = 0
    freqsmatch = True
    for channel in TFs:
        freqpoints[i] = len(TFs[channel]["Frequencys"])
        if i > 0 and freqsmatch:
            result = (freqpoints[0] == freqpoints[i]).all()
            if result == False:
                freqsmatch = False
                raise ValueError("All ADC Channels need to have the same frequencys")
        i = i + 1
    channeloder = ["ADC1", "ADC2", "ADC3"]
    DSGroups = {}
    try:
        refgroup = hdffile["REFERENCEDATA"]
        try:
            adcrefgroup = refgroup[adcname]
        except KeyError:
            adcrefgroup = refgroup.create_group(adcname)
    except KeyError:
        refgroup = hdffile.create_group("REFERENCEDATA")
        adcrefgroup = refgroup.create_group(adcname)
    adctftopgroup = adcrefgroup.create_group("Transferfunction")
    try:
        hdffile["RAWDATA/" + adcname].attrs["Transferfunction"] = adctftopgroup
    except KeyError:
        warnings.warn("Group RAWDATA/"+str(adcname)+" not found link was to this group was not added")
    DSGroups["Frequency"]= adctftopgroup.create_group(
        "Frequency"
    )
    freqval= DSGroups["Frequency"].create_dataset(
        "value", ([freqpoints[0]]), dtype="float64"
    )
    frequncer= DSGroups["Frequency"].create_dataset(
        "uncertainty", ([freqpoints[0]]), dtype="float64"
    )
    DSGroups["Frequency"]['value'].make_scale("Frequency")
    DSGroups["Frequency"].attrs["Unit"] = "/hertz"
    DSGroups["Frequency"].attrs["Physical_quantity"] = "Excitation frequency"
    DSGroups["Frequency"]['value'][0:] = TFs[channeloder[0]]["Frequencys"]
    DSGroups["Frequency"]['uncertainty'][:] = DSGroups["Frequency"]['value'][0:]*np.NaN

    DSGroups["Magnitude"] = adctftopgroup.create_group(
        "Magnitude")
    magval = DSGroups["Magnitude"].create_dataset(
        "value", ([channelcount, freqpoints[0]]), dtype="float64"
    )
    maguncer = DSGroups["Magnitude"].create_dataset(
        "uncertainty", ([channelcount, freqpoints[0]]), dtype="float64"
    )
    DSGroups["Magnitude"].attrs["Unit"] = "\\one"
    DSGroups["Magnitude"].attrs['Unit_numerator']='\\volt'
    DSGroups["Magnitude"].attrs['Unit_denominator']='\\volt'
    DSGroups["Magnitude"].attrs["Physical_quantity"] = [
        "Magnitude response Voltage Ch 1",
        "Magnitude response Voltage Ch 2",
        "Magnitude response Voltage Ch 3",
    ]
    DSGroups["Magnitude"].attrs['Uncertainty_type'] = "95% coverage gausian"
    i = 0
    for channel in channeloder:
        DSGroups["Magnitude"]['value'][i, :] = TFs[channel]["AmplitudeCoefficent"]
        DSGroups["Magnitude"]['uncertainty'][i, :] = TFs[channel][
            "AmplitudeCoefficentUncer"
        ]
        i = i + 1
    DSGroups["Magnitude"]['value'].dims[0].label = "Frequency"
    DSGroups["Magnitude"]['value'].dims[0].attach_scale(DSGroups["Frequency"]['value'])


    DSGroups["Phase"] = adctftopgroup.create_group(
        "Phase")
    phaseval = DSGroups["Phase"].create_dataset(
        "value", ([channelcount, freqpoints[0]]), dtype="float64"
    )
    phaseuncer = DSGroups["Phase"].create_dataset(
        "uncertainty", ([channelcount, freqpoints[0]]), dtype="float64"
    )
    DSGroups["Phase"].attrs["Unit"] = "\\radian"
    DSGroups["Phase"].attrs["Physical_quantity"] = [
        "Phase response Voltage Ch 1",
        "Phase response Voltage Ch 2",
        "Phase response  Voltage Ch 3",
    ]
    DSGroups["Phase"].attrs['Uncertainty_type'] = "95% coverage gausian"
    i = 0
    for channel in channeloder:
        DSGroups["Phase"]["value"][i, :] = TFs[channel]["Phase"]
        DSGroups["Phase"]["uncertainty"][i, :] = TFs[channel]["PhaseUncer"]
        if isdeg:
            DSGroups["Phase"]["value"][i, :] = (
                    DSGroups["Phase"]["value"][i, :] / 180 * np.pi
            )
            DSGroups["Phase"]["uncertainty"][i, :] = (
                    DSGroups["Phase"]["uncertainty"][i, :] / 180 * np.pi
            )
        i = i + 1
    DSGroups["Phase"]["value"].dims[0].label = "Frequency"
    DSGroups["Phase"]["value"].dims[0].attach_scale(DSGroups["Frequency"]['value'])

    DSGroups["Repetition_count"] = adctftopgroup.create_group(
        "Repetition_count")
    Nval = DSGroups["Repetition_count"].create_dataset(
        "value", ([channelcount, freqpoints[0]]), dtype=np.int32
    )
    Nuncer = DSGroups["Repetition_count"].create_dataset(
        "uncertainty", ([channelcount, freqpoints[0]]), dtype=np.int32
    )
    DSGroups["Repetition_count"].attrs["Unit"] = "\\one"
    DSGroups["Repetition_count"].attrs["Physical_quantity"] = [
        "Datapoints Voltage Ch 1",
        "Datapoints Voltage Ch 2",
        "Datapoints Voltage Ch 3",
    ]
    DSGroups["Repetition_count"].attrs['Uncertainty_type'] = "Errorless integer number"
    i = 0
    for channel in channeloder:
        DSGroups["Repetition_count"]['value'][i, :] = TFs[channel]["N"]
        DSGroups["Repetition_count"]["uncertainty"][i, :] = TFs[channel]["N"]*0# zero uncertanity since its intergervalue
        i = i + 1
    DSGroups["Repetition_count"]['value'].dims[0].label = "Frequency"
    DSGroups["Repetition_count"]['value'].dims[0].attach_scale(DSGroups["Frequency"]['value'])
    hdffile.flush()


def add3compTDMSData(
    TDMSDatafile, hdffile, sensitivity=np.array([8.163, 8.163, 8.163]), chunksize=None
):
    import nptdms as tdms

    tdms_file = tdms.TdmsFile.read(TDMSDatafile)
    print("Grabbing Group -->" + str(tdms_file.groups()[0]))
    group = tdms_file.groups()[0]
    print("Channels-->" + str(tdms_file.groups()[0].channels()))
    X = tdms_file.groups()[0]["PXI1Slot13_ai0"]
    Y = tdms_file.groups()[0]["PXI1Slot13_ai1"]
    Z = tdms_file.groups()[0]["PXI1Slot13_ai2"]
    Ref = tdms_file.groups()[0]["PXI1Slot13_ai4"]
    print("Assuming simultanious äquidistant sampling")
    reltime = X.time_track()
    if chunksize == None:
        chunksize = int(1 / reltime[1])
    try:
        rawrefgroup = hdffile["RAWREFERENCEDATA"]
    except KeyError:
        rawrefgroup = hdffile.create_group("RAWREFERENCEDATA")

    try:
        velodatagpr = rawrefgroup["0x00000000_PTB_3_Component"]
    except KeyError:
        velodatagpr = rawrefgroup.create_group("0x00000000_PTB_3_Component")

    dsreltime = velodatagpr.create_dataset(
        "Releativetime",
        ([1, chunksize]),
        maxshape=(1, reltime.size),
        dtype="uint64",
        compression="gzip",
        shuffle=True,
    )
    dsreltime.make_scale("Relative Time")
    dsreltime.attrs["Unit"] = "\\nano\\seconds"
    dsreltime.attrs["Physical_quantity"] = "Relative Time"
    dsreltime.attrs["Resolution"] = np.exp2(64)
    dsreltime.attrs["Max_scale"] = np.exp2(64)
    dsreltime.attrs["Min_scale"] = 0

    dsvelodata = velodatagpr.create_dataset(
        "Velocity",
        ([3, chunksize]),
        maxshape=(3, reltime.size),
        dtype="float32",
        compression="gzip",
        shuffle=True,
    )
    dsvelodata.resize([3, reltime.size])
    dsvelodata.attrs["Unit"] = "\\metre\\second\\tothe{-1}"
    dsvelodata.attrs["Physical_quantity"] = ["Velocity X", "Velocity Y", "Velocity Z"]
    dsvelodata.attrs["Resolution"] = int(16777216 / 10 * 4)  # TODO check this
    dsvelodata.attrs["Max_scale"] = 2.0 / np.mean(sensitivity)  # TODO check this
    dsvelodata.attrs["Min_scale"] = -2.0 / np.mean(sensitivity)  # TODO check this
    dsvelodata.dims[0].label = "Relative Time"
    dsvelodata.dims[0].attach_scale(dsreltime)

    dsrefdata = velodatagpr.create_dataset(
        "Reference voltage",
        ([1, chunksize]),
        maxshape=(1, reltime.size),
        dtype="float32",
        compression="gzip",
        shuffle=True,
    )
    dsrefdata.resize([1, reltime.size])
    dsrefdata.attrs["Unit"] = "\\volt"
    dsrefdata.attrs["Physical_quantity"] = ["Reference Signal"]
    dsrefdata.attrs["Resolution"] = int(16777216)
    dsrefdata.attrs["Max_scale"] = 5.0
    dsrefdata.attrs["Min_scale"] = -5.0
    dsrefdata.dims[0].label = "Relative Time"
    dsrefdata.dims[0].attach_scale(dsreltime)

    dsSN = velodatagpr.create_dataset(
        "Sample_number",
        ([1, chunksize]),
        maxshape=(1, reltime.size),
        dtype="uint32",
        compression="gzip",
        shuffle=True,
    )

    dsSN.attrs["Unit"] = "/one"
    dsSN.attrs["Physical_quantity"] = "Sample_number"
    dsSN.attrs["Resolution"] = np.exp2(32)
    dsSN.attrs["Max_scale"] = np.exp2(32)
    dsSN.attrs["Min_scale"] = 0
    dsSN.dims[0].label = "Relative Time"
    dsSN.dims[0].attach_scale(dsreltime)
    dsSN.resize([1, reltime.size])
    dsSN[:] = np.arange(reltime.size)

    dsreltime.resize([1, reltime.size])

    dsvelodata[0, :] = X[:] / sensitivity[0]
    dsvelodata[1, :] = Y[:] / sensitivity[1]
    dsvelodata[2, :] = Z[:] / sensitivity[2]
    # as it it is since its voltage
    dsreltime[:] = Ref[:]

    # convert to nanosecond uint64
    nstime = reltime * 1e9
    dsreltime[:] = nstime.astype(np.uint64)

    hdffile.flush()
    hdffile.close()

def initSensorGroup(group,sensorParams,chunksize,size,dsdict= {}):
    maxshape=((size//chunksize)+1)*chunksize#maxsize musst fit all data but also be multiple of chunk size

    dsdict["Absolutetime"] = group.create_dataset(
        "Absolutetime",
        (1, maxshape),
        chunks=(1, chunksize),
        dtype="uint64",
        compression="gzip",
        shuffle=True,
    )
    for key in sensorParams:
        group.attrs[key]=sensorParams[key]
    dsdict["Absolutetime"].make_scale("Absoluitetime")
    dsdict["Absolutetime"].attrs["Unit"] = "\\nano\\seconds"
    dsdict["Absolutetime"].attrs[
        "Physical_quantity"
    ] = "Uinix_time_in_nanoseconds"
    dsdict["Absolutetime"].attrs["Resolution"] = np.exp2(64)
    dsdict["Absolutetime"].attrs["Max_scale"] = np.exp2(64)
    dsdict["Absolutetime"].attrs["Min_scale"] = 0
    dsdict["Absolutetime_uncertainty"] = group.create_dataset(
        "Absolutetime_uncertainty",
        (1, maxshape),
        chunks=(1, chunksize),
        dtype="uint32",
        compression="gzip",
        shuffle=True,
    )
    dsdict["Absolutetime_uncertainty"].attrs[
        "Unit"
    ] = "\\nano\\seconds"
    dsdict["Absolutetime_uncertainty"].attrs[
        "Physical_quantity"
    ] = "Uinix_time_uncertainty_in_nanosconds"
    dsdict["Absolutetime_uncertainty"].attrs["Resolution"] = np.exp2(
        32
    )
    dsdict["Absolutetime_uncertainty"].attrs["Max_scale"] = np.exp2(
        32
    )
    dsdict["Absolutetime_uncertainty"].attrs["Min_scale"] = 0
    dsdict["Sample_number"] = group.create_dataset(
        "Sample_number",
        (1, maxshape),
        chunks=(1, chunksize),
        dtype="uint32",
        compression="gzip",
        shuffle=True,
    )
    dsdict["Sample_number"].attrs["Unit"] = "\\one"
    dsdict["Sample_number"].attrs[
        "Physical_quantity"
    ] = "Sample_number"
    dsdict["Sample_number"].attrs["Resolution"] = np.exp2(32)
    dsdict["Sample_number"].attrs["Max_scale"] = np.exp2(32)
    dsdict["Sample_number"].attrs["Min_scale"] = 0
    return dsdict

def addDataGroup(group,dsdict,name,params,chunksize,shape):
    maxshape=((shape[1]//chunksize)+1)*chunksize#maxsize musst fit all data but also be multiple of chunk size
    ds=dsdict[name] = group.create_dataset(
        name,
        (shape[0], maxshape),
        chunks=(shape[0], chunksize),
        dtype="float32",
        compression="gzip",
        shuffle=True,
    )
    for key in params:
        ds.attrs[key]=params[key]

def velocityFromCounts(countsTdmsGroup, counterfreq=40e6, wavelength=1542.16e-9):
    expectedCountsperSample= counterfreq/countsTdmsGroup.properties['wf_samples'] # calculate expected counts per sample from carrier freq/samplefreq eg 40e6/1e4=40e2
    deltaCounts=np.diff(countsTdmsGroup[:].astype(np.uint32))# cast to unit32 so dviation (np.diff) don't needs to care about overflows
    velocity=(deltaCounts-expectedCountsperSample)*wavelength*countsTdmsGroup.properties['wf_samples']*-1*0.5 #v=ds/dt
    return velocity

def add3compZemaTDMSData(
    TDMSDatafile, hdffile, sensitivityLaser=np.array([0.6125*2, 0.6125*2, 0.6125*2]),sensitivitySensor=1.0, chunksize=None):
    import nptdms as tdms

    tdms_file = tdms.TdmsFile.read(TDMSDatafile)
    #ts = int((tdms_file.groups()[1]['Sensor '].properties['wf_start_time']- np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 'ns'))
    ts = int((tdms_file.groups()[0]['Untitled'].properties['wf_start_time'] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 'ns'))#time stamp for hdf5 file is nanosseconds since 1970.01.01 00:00 in nanoSeconds
    if ts<1622470688*1e9:
        ts=ts-int(37*1e9) #backwards correct time to tai to aling mit gps time missing offset
    print("Groups-->" + str(tdms_file.groups()))
    print("Channels-->" + str(tdms_file.groups()[1].channels()))
    zemaKistlerSensor=tdms_file.groups()[1]['Sensor ']
    countX=tdms_file.groups()[1]['counter x']
    countY = tdms_file.groups()[1]['counter y']
    countZ = tdms_file.groups()[1]['counter z']
    vibroVeloAnalogX = tdms_file.groups()[1]["analog x"]
    vibroVeloAnalogY = tdms_file.groups()[1]["analog y"]
    vibroVeloAnalogZ = tdms_file.groups()[1]["analog z"]
    cola = tdms_file.groups()[1]['COLA ']
    print("Assuming simultanious äquidistant sampling")
    if chunksize == None:
        zemaKistlerChunksize = int(zemaKistlerSensor.properties['wf_samples'])
    else:
        zemaKistlerChunksize=chunksize
    try:
        rawdatagroup = hdffile["RAWDATA"]
    except KeyError:
        rawdatagroup = hdffile.create_group("RAWDATA")

    try:
        zemaKistlerGpr = rawdatagroup["0x00000000_Kistler_8712A5M1"]
    except KeyError:
        zemaKistlerGpr = rawdatagroup.create_group("0x00000000_Kistler_8712A5M1")
    length = zemaKistlerSensor[:].shape[0]-64
    zemaKistlerSensorParams={
                    "Data_point_number": length,
                    "Sensor_ID": 0x00000000,
                    "Sensor_name": "Kistler_8712A5M1",
                    "Start_time": ts,
                    "Start_time_uncertainty": 100}
    zemaKistlerDataParams={
                    "Physical_quantity": ["Acceleration"],
                    "Unit": "\\metre\\second\\tothe{-2}",
                    "Resolution": [65536.0],
                    "Min_scale": [-10.0],
                    "Max_scale": [10.0]}
    zemaKistlerGprDict=initSensorGroup(zemaKistlerGpr,zemaKistlerSensorParams,zemaKistlerChunksize,length)

    addDataGroup(zemaKistlerGpr, zemaKistlerGprDict, 'Acceleration', zemaKistlerDataParams, zemaKistlerChunksize, [1,length])
    zemaKistlerGprDict["Absolutetime"][0,:length]= ts+zemaKistlerSensor.time_track()[64:]*1e9
    zemaKistlerGprDict["Acceleration"][0,:length]= zemaKistlerSensor[64:]/sensitivitySensor

    if chunksize == None:
        vibroVeloAnalogChunksize = int(vibroVeloAnalogX.properties['wf_samples'])
    else:
        vibroVeloAnalogChunksize=chunksize
    try:
        vibroGpr = rawdatagroup["0x00000100_OptoMet_Vibrometer"]
    except KeyError:
        vibroGpr = rawdatagroup.create_group("0x00000100_OptoMet_Vibrometer")
    length = vibroVeloAnalogX[:].shape[0]-64
    vibroSensorParams={
                    "Data_point_number": length,
                    "Sensor_ID": 0x00000100,
                    "Sensor_name": "OptoMet_Vibrometer",
                    "Start_time": ts,
                    "Start_time_uncertainty": 100}
    vibroDataParams={
                    "Physical_quantity": ["Velocity X","Velocity Y","Velocity Z"],
                    "Unit": "\\metre\\second\\tothe{-1}",
                    "Resolution": [65536.0,65536.0,65536.0],
                    "Min_scale": [-2.0,-2.0,-2.0],
                    "Max_scale": [2.0,2.0,2.0]}
    vibroGprDict=initSensorGroup(vibroGpr,vibroSensorParams,vibroVeloAnalogChunksize ,length-64)

    addDataGroup(vibroGpr, vibroGprDict, 'Velocity', vibroDataParams, vibroVeloAnalogChunksize , [3,length-64])
    vibroGprDict["Absolutetime"][0,:length]= ts+vibroVeloAnalogX.time_track()[64:]*1e9
    vibroGprDict['Velocity'][0,:length]= vibroVeloAnalogX[64:]/sensitivityLaser[0]
    vibroGprDict['Velocity'][1, :length] = vibroVeloAnalogY[64:] / sensitivityLaser[1]
    vibroGprDict['Velocity'][2, :length] = vibroVeloAnalogZ[64:] / sensitivityLaser[2]



    if chunksize == None:
        dispChunksize = int(countX.properties['wf_samples'])
    else:
        dispChunksize=chunksize
    try:
        vibroCntGpr = rawdatagroup["0x00000200_OptoMet_Velocity_from_counts"]
    except KeyError:
        vibroCntGpr = rawdatagroup.create_group("0x00000200_OptoMet_Velocity_from_counts")
    length = countX[:].shape[0]
    vibroCntSensorParams={
                    "Data_point_number": length-64,
                    "Sensor_ID": 0x00000200,
                    "Sensor_name": "OptoMet_Vibrometer",
                    "Start_time": ts,
                    "Start_time_uncertainty": 10}
    vibroCntDataParams={
                    "Physical_quantity": ["Velocity X","Velocity Y","Velocity Z"],
                    "Unit": "\\metre\\second\\tothe{-1}",
                    "Resolution": [np.NaN,np.NaN,np.NaN],
                    "Min_scale": [np.NaN,np.NaN,np.NaN],
                    "Max_scale": [np.NaN,np.NaN,np.NaN]}
    vibroCntGprDict=initSensorGroup(vibroCntGpr,vibroCntSensorParams,dispChunksize,length-64)

    addDataGroup(vibroCntGpr, vibroCntGprDict, 'Velocity', vibroCntDataParams, dispChunksize, [3,length-64])
    vibroGprDict["Absolutetime"][0,:length]= ts+countX.time_track()*1e9
    vibroGprDict['Velocity'][0,1:length]= velocityFromCounts(countX)
    vibroGprDict['Velocity'][1, 1:length] = velocityFromCounts(countY)
    vibroGprDict['Velocity'][2, 1:length] = velocityFromCounts(countZ)

    if chunksize == None:
        colaChunksize = int(zemaKistlerSensor.properties['wf_samples'])
    else:
        colaChunksize=chunksize
    try:
        colaGpr = rawdatagroup["0x00000300_Cola_Reference"]
    except KeyError:
        colaGpr = rawdatagroup.create_group("0x00000300_Cola_Reference")
    length = cola[:].shape[0]-64
    colaSensorParams={
                    "Data_point_number": length,
                    "Sensor_ID": 0x00000300,
                    "Sensor_name": "Cola_Reference",
                    "Start_time": ts,
                    "Start_time_uncertainty": 10}
    colaDataParams={
                    "Physical_quantity": ["Voltage"],
                    "Unit": "\\volt",
                    "Resolution": [65536.0],
                    "Min_scale": [-1.0],
                    "Max_scale": [1.0]}
    colaGprDict=initSensorGroup(colaGpr,colaSensorParams,colaChunksize ,length)

    addDataGroup(colaGpr, colaGprDict, 'Voltage', colaDataParams, colaChunksize , [1,length])
    zemaKistlerGprDict["Absolutetime"][0,:length]= ts+cola.time_track()[64:]*1e9
    zemaKistlerGprDict["Voltage"][0,:length]= cola[64:]

    """    
    dsvelodata = velodatagpr.create_dataset(
        "Velocity",
        ([3, chunksize]),
        maxshape=(3, reltime.size),
        dtype="float32",
        compression="gzip",
        shuffle=True,
    )
    dsvelodata.resize([3, reltime.size])
    dsvelodata.attrs["Unit"] = "\\metre\\second\\tothe{-1}"
    dsvelodata.attrs["Physical_quantity"] = ["Velocity X", "Velocity Y", "Velocity Z"]
    dsvelodata.attrs["Resolution"] = int(16777216 / 10 * 4)  # TODO check this
    dsvelodata.attrs["Max_scale"] = 2.0 / np.mean(sensitivity)  # TODO check this
    dsvelodata.attrs["Min_scale"] = -2.0 / np.mean(sensitivity)  # TODO check this
    dsvelodata.dims[0].label = "Relative Time"
    dsvelodata.dims[0].attach_scale(dsreltime)

    dsrefdata = velodatagpr.create_dataset(
        "Reference voltage",
        ([1, chunksize]),
        maxshape=(1, reltime.size),
        dtype="float32",
        compression="gzip",
        shuffle=True,
    )
    dsrefdata.resize([1, reltime.size])
    dsrefdata.attrs["Unit"] = "\\volt"
    dsrefdata.attrs["Physical_quantity"] = ["Reference Signal"]
    dsrefdata.attrs["Resolution"] = int(16777216)
    dsrefdata.attrs["Max_scale"] = 5.0
    dsrefdata.attrs["Min_scale"] = -5.0
    dsrefdata.dims[0].label = "Relative Time"
    dsrefdata.dims[0].attach_scale(dsreltime)

    dsSN = velodatagpr.create_dataset(
        "Sample_number",
        ([1, chunksize]),
        maxshape=(1, reltime.size),
        dtype="uint32",
        compression="gzip",
        shuffle=True,
    )

    dsSN.attrs["Unit"] = "/one"
    dsSN.attrs["Physical_quantity"] = "Sample_number"
    dsSN.attrs["Resolution"] = np.exp2(32)
    dsSN.attrs["Max_scale"] = np.exp2(32)
    dsSN.attrs["Min_scale"] = 0
    dsSN.dims[0].label = "Relative Time"
    dsSN.dims[0].attach_scale(dsreltime)
    dsSN.resize([1, reltime.size])
    dsSN[:] = np.arange(reltime.size)

    dsreltime.resize([1, reltime.size])

    dsvelodata[0, :] = X[:] / sensitivity[0]
    dsvelodata[1, :] = Y[:] / sensitivity[1]
    dsvelodata[2, :] = Z[:] / sensitivity[2]
    # as it it is since its voltage
    dsreltime[:] = Ref[:]

    # convert to nanosecond uint64
    nstime = reltime * 1e9
    dsreltime[:] = nstime.astype(np.uint64)
    """
    hdffile.flush()
    #hdffile.close()


def getRAWTFFromExperiemnts(
    group,
    sensor,
    numeratorQuantity="Acceleration",
    denominatorQuantity="Acceleration",
    type="1D_Z",
):
    keys = list(group.keys())
    length = len(keys)
    path = (
        keys[0]
        + "/"
        + sensor
        + "/"
        + numeratorQuantity
        + "/Transfer_coefficients/"
        + denominatorQuantity
    )
    originalshape = group[path]["Magnitude"]["value"].shape
    TC_components = list(group[path].keys())
    Data = {}
    if type == "1D_X" or type == "1D_Y" or type == "1D_Z":
        for component in TC_components:
            Data[component] = {}
            Data[component]["value"] = np.zeros(length)
            Data[component]["uncertainty"] = np.zeros(length)

    if type == "nD":
        for component in TC_components:
            Data[component] = {}
            if len(group[path][component]["value"].shape) == 2:
                Data[component]["value"] = np.zeros(
                    [length, originalshape[0], originalshape[1]]
                )
                Data[component]["uncertainty"] = np.zeros(
                    [length, originalshape[0], originalshape[1]]
                )
            if len(group[path][component]["value"].shape) == 1:
                Data[component]["value"] = np.zeros([length, originalshape[0]])
                Data[component]["uncertainty"] = np.zeros([length, originalshape[0]])
    if type == "1D_X":
        TCIdxData = (0, 0)
        TCIdxFreq = 0
    if type == "1D_Y":
        TCIdxData = (1, 1)
        TCIdxFreq = 1
    if type == "1D_Z":
        TCIdxData = (2, 2)
        TCIdxFreq = 2
    if type == "nD":
        TCIdxData = (slice(None), slice(None))
        TCIdxFreq = slice(None)
    i = 0
    for experiment in keys:
        path = (
            experiment
            + "/"
            + sensor
            + "/"
            + numeratorQuantity
            + "/Transfer_coefficients/"
            + denominatorQuantity
        )
        for component in TC_components:
            if len(group[path][component]["value"].shape) == 2:
                Data[component]["value"][i] = group[path][component]["value"][TCIdxData]
                Data[component]["uncertainty"][i] = group[path][component][
                    "uncertainty"
                ][TCIdxData]
            if len(group[path][component]["value"].shape) == 1:
                Data[component]["value"][i] = group[path][component]["value"][TCIdxFreq]
                Data[component]["uncertainty"][i] = group[path][component][
                    "uncertainty"
                ][TCIdxFreq]
        i = i + 1
    print(Data)
    return Data

def combineHDFRawdata(outputfilename,listfiles):
    outfile=h5py.File(outputfilename, 'w')
    inputfiles=[None]*len(listfiles)
    outputGroups={}
    for i in range(len(listfiles)):
        inputfiles[i]=h5py.File(listfiles[i], 'r')
        Rawdatagroupnames=list(inputfiles[i]['RAWDATA'].keys())
        for rawdataName in Rawdatagroupnames:
            try:
                group=outputGroups[rawdataName]
                group['overalllength']+=inputfiles[i]['RAWDATA/'+rawdataName].attrs['Data_point_number']
                group['sublength'].append(inputfiles[i]['RAWDATA/'+rawdataName].attrs['Data_point_number'])
            except KeyError:
                group = outputGroups[rawdataName]={'overalllength':inputfiles[i]['RAWDATA/'+rawdataName].attrs['Data_point_number'],
                                                   'sublength':[inputfiles[i]['RAWDATA/'+rawdataName].attrs['Data_point_number']],
                                                   'attrs':dict(inputfiles[i]['RAWDATA/'+rawdataName].attrs),
                                                   'dsets':{}}
                dsetnames=list(inputfiles[i]['RAWDATA/'+rawdataName].keys())
                for dset in dsetnames:
                    group['dsets'][dset]={'dimension':inputfiles[i]['RAWDATA/'+rawdataName+'/'+dset].shape[:-1],
                                          'attrs':dict(inputfiles[i]['RAWDATA/'+rawdataName+'/'+dset].attrs),
                                          'chunks':inputfiles[i]['RAWDATA/'+rawdataName+'/'+dset].chunks,
                                          'dtype':inputfiles[i]['RAWDATA/'+rawdataName+'/'+dset].dtype}

    print(outputGroups)
    RD=outfile.create_group("RAWDATA")
    for rawdataName in Rawdatagroupnames:
        SensorGpr=RD.create_group(rawdataName)
        for key in outputGroups[rawdataName]['attrs'].keys():
            SensorGpr.attrs[key]=outputGroups[rawdataName]['attrs'][key]
        SensorGpr.attrs['Data_point_number']=length=outputGroups[rawdataName]['overalllength']
        for dsetname in outputGroups[rawdataName]['dsets'].keys():
            dimension=list(outputGroups[rawdataName]['dsets'][dsetname]['dimension'])
            dimension.append(length)
            chunks = outputGroups[rawdataName]['dsets'][dsetname]['chunks']
            for i in range(len(dimension)):
                if dimension[i]<chunks[i]:
                    print("dimension to smal seting to chunksize")
                    dimension[i]=chunks[i]
                    i=i+1
            dtype= outputGroups[rawdataName]['dsets'][dsetname]['dtype']
            dset=SensorGpr.create_dataset(dsetname,dimension,chunks=chunks,compression="gzip",shuffle=True,dtype=dtype)
            for key in outputGroups[rawdataName]['dsets'][dsetname]['attrs'].keys():
                dset.attrs[key] = outputGroups[rawdataName]['dsets'][dsetname]['attrs'][key]
            i=0
            startidx=0
            stopidx=0
            for infile in inputfiles:
                sublength=outputGroups[rawdataName]['sublength'][i]
                stopidx+=sublength
                print("start copy block "+str(infile)+rawdataName+'/'+dsetname)
                dset[:,startidx:stopidx]=infile['RAWDATA/'+rawdataName+'/'+dsetname][:,:sublength]
                startidx=stopidx
                i=i+1
                print("Block done ")

    print("Done")
if __name__ == "__main__":
    """
    combineHDFRawdata('/media/benedikt/nvme/data/zema_dynamic_cal/tmp/zyx_250_10_delta_10Hz_50ms2max.hdf5',['/media/benedikt/nvme/data/zema_dynamic_cal/tmp/z_250_10_delta_10Hz_50ms2max.hdf5',
                                 '/media/benedikt/nvme/data/zema_dynamic_cal/tmp/y_250_10_delta_10Hz_50ms2max.hdf5',
                                 '/media/benedikt/nvme/data/zema_dynamic_cal/tmp/x_250_10_delta_10Hz_50ms2max.hdf5'])
    
    folder = r"/media/benedikt/nvme/data/strath/DOE2"
    #reffile = r"/media/benedikt/nvme/data/IMUPTBCEM/WDH3/20200907160043_MPU_9250_0x1fe40000_metallhalter_sensor_sensor_SN31_WDH3_Ref_TF.csv"
    # find all dumpfiles in folder matching str

    # find al spektra reference files
    #reffilenames = findfilesmatchingstr(folder, 'prp.txt')
    # parse spektra reference files
    #cemref=spektraprptohdfref(reffilenames)
    """
    hdffilename='/home/benedikt/tmp/test.hdf5'
    folder = r"/home/benedikt/tmp"
    dumpfilenames = findfilesmatchingstr(folder, r".dump")  # input file name
    for dumpfilename in dumpfilenames:
        #if dumpfilename.find("MPU_9250") != -1:
        adddumptohdf(dumpfilename, hdffilename, extractadcdata=False,correcttimeglitches=True,chunksize=128)
        #if dumpfilename.find("BMA_280") != -1:
        #    adddumptohdf(dumpfilename, hdffilename, extractadcdata=True,correcttimeglitches=True)
        # if dumpfilename.find("MS5837") != -1:
        #    adddumptohdf(dumpfilename, hdffilename, correcttimeglitches=False)
        #    print("skipping MS5837 data")
        # else:
        # adddumptohdf(dumpfilename, hdffilename, extractadcdata=False)
    """
    csvfilenames = findfilesmatchingstr(folder, 'results_with_uncer.csv')

    cemref=spektraCSVtohdfref(csvfilenames)
    """
    hdffile = h5py.File(hdffilename, "a")
    addadctransferfunctiontodset(hdffile, '0x1fe40a00_STM32_Internal_ADC', ['cal_data/1FE4_AC_CAL/200320_1FE4_ADC123_3CYCLES_19V5_1HZ_1MHZ.json'])
    ptbref=r"/home/benedikt/data/IMUPTBCEM/PTB/20200907160043_MPU_9250_0x1fe40000_metallhalter_sensor_sensor_SN31_WDH3_Ref_TF.csv"
    add1dsinereferencedatatohdffile(ptbref, hdffile, "PTB HF acceleration standard", 2, isdeg=True,overWrite=True)
    hdffile.flush()
    hdffile.close()


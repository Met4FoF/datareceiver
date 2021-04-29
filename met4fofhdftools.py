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


def adddumptohdf(
    dumpfilename,
    hdffilename,
    hdfdumplock=threading.Lock(),
    adcbaseid=10,
    extractadcdata=False,
    correcttimeglitches=True
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
            paramsdictjson["1"]["HIERARCHY"] = "Acceleration/0"
            paramsdictjson["2"]["HIERARCHY"] = "Acceleration/1"
            paramsdictjson["3"]["HIERARCHY"] = "Acceleration/2"

            paramsdictjson["10"]["HIERARCHY"] = "Temperature/0"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        elif paramsdictjson["Name"] == "STM32 Internal ADC":
            print("STM32 Internal ADC description found adding hieracey")
            paramsdictjson["1"]["HIERARCHY"] = "Voltage/0"
            paramsdictjson["2"]["HIERARCHY"] = "Voltage/1"
            paramsdictjson["3"]["HIERARCHY"] = "Voltage/2"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        elif paramsdictjson["Name"] == "MS5837_02BA":
            print("MS5837_02BA description found adding hieracey")
            paramsdictjson["1"]["HIERARCHY"] = "Temeprature/0"
            paramsdictjson["2"]["HIERARCHY"] = "Releative humidity/0"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        else:
            print("sensor " + str(paramsdictjson["Name"]) + " not supported exiting")
            exit()
        baseid = int(np.floor(paramsdictjson["ID"] / 65536))
        # descriptions are now ready start the hdf dumpers
        sensordumper = HDF5Dumper(sensordscp, hdfdumpfile, hdfdumplock,correcttimeglitches=correcttimeglitches)
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


def add1dsinereferencedatatohdffile(
    dataframeOrFilename, hdffile, refference_name, axis, isdeg=True
):
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
        group = REFDATA.create_group("Acceleration_refference")
        group.attrs["Refference_name"] = refference_name
        group.attrs["Sensor_name"] = group.attrs["Refference_name"]
        group.attrs["Refference_type"] = "1D Acceleration"
        group.attrs["Refference_Qauntitiy"] = "Acceleration"
        DSGroups["Frequency"] = group.create_group(
            "Frequency"
        )
        FreqVal= DSGroups["Frequency"].create_dataset(
            "value", ([3, refcsv.shape[0]]), dtype="float64"
        )
        FreqUncer= DSGroups["Frequency"].create_dataset(
            "uncertainty", ([3, refcsv.shape[0]]), dtype="float64"
        )
        DSGroups["Frequency"]['value'].make_scale("Frequency")
        DSGroups["Frequency"].attrs["Unit"] = "\\hertz"
        DSGroups["Frequency"].attrs["Physical_quantity"] = "Excitation frequency"
        DSGroups["Frequency"]['value'][axis, :] = refcsv["frequency"].to_numpy()
        DSGroups["Frequency"]['uncertainty'][axis, :] = refcsv["frequency"].to_numpy()*np.NaN

        DSGroups["Cycle_count"] = group.create_group(
            "Cycle_count")
        NVal = DSGroups["Cycle_count"].create_dataset(
            "value", ([refcsv.shape[0]]), dtype="int32"
        )
        NUncer = DSGroups["Cycle_count"].create_dataset(
            "uncertainty", ([refcsv.shape[0]]), dtype="int32"
        )

        DSGroups["Cycle_count"].attrs["Unit"] = "\\one"
        DSGroups["Cycle_count"].attrs["Physical_quantity"] = "Cycle_count"
        DSGroups["Cycle_count"].attrs['Uncertainty_type'] = "Errorless integer number"
        DSGroups["Cycle_count"]['value'][:] = refcsv["loop"].to_numpy()
        DSGroups["Cycle_count"]["uncertainty"][:] = refcsv["loop"].to_numpy()*0
        DSGroups["Cycle_count"]['value'].dims[0].label = "Frequency"
        DSGroups["Cycle_count"]['value'].dims[0].attach_scale(DSGroups["Frequency"]['value'])

        DSGroups["Excitation_amplitude"] = group.create_group("Excitation_amplitude")
        ExAmpval = DSGroups["Excitation_amplitude"].create_dataset("value", ([3, refcsv.shape[0]]), dtype=float)

        ExAmpuncer= DSGroups["Excitation_amplitude"].create_dataset("uncertainty", ([3, refcsv.shape[0]]), dtype=float)
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
        DSGroups["Phase"] = group.create_group("Phase")
        PhaseVal=DSGroups["Phase"].create_dataset(
            "value", ([3, refcsv.shape[0]]), dtype=float
        )
        PhaseUcer=DSGroups["Phase"].create_dataset(
            "uncertainty", ([3, refcsv.shape[0]]), dtype=float
        )
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


def addadctransferfunctiontodset(hdffile, adcname, jsonfilelist, isdeg=True):
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


if __name__ == "__main__":

    folder = r"/home/benedikt/testdata/"
    #reffile = r"/media/benedikt/nvme/data/IMUPTBCEM/WDH3/20200907160043_MPU_9250_0x1fe40000_metallhalter_sensor_sensor_SN31_WDH3_Ref_TF.csv"
    # find all dumpfiles in folder matching str
    dumpfilenames = findfilesmatchingstr(folder, r".dump")  # input file name
    hdffilename = folder+r"test.hdf5"
    try:
        os.remove(hdffilename)
    except FileNotFoundError:
        pass
    for dumpfilename in dumpfilenames:
        #if dumpfilename.find("MPU_9250") != -1:
        #    adddumptohdf(dumpfilename, hdffilename, extractadcdata=True)
        if dumpfilename.find("MS5837") != -1:
            adddumptohdf(dumpfilename, hdffilename, correcttimeglitches=False)
        #    print("skipping MS5837 data")
        #else:
        adddumptohdf(dumpfilename, hdffilename, extractadcdata=False)

    # find al spektra reference files
    #reffilenames = findfilesmatchingstr(folder, 'prp.txt')
    # parse spektra reference files
    #cemref=spektraprptohdfref(reffilenames)
    #hdffile = h5py.File(hdffilename, "a")
    # add reference file
    #add1dsinereferencedatatohdffile(reffile, hdffile, "PTB HF acceleration standard", 2, isdeg=True)
    #addadctransferfunctiontodset(hdffile,'0xbccb0a00_STM32_Internal_ADC', [r"/home/benedikt/datareceiver/cal_data/BCCB_AC_CAL/201006_BCCB_ADC123_3CLCES_19V5_1HZ_1MHZ.json"])
    #addadctransferfunctiontodset(
    #   hdffile,
    #   "0x1fe40a00_STM32_Internal_ADC",
    #   [
    #       r"/home/benedikt/datareceiver/cal_data/1FE4_AC_CAL/200615_1FE4_ADC123_3CLCES_19V5_1HZ_1MHZ.json"
    #   ],
    #)
    #hdffile.close()

    # hdffilename = r"D:\data\MessdatenTeraCube\Test2_XY 10_4Hz\Test2 XY 10_4Hz.hdf5"
    # TDMSDatafile = r"D:\data\MessdatenTeraCube\Test2_XY 10_4Hz\27_10_2020_122245\Spannung.tdms"
    # hdffile=h5py.File(hdffilename, 'a')
    # add3compTDMSData(TDMSDatafile, hdffile)

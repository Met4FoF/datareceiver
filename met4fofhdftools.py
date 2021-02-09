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

def findfilesmatchingstr(folder,pattern):
    matchingfiles = []
    for (dirpath, dirnames, filenames) in os.walk(folder):
        filenames = [os.path.join(dirpath, file) for file in filenames]
        for filename in filenames:
            if filename.find(pattern) > -1:
                matchingfiles.append(filename)
    return sorted(matchingfiles)

def readspektraprpasdf(filename):
    df=pd.read_csv(filename,decimal=',', sep='\t',encoding='ISO-8859-1',header=17,na_values=['Hz','m/s² Peak','mV/(m/s²)','%','dB','Degree','%','(Ref. value)'])
    df.apply(lambda x: x.replace(',', '.'))# remove nasty commas
    return df.iloc[1:]

def spektraprptohdfref(filenamelist):
    df=readspektraprpasdf(filenamelist[0])
    df['loop']=np.zeros(df.shape[0])
    filecount=len(filenamelist)
    for i in range(filecount-1):
        nextdf=readspektraprpasdf(filenamelist[i + 1])
        nextdf['loop']=np.ones(nextdf.shape[0])*(i+1)
        df=df.append(nextdf)
    resultdf=pd.DataFrame(df['loop'])
    #loop;frequency;ex_amp;ex_amp_std;phase;phase_std<
    resultdf['frequency']=df['Frequency,']
    resultdf['frequency'] = df['Frequency,']
    resultdf['ex_amp']=df['Accel.,']
    #resultdf['ex_amp_std']=df['Accel.,']*(df['Stdrd. dev.,']/df['S,'])*2
    resultdf['ex_amp_std'] = df['Accel.,']*0.001# ausimming 0.1% uncertanty
    resultdf['phase']=df['Phase,']
    resultdf['phase_std'] = 0.1
    return resultdf


def adddumptohdf(dumpfilename,hdffilename,hdfdumplock=threading.Lock(),adcbaseid=10,extractadcdata = False):
    # lock use for multi threading lock in met4FOF hdf dumper implementation
    #adcbaseid=10
    #extractadcdata = False #legacy mode for data where channel 11,12 and 13 contain STM32 internal adc data
    hdfdumpfile = h5py.File(hdffilename, 'a')  # open the hdf file

    with open(dumpfilename) as dumpfile:
        reader = csv.reader(dumpfile, delimiter=";")
        descpparsed = False
        skiprowcount = 0
        while (not descpparsed):
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

        if paramsdictjson['Name'] == 'MPU 9250':
            print("MPU9250 description found adding hieracey")
            paramsdictjson['1']["HIERARCHY"] = "Acceleration/0"
            paramsdictjson['2']["HIERARCHY"] = "Acceleration/1"
            paramsdictjson['3']["HIERARCHY"] = "Acceleration/2"

            paramsdictjson['4']["HIERARCHY"] = "Angular_velocity/0"
            paramsdictjson['5']["HIERARCHY"] = "Angular_velocity/1"
            paramsdictjson['6']["HIERARCHY"] = "Angular_velocity/2"

            paramsdictjson['7']["HIERARCHY"] = "Magnetic_flux_density/0"
            paramsdictjson['8']["HIERARCHY"] = "Magnetic_flux_density/1"
            paramsdictjson['9']["HIERARCHY"] = "Magnetic_flux_density/2"

            paramsdictjson['10']["HIERARCHY"] = "Temperature/0"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        elif paramsdictjson['Name'] == 'BMA 280':
            print("BMA description found adding hieracey")
            paramsdictjson['1']["HIERARCHY"] = "Acceleration/0"
            paramsdictjson['2']["HIERARCHY"] = "Acceleration/1"
            paramsdictjson['3']["HIERARCHY"] = "Acceleration/2"

            paramsdictjson['10']["HIERARCHY"] = "Temperature/0"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        elif paramsdictjson['Name'] == 'STM32 Internal ADC':
            print("STM32 Internal ADC description found adding hieracey")
            paramsdictjson['1']["HIERARCHY"] = "Voltage/0"
            paramsdictjson['2']["HIERARCHY"] = "Voltage/1"
            paramsdictjson['3']["HIERARCHY"] = "Voltage/2"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        elif paramsdictjson['Name'] == 'MS5837_02BA':
            print("MS5837_02BA description found adding hieracey")
            paramsdictjson['1']["HIERARCHY"] = "Temeprature/0"
            paramsdictjson['2']["HIERARCHY"] = "Releative humidity/0"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        else:
            print("sensor " + str(paramsdictjson['Name']) + ' not supported exiting')
            exit()
        baseid = int(np.floor(paramsdictjson['ID'] / 65536))
        # descriptions are now ready start the hdf dumpers
        sensordumper = HDF5Dumper(sensordscp, hdfdumpfile, hdfdumplock)
        if extractadcdata:
            adcid = int(baseid * 65536 + 256 * adcbaseid)
            print("ADC ID " + hex(adcid))
            adcparamsdict = {
                'ID': int(adcid),
                'Name': 'STM32 Internal ADC',
                '1': {'CHID': 1,
                      'PHYSICAL_QUANTITY': 'Voltage Ch 1',
                      'UNIT': '\\volt',
                      'RESOLUTION': 4096.0,
                      'MIN_SCALE': -10,
                      'MAX_SCALE': 10,
                      "HIERARCHY": 'Voltage/0'},
                '2': {'CHID': 2,
                      'PHYSICAL_QUANTITY': 'Voltage Ch 2',
                      'UNIT': '\\volt',
                      'RESOLUTION': 4096.0,
                      'MIN_SCALE': -10,
                      'MAX_SCALE': 10,
                      "HIERARCHY": 'Voltage/1'},
                '3': {'CHID': 3,
                      'PHYSICAL_QUANTITY': 'Voltage Ch 3',
                      'UNIT': '\\volt',
                      'RESOLUTION': 4096.0,
                      'MIN_SCALE': -10,
                      'MAX_SCALE': 10,
                      "HIERARCHY": 'Voltage/2'}}
            adcdscp = SensorDescription(fromDict=adcparamsdict, ID=adcid)
            adcdumper = HDF5Dumper(adcdscp, hdfdumpfile, hdfdumplock)
        cloumnames = next(reader)
        # loop over the remaining file content
        for row in reader:
            sensormsg = messages_pb2.DataMessage()
            try:
                id = int(row[0])
                if paramsdictjson['ID']==id:
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
                    warnings.warn("Sensor ID in line mismatach! Line ignored", category=RuntimeWarning)
            except ValueError as VE:
                print(VE)
                print(row)
                warnings.warn("line could not converted to values!Lione ignored", category=RuntimeWarning)
        hdfdumpfile.flush()
        hdfdumpfile.close()

def add1dsinereferencedatatohdffile(dataframeOrFilename, hdffile, axis=2, isdeg=True):
    if isinstance(dataframeOrFilename, pd.DataFrame):
        refcsv=dataframeOrFilename
        isaccelerationreference1d = True
        #todo add aditional check for correct dataframe names
    else:
        saccelerationreference1d = False
        refcsv = pd.read_csv(dataframeOrFilename, delimiter=";", comment='#')
        hdffile = hdffile
        with open(dataframeOrFilename, "r") as file:
            first_line = file.readline()
            second_line = file.readline()
            third_line = file.readline()
            if r'loop;frequency;ex_amp;ex_amp_std;phase;phase_std' in first_line and r'#Number;Hz;m/s^2;m/s^2;deg;deg' in third_line:
                isaccelerationreference1d = True
                print("1D Accelerationrefference fiele given creating hdf5 data set")
            else:
                if not r'loop;frequency;ex_amp;ex_amp_std;phase;phase_std' in first_line:
                    raise RuntimeError(
                        "Looking for >>>loop;frequency;ex_amp;ex_amp_std;phase;phase_std<<< in csvfile first row got" + first_line)
                if not r'#Number;Hz;m/s^2;m/s^2;deg;deg' in third_line:
                    raise RuntimeError(
                        "Looking for >>>loop;frequency;ex_amp;ex_amp_std;phase;phase_std<<< in csvfile first row got" + third_line)
    if isaccelerationreference1d:
        Datasets = {}
        try:
            REFDATA = hdffile["REFERENCEDATA"]
        except KeyError:
            REFDATA = hdffile.create_group("REFERENCEDATA")
        group = REFDATA.create_group("Acceleration_refference")
        group.attrs['Refference_name'] = "PTB HF acceleration standard"
        group.attrs['Sensor_name'] = group.attrs['Refference_name']
        group.attrs['Refference_type'] = "1D acceleration"
        Datasets['Frequency'] = group.create_dataset('Frequency', ([3, refcsv.shape[0]]),
                                                     dtype=uncerval)
        Datasets['Frequency'].make_scale("Frequency")
        Datasets['Frequency'].attrs['Unit'] = "/hertz"
        Datasets['Frequency'].attrs['Physical_quantity'] = "Excitation frequency"
        Datasets['Frequency'][axis, :, "value"] = refcsv['frequency'].to_numpy()
        Datasets['Repetition count'] = group.create_dataset('repetition count', ([refcsv.shape[0]]),
                                                            dtype='int32')
        Datasets['Repetition count'].attrs['Unit'] = "/one"
        Datasets['Repetition count'].attrs['Physical_quantity'] = "Repetition count"
        Datasets['Repetition count'][:] = refcsv['loop'].to_numpy()
        Datasets['Repetition count'].dims[0].label = 'Frequency'
        Datasets['Repetition count'].dims[0].attach_scale(Datasets['Frequency'])
        Datasets['Excitation amplitude'] = group.create_dataset('Excitation amplitude', ([3, refcsv.shape[0]]),
                                                                dtype=uncerval)
        Datasets['Excitation amplitude'].attrs['Unit'] = "\\metre\\second\\tothe{-2}"
        Datasets['Excitation amplitude'].attrs['Physical_quantity'] = ["X Excitation amplitude",
                                                                       "Y Excitation amplitude",
                                                                       "Z Excitation amplitude"]
        Datasets['Excitation amplitude'].attrs['UNCERTAINTY_TYPE'] = "95% coverage gausian"
        Datasets['Excitation amplitude'][axis, :, "value"] = refcsv['ex_amp'].to_numpy()
        Datasets['Excitation amplitude'][axis, :, "uncertainty"] = refcsv['ex_amp_std'].to_numpy()
        Datasets['Excitation amplitude'].dims[0].label = 'Frequency'
        Datasets['Excitation amplitude'].dims[0].attach_scale(Datasets['Frequency'])

        Datasets['Phase'] = group.create_dataset('Phase', ([3, refcsv.shape[0]]),
                                                 dtype=uncerval)
        Datasets['Phase'].attrs['Unit'] = "\\degree"
        Datasets['Phase'].attrs['Physical_quantity'] = ["X Phase",
                                                        "Y Phase",
                                                        "Z Phase"]
        Datasets['Phase'].attrs['UNCERTAINTY_TYPE'] = "95% coverage gausian"
        Datasets['Phase'][axis, :, "value"] = refcsv['phase'].to_numpy()
        Datasets['Phase'][axis, :, "uncertainty"] = refcsv['phase_std'].to_numpy()
        if (isdeg):
            Datasets['Phase'][axis, :, "value"] = Datasets['Phase'][axis, :, "value"] / 180 * np.pi
            Datasets['Phase'][axis, :, "uncertainty"] = Datasets['Phase'][axis, :, "uncertainty"] / 180 * np.pi

        Datasets['Phase'].dims[0].label = 'Frequency'
        Datasets['Phase'].dims[0].attach_scale(Datasets['Frequency'])
        hdffile.flush()


def addadctransferfunctiontodset(hdffile,adcname, jsonfilelist, isdeg=True):
    ADCCal = Met4FOFADCCall(Filenames=jsonfilelist)
    TFs = {}
    for channel in ADCCal.fitResults.keys():
        TFs[channel] = ADCCal.GetTransferFunction(channel)
    channelcount = len(TFs.keys())
    freqpoints = np.empty(channelcount)
    i = 0
    freqsmatch = True
    for channel in TFs:
        freqpoints[i] = len(TFs[channel]['Frequencys'])
        if i > 0 and freqsmatch:
            result = (freqpoints[0] == freqpoints[i]).all()
            if result == False:
                freqsmatch = False
                raise ValueError("All ADC Channels need to have the same frequencys")
        i = i + 1
    channeloder = ['ADC1', 'ADC2', 'ADC3']
    Datasets = {}
    try:
        refgroup = hdffile["REFERENCEDATA"]
        try:
            adcrefgroup = refgroup[adcname]
        except KeyError:
            adcrefgroup = refgroup.create_group(adcname)
    except KeyError:
        refgroup = hdffile.create_group("REFERENCEDATA")
        adcrefgroup= refgroup.create_group(adcname)
    adctfgroup=adcrefgroup.create_group("Transferfunction")
    hdffile['RAWDATA/'+adcname].attrs['Transferfunction'] = adctfgroup
    Datasets['Frequency'] = adctfgroup.create_dataset('Frequency', ([freqpoints[0]]), dtype='float64')
    Datasets['Frequency'].make_scale("Frequency")
    Datasets['Frequency'].attrs['Unit'] = "/hertz"
    Datasets['Frequency'].attrs['Physical_quantity'] = "Excitation frequency"
    Datasets['Frequency'][0:] = TFs[channeloder[0]]['Frequencys']
    Datasets['Magnitude'] = adctfgroup.create_dataset('Magnitude', ([channelcount, freqpoints[0]]),
                                                 dtype=uncerval)
    Datasets['Magnitude'].attrs['Unit'] = "\\one"
    Datasets['Magnitude'].attrs['Physical_quantity'] = ['Magnitude response Voltage Ch 1',
                                                        'Magnitude response Voltage Ch 2',
                                                        'Magnitude response Voltage Ch 3']
    Datasets['Magnitude'].attrs['UNCERTAINTY_TYPE'] = "95% coverage gausian"
    i = 0
    for channel in channeloder:
        Datasets['Magnitude'][i, :, "value"] = TFs[channel]['AmplitudeCoefficent']
        Datasets['Magnitude'][i, :, "uncertainty"] = TFs[channel]['AmplitudeCoefficentUncer']
        i = i + 1
    Datasets['Magnitude'].dims[0].label = 'Frequency'
    Datasets['Magnitude'].dims[0].attach_scale(Datasets['Frequency'])

    Datasets['Phase'] = adctfgroup.create_dataset('Phase', ([channelcount, freqpoints[0]]),
                                             dtype=uncerval)
    Datasets['Phase'].attrs['Unit'] = "\\radian"
    Datasets['Phase'].attrs['Physical_quantity'] = ['Phase response Voltage Ch 1',
                                                    'Phase response Voltage Ch 2',
                                                    'Phase response  Voltage Ch 3']
    Datasets['Phase'].attrs['UNCERTAINTY_TYPE'] = "95% coverage gausian"
    i = 0
    for channel in channeloder:
        Datasets['Phase'][i, :, "value"] = TFs[channel]['Phase']
        Datasets['Phase'][i, :, "uncertainty"] = TFs[channel]['PhaseUncer']
        if isdeg:
            Datasets['Phase'][i, :, "value"] = Datasets['Phase'][i, :, "value"] / 180 * np.pi
            Datasets['Phase'][i, :, "uncertainty"] = Datasets['Phase'][i, :, "uncertainty"] / 180 * np.pi
        i = i + 1
    Datasets['Phase'].dims[0].label = 'Frequency'
    Datasets['Phase'].dims[0].attach_scale(Datasets['Frequency'])

    Datasets['N'] = adctfgroup.create_dataset('N', ([channelcount, freqpoints[0]]),
                                         dtype=np.int32)
    Datasets['N'].attrs['Unit'] = "\\one"
    Datasets['N'].attrs['Physical_quantity'] = ['Datapoints Voltage Ch 1',
                                                'Datapoints Voltage Ch 2',
                                                'Datapoints Voltage Ch 3']
    i = 0
    for channel in channeloder:
        Datasets['N'][i, :] = TFs[channel]['N']
        i = i + 1
    Datasets['N'].dims[0].label = 'Frequency'
    Datasets['N'].dims[0].attach_scale(Datasets['Frequency'])
    hdffile.flush()

if __name__ == "__main__":
    folder=r"/media/benedikt/nvme/data/IMUPTBCEM/Messungen_CEM/"
    #reffile=r"/media/benedikt/nvme/data/IMUPTBCEM/WDH3/20200907160043_MPU_9250_0x1fe40000_metallhalter_sensor_sensor_SN31_WDH3_Ref_TF.csv"
    #find all dumpfiles in folder matching str
    dumpfilenames=findfilesmatchingstr(folder,r".dump") # input file name

    hdffilename=r"/media/benedikt/nvme/data/IMUPTBCEM/Messungen_CEM/MPU9250CEM.hdf5"
    for dumpfilename in dumpfilenames:
        if(dumpfilename.find('MPU_9250')!=-1):
            adddumptohdf(dumpfilename, hdffilename, extractadcdata = True)
        elif(dumpfilename.find('MS5837')!=-1):
            print("skipping MS5837 data")
        else:
            adddumptohdf(dumpfilename, hdffilename, extractadcdata=False)
    #find al spektra reference files
    reffilenames = findfilesmatchingstr(folder, 'prp.txt')
    #parse spektra reference files
    cemref=spektraprptohdfref(reffilenames)
    hdffile=h5py.File(hdffilename, 'a')
    #add reference file
    add1dsinereferencedatatohdffile(cemref, hdffile, axis=2, isdeg=True)
    addadctransferfunctiontodset(hdffile,'0xbccb0a00_STM32_Internal_ADC', [r"/home/benedikt/datareceiver/cal_data/BCCB_AC_CAL/201006_BCCB_ADC123_3CLCES_19V5_1HZ_1MHZ.json"])
    hdffile.close()


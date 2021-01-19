import json
import h5py
import numpy as np
import csv
from MET4FOFDataReceiver import HDF5Dumper
from MET4FOFDataReceiver import SensorDescription
import messages_pb2
import threading

if __name__ == "__main__":
    adcbaseid=10
    extractadcdata = False #legacy mode for data where channel 11,12 and 13 contain STM32 internal adc data

    dumpfilename=r"/media/benedikt/nvme/data/2020-09-07_Messungen_MPU9250_SN31_Zweikanalig/WDH3/20200907160043_MPU_9250_0x1fe40000_metallhalter_sensor_sensor_SN31_WDH3.dump" # input file name
    hdffilename=dumpfilename.replace('.dump','.hdf5')# if you want to add an dataset to an existing hdf file paste file name here
    hdfdumplock = threading.Lock() # lock use for multi threading actualy not implementeted
    hdfdumpfile = h5py.File(hdffilename, 'a') # open the hdf file

    with open(dumpfilename) as dumpfile:
        reader=csv.reader(dumpfile, delimiter=";")
        descpparsed=False
        skiprowcount=0
        while(not descpparsed):
            row = next(reader)
            try:
                paramsdictjson = json.loads(row[0])
                if isinstance(paramsdictjson, dict):
                    print(paramsdictjson)
                    descpparsed=True
            except json.decoder.JSONDecodeError :
                skiprowcount=skiprowcount+1
                print("skipped "+str(skiprowcount)+" rows")
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
            print("MPU9250 description found adding hieracey")
            paramsdictjson['1']["HIERARCHY"] = "Acceleration/0"
            paramsdictjson['2']["HIERARCHY"] = "Acceleration/1"
            paramsdictjson['3']["HIERARCHY"] = "Acceleration/2"

            paramsdictjson['10']["HIERARCHY"] = "Temperature/0"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        elif paramsdictjson['Name'] == 'STM32 Internal ADC':
            print("MPU9250 description found adding hieracey")
            paramsdictjson['1']["HIERARCHY"] = "Voltage/0"
            paramsdictjson['2']["HIERARCHY"] = "Voltage/1"
            paramsdictjson['3']["HIERARCHY"] = "Voltage/2"
            sensordscp = SensorDescription(fromDict=paramsdictjson)
        else:
            print("sensor "+str(paramsdictjson['Name'])+' not supported exiting')
            exit()
        baseid = int(np.floor(paramsdictjson['ID'] / 65536))
        #descriptions are now ready start the hdf dumpers
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
        #loop over the remaining file content
        for row in reader:
            sensormsg = messages_pb2.DataMessage()
            sensormsg.id = int(row[0])
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
            sensordumper.pushmsg(sensormsg,sensordscp)
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
        hdfdumpfile.flush()
        hdfdumpfile.close()
    #hdfdumpfile = h5py.File("multi_position_4.hdf5", 'w')
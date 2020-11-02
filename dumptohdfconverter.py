import pandas as pd
import json
import h5py
import numpy as np
import csv
from MET4FOFDataReceiver import HDF5Dumper
from MET4FOFDataReceiver import SensorDescription

if __name__ == "__main__":
    adcbaseid=10
    dumpfilename=r"D:\data\200907_mpu9250_BMA280_cal\2020-09-07 Messungen MPU9250_SN31_Zweikanalig\Messungen_CEM\m1-Kopie\20201023130103_MPU_9250_0xbccb0000_00000.dump"
    with open(dumpfilename) as dumpfile:
        reader=csv.reader(dumpfile, delimiter=";")
        descpparsed=False
        skiprowcount=0
        while(not descpparsed):
            row = next(reader)
            try:
                paramsdictjson = json.loads(row[0])
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
        else:
            print("sensor "+str(paramsdictjson['Name'])+' not supported exiting')
            exit()
        baseid = int(np.floor(paramsdictjson['ID'] / 65536))
        adcid = baseid*65536 + 256 * adcbaseid
        print("ADC ID " + hex(adcid))
        adcparamsdict = {
            'ID':adcid,
            'Name': 'STM32 Internal ADC',
                              '1': {'CHID': 1,
                                    'PHYSICAL_QUANTITY': 'Voltage Ch 1',
                                    'UNIT': '\\volt',
                                    'RESOLUTION': 4096.0,
                                    'MIN_SCALE': -10,
                                    'MAX_SCALE':10 ,
                                    "HIERARCHY":'Voltage/0'},
                              '2': {'CHID': 2,
                                    'PHYSICAL_QUANTITY': 'Voltage Ch 2',
                                    'UNIT': '\\volt',
                                    'RESOLUTION': 4096.0,
                                    'MIN_SCALE': -10,
                                    'MAX_SCALE': 10,
                                    "HIERARCHY":'Voltage/1'},
                              '3': {'CHID': 3,
                                    'PHYSICAL_QUANTITY': 'Voltage Ch 3',
                                    'UNIT': '\\volt',
                                    'RESOLUTION': 4096.0,
                                    'MIN_SCALE': -10,
                                    'MAX_SCALE': 10,
                                    "HIERARCHY":'Voltage/2'}}


        adcdscp = SensorDescription(fromDict=adcparamsdict,ID=adcid)

    #hdfdumpfile = h5py.File("multi_position_4.hdf5", 'w')
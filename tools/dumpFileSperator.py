#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json as json
import csv


# In[2]:


dumpfilename = r"D:\data\2020-03-03_Messungen_MPU9250_SN_IMEKO_Frequenzgang_Firmware_0.3.0\Met4FOF_mpu9250_Z_Acc_10_hz_250_hz_6rep.dump"
df = pd.read_csv(dumpfilename, delimiter=";", skiprows=[0])
reader = csv.reader(open(dumpfilename), delimiter=";")
fristrow = next(reader)
paramsdictjson = json.loads(fristrow[0])


# In[3]:


namelist = df.columns.values.tolist()
SensorDataNames = namelist[:-6]
ADCDataNames = namelist[:-16]
ADCDataNames.append(namelist[-6])
ADCDataNames.append(namelist[-5])
ADCDataNames.append(namelist[-4])


# In[4]:


# 	bool setADCCalCoevs(uint8_t ADCNumber,float slope,float xAxisCrossPoint,float RMSNoise);
# 	configMan.setADCCalCoevs(0, 0.00488040211169927, -10.029208660668372,
# 			4.6824163159348675e-3);
# 	configMan.setADCCalCoevs(1, 0.004864769104581888, -9.911472983085314,
# 			13.68572038605262e-3);
# 	configMan.setADCCalCoevs(2, 0.004884955868836948, -10.031544601902738,
# 			4.721804326558252e-3);
ADCDecsriptionDict = {
    "Name": "STM32F767 ADC",
    "1": {
        "CHID": 1,
        "PHYSICAL_QUANTITY": "ADC 1 Voltage",
        "UNIT": "\\volt",
        "RESOLUTION": 4096.0,
        "MIN_SCALE": -10.029208660668372 - 0 * 0.00488040211169927,
        "MAX_SCALE": -10.029208660668372 + 4096 * 0.00488040211169927,
    },
    "2": {
        "CHID": 2,
        "PHYSICAL_QUANTITY": "ADC 2 Voltage",
        "UNIT": "\\metre\\second\\tothe{-2}",
        "RESOLUTION": 4096.0,
        "MIN_SCALE": -9.911472983085314 - 0 * 0.004864769104581888,
        "MAX_SCALE": -9.911472983085314 + 4096 * 0.004864769104581888,
    },
    "3": {
        "CHID": 3,
        "PHYSICAL_QUANTITY": "ADC 3 Voltage",
        "UNIT": "\\metre\\second\\tothe{-2}",
        "RESOLUTION": 4096.0,
        "MIN_SCALE": -10.031544601902738 - 0 * 0.004884955868836948,
        "MAX_SCALE": -10.031544601902738 + 4096 * 0.004884955868836948,
    },
}


# In[5]:


ADCDataFileName = dumpfilename.replace(".dump", "_ADC.dump")
ADCSensorIDOffset = 128
ADCDataFile = open(ADCDataFileName, mode="a")
ADCDataFile.write(json.dumps(ADCDecsriptionDict))
ADCDataFile.write("\n")
ADCDataFile.write(
    "id;sample_number;unix_time;unix_time_nsecs;time_uncertainty;Data_01;Data_02;Data_03\n"
)
ADCDataFile.close()
ADCData = df[ADCDataNames]
ADCData["id"] = ADCData["id"] + ADCSensorIDOffset
ADCData.to_csv(ADCDataFileName, header=False, index=False, sep=";", mode="a")


# In[6]:


SensorDataFileName = dumpfilename.replace(".dump", "_Sensor.dump")
SensorDataFile = open(SensorDataFileName, mode="a")
SensorDataFile.write(json.dumps(paramsdictjson))
SensorDataFile.write("\n")
SensorDataFile.write(
    "id;sample_number;unix_time;unix_time_nsecs;time_uncertainty;Data_01;Data_02;Data_03\n"
)
SensorDataFile.close()
SensorData = df[SensorDataNames]
SensorData.to_csv(SensorDataFileName, header=False, index=False, sep=";", mode="a")


# In[5]:


# In[ ]:

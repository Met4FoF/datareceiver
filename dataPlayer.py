import socket
import messages_pb2
import google.protobuf as pb
from google.protobuf.internal.encoder import _VarintBytes
import threading
import time
import numpy as np
import csv
import json
import MET4FOFDataReceiver as DR

class SensorDataPlayer:
    strFieldNames = [
        "str_Data_01",
        "str_Data_02",
        "str_Data_03",
        "str_Data_04",
        "str_Data_05",
        "str_Data_06",
        "str_Data_07",
        "str_Data_08",
        "str_Data_09",
        "str_Data_10",
        "str_Data_11",
        "str_Data_12",
        "str_Data_13",
        "str_Data_14",
        "str_Data_15",
        "str_Data_16",
    ]
    fFieldNames = [
        "f_Data_01",
        "f_Data_02",
        "f_Data_03",
        "f_Data_04",
        "f_Data_05",
        "f_Data_06",
        "f_Data_07",
        "f_Data_08",
        "f_Data_09",
        "f_Data_10",
        "f_Data_11",
        "f_Data_12",
        "f_Data_13",
        "f_Data_14",
        "f_Data_15",
        "f_Data_16",
    ]
    dataFieldNames = [
        "Data_01",
        "Data_02",
        "Data_03",
        "Data_04",
        "Data_05",
        "Data_06",
        "Data_07",
        "Data_08",
        "Data_09",
        "Data_10",
        "Data_11",
        "Data_12",
        "Data_13",
        "Data_14",
        "Data_15",
        "Data_16",
    ]
    """
    Class for replay of saved Met4FoF ASCII Data.
    """

    def __init__(
        self,
        filename,
        tagetip="127.0.0.1",
        port=7654,
        idOverride=0x00000000,
        paramupdateratehz=0.5,
    ):
        """
        

        Parameters
        ----------
        filename : path.
            Path to the file to be played back.
        tagetip : sting, optional
            IP Adress of the DataReceiver. The default is "127.0.0.1".
        port : intger, optional
            UDP Port of the Datareceiver. The default is 7654.
        id : integer, optional
            ID of the simulated sensor. The default is 0x00000001.
        resolutionbit : integer, optional
            The simulated Sensor data are quantizised in 2^resolutionbit  steps beween min and max. The default is 8.
        paramupdateratehz : integer, optional
            Update rate of sensor descriptions. The default is 0.5.

        Returns
        -------
        None.

        """
        self.flags = {"Networtinited": False}
        self.params = {
            "TargetIp": tagetip,
            "Port": port,
            "ParamUpdateRateHz": paramupdateratehz,
            "idOverride": idOverride,
            "fileName":filename
        }
        self.reader = csv.reader(open(filename), delimiter=";")
        fristrow=next(self.reader)
        paramsdictjson=json.loads(fristrow[0])
        self.line = next(self.reader)# discard second line with headers
        self.line = next(self.reader)
        if idOverride !=0:
            self.params["ID"]=idOverride
        else:
            self.params["ID"]=int(self.line[0])
        print("Sensor ID is: "+hex(self.params["ID"]))
        self.firstpacket_time=float(self.line[2]) + (float(self.line[3]) * 1e-9)
        self.Description=DR.SensorDescription(fromDict=paramsdictjson)
        self.Description.ID=self.params["ID"]
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.flags["Networtinited"] = True
        self.msgcount = 0
        self.lastTimestamp = 0
        self.DataRate=0
        self._stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self.run, name="Sensor Simulator", args=()
        )
        self.packetssend = 0
        self.thread.start()

    def run(self):
        """
        Starts the sensor playback task.

        Returns
        -------
        None.

        """
        firsttime = time.time()
        delta_t_dscp = 1 / self.params["ParamUpdateRateHz"]
        next_time_dscp = firsttime + delta_t_dscp
        first_time_data=time.time()
        next_time_data = time.time()
        lastTousandPacktesTimeStamp=time.time()
        while not self._stop_event.is_set():
            # TODO improve time scheduling
            if next_time_data - time.time() < 0:
                self.__sendDataMsg(self.line)
                delta=float(self.line[2]) + (float(self.line[3]) * 1e-9)-self.firstpacket_time
                next_time_data = first_time_data + delta
                self.line=next(self.reader)
                if self.packetssend % 1000 == 0:
                    sendTime=time.time()
                    self.DataRate=1000/(sendTime-lastTousandPacktesTimeStamp)
                    print(str(self.packetssend) + " Packets sent Updaterate is "+str(self.DataRate))
                    lastTousandPacktesTimeStamp=sendTime
                self.packetssend = self.packetssend + 1
            if next_time_dscp - time.time() < 0:
                self.__sendDescription()
                print("description sent")
                next_time_dscp = next_time_dscp + delta_t_dscp

    def stop(self):
        """
        Stops the sensor simulator Task.

        Returns
        -------
        None.

        """
        print("Stopping SensorSimulator")
        self._stop_event.set()
    def __sendDataMsg(self,line):
        """
        Sends out data.

        Returns
        -------
        None.

        """
        # setting up the proto message
        protodata = messages_pb2.DataMessage()
        protodata.id = int(line[0])
        protodata.sample_number = int(line[1])
        protodata.unix_time = int(line[2])
        protodata.unix_time_nsecs = int(line[3])
        protodata.time_uncertainty = int(line[4])
        for i in range(16):
            protodata.__setattr__(self.dataFieldNames[i],float(line[5+i]))
        binproto = protodata.SerializeToString()
        # add DATA peramble for data Pacet
        binarymessage = b"DATA"
        binarymessage = binarymessage + _VarintBytes(len(binproto)) + binproto
        self.socket.sendto(
            binarymessage, (self.params["TargetIp"], self.params["Port"])
        )

    def __sendDescription(self):
        """
        Sends out an description.

        Returns
        -------
        None.

        """
        # 0: "PHYSICAL_QUANTITY",1: "UNIT",3: "RESOLUTION",4: "MIN_SCALE",5: "MAX_SCALE",
        DescriptonType = {0: "str", 1: "str", 3: "float", 4: "float", 5: "float"}
        DescriptonKeys = {0: "PHYSICAL_QUANTITY", 1: "UNIT", 3: "RESOLUTION", 4: "MIN_SCALE", 5: "MAX_SCALE"}
        for desckeys in DescriptonKeys :
            proto_description = messages_pb2.DescriptionMessage()
            proto_description.Sensor_name = self.Description.SensorName
            proto_description.id = self.Description.ID
            proto_description.Description_Type = desckeys
            for i in range(16):
                try:
                    descVal=self.Description[i+1][DescriptonKeys[desckeys]]
                    if DescriptonType[desckeys] == "str":
                        proto_description.__setattr__(self.strFieldNames[i],descVal)
                    if DescriptonType[desckeys] == "float":
                        proto_description.__setattr__(self.fFieldNames[i], descVal)
                except KeyError:
                    pass
            binproto = proto_description.SerializeToString()
            binarymessage = b"DSCP"
            binarymessage = binarymessage + _VarintBytes(len(binproto)) + binproto
            self.socket.sendto(
                binarymessage, (self.params["TargetIp"], self.params["Port"])
            )
        return

    def __repr__(self):
        return "Dataplayer: playing "+str(self.params["fileName"])

if __name__ == "__main__":
    Player=SensorDataPlayer('../data/2020-03-03_Messungen_MPU9250_SN12 Frequenzgang_Firmware_0.3.0/mpu9250_12_10_hz_250_Hz_6wdh.dump')

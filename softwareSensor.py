import socket
import messages_pb2
from google.protobuf.internal.encoder import _VarintBytes
import threading
import time
import numpy as np
DescriptonType ={'PHYSICAL_QUANTITY': {'id':0,'type':"str"},
'UNIT': {'id':1,'type':"str"},
#"UNCERTAINTY_TYPE":{'id':2,'type':"str"}, #unsed at the moment
"RESOLUTION":{'id':3,'type':"float"},
"MIN_SCALE": {'id':4,'type':"float"},
"MAX_SCALE": {'id':5,'type':"float"},
"HIERARCHY":{'id':6,'type':"str"},
}
class SoftwareSensor:
    """
    Class for simulationg an MET4FoF compatible Sensor
    """

    def __init__(
        self,
        tagetip,
        port,
        id,
        name,
        description,
        paramupdateratehz=0.5

    ):
        self.ID=id
        self.name=name
        self.port=port
        self.targetIP=tagetip
        self.paramUpadteRate=paramupdateratehz
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.description=description
        self.msgcount = 0
        self.lastTimestamp = 0
        self._stop_event = threading.Event()
        self.packetssend = 0
        self.thread = threading.Thread(
            target=self.run, name='{:08X}'.format(self.ID)+" "+self.name+" SoftwareSensor", args=()
        )
        self.packetssend = 0
        self.thread.start()
    def run(self):

        delta_t_dscp = 1 / self.paramUpadteRate
        while not self._stop_event.is_set():
            self.sendDescription()
            print("description sent")
            time.sleep(delta_t_dscp)

    def stop(self):
        print("Stopping SensorSimulator")
        self._stop_event.set()
    def sendDataMsg(self,data):
        tmptime = time.time()
        secs = int(np.floor(tmptime))
        nsecs = int((tmptime - secs) * 1e9)
        # setting up the proto message
        protodata = messages_pb2.DataMessage()
        protodata.id = self.ID
        protodata.sample_number = self.packetssend
        protodata.unix_time = secs
        protodata.unix_time_nsecs = nsecs
        protodata.time_uncertainty = 1000000
        # this is nasty in protobuff 3 there is an option to iterate over fields
        # or we should youse repeded fields
        l=data.shape[0]
        if l > 0:
            protodata.Data_01 = data[0]
        if l > 1:
            protodata.Data_02 = data[1]
        if l > 2:
            protodata.Data_03 = data[2]
        if l > 3:
            protodata.Data_04 = data[3]
        if l > 4:
            protodata.Data_04 = data[4]
        if l > 5:
            protodata.Data_04 = data[5]
        if l > 6:
            protodata.Data_04 = data[6]
        if l > 7:
            protodata.Data_04 = data[7]
        if l > 8:
            protodata.Data_04 = data[8]
        if l > 9:
            protodata.Data_04 = data[9]
        if l > 10:
            protodata.Data_04 = data[10]
        if l > 11:
            protodata.Data_04 = data[11]
        if l > 12:
            protodata.Data_04 = data[12]
        if l > 13:
            protodata.Data_04 = data[13]
        if l > 14:
            protodata.Data_04 = data[14]
        if l > 15:
            protodata.Data_04 = data[15]
        if l > 16:
            protodata.Data_04 = data[16]
        binproto = protodata.SerializeToString()
        binarymessage = b"DATA"
        binarymessage = binarymessage + _VarintBytes(len(binproto)) + binproto
        self.socket.sendto(binarymessage, (self.targetIP, self.port))
        self.packetssend = self.packetssend + 1
        return
    
    def sendDescription(self):
        for desckeys in Description.keys():
            proto_description = messages_pb2.DescriptionMessage()
            proto_description.Sensor_name = self.name
            proto_description.id = self.ID
            proto_description.Description_Type = DescriptonType[desckeys]['id']
            l=len(self.description[desckeys])
            if DescriptonType[desckeys]['type'] == "str":
                if l > 0:
                    proto_description.str_Data_01 = self.description[desckeys][0]
                if l > 1:
                    proto_description.str_Data_02 = self.description[desckeys][1]
                if l > 2:
                    proto_description.str_Data_03 = self.description[desckeys][2]
                if l > 3:
                    proto_description.str_Data_04 = self.description[desckeys][3]
                if l > 4:
                    proto_description.str_Data_05 = self.description[desckeys][4]
                if l > 5:
                    proto_description.str_Data_06 = self.description[desckeys][5]
                if l > 6:
                    proto_description.str_Data_07 = self.description[desckeys][6]
                if l > 7:
                    proto_description.str_Data_08 = self.description[desckeys][7]
                if l > 8:
                    proto_description.str_Data_09 = self.description[desckeys][8]
                if l > 9:
                    proto_description.str_Data_10 = self.description[desckeys][9]
                if l > 10:
                    proto_description.str_Data_11 = self.description[desckeys][10]
                if l > 11:
                    proto_description.str_Data_12 = self.description[desckeys][11]
                if l > 12:
                    proto_description.str_Data_13 = self.description[desckeys][12]
                if l > 13:
                    proto_description.str_Data_14 = self.description[desckeys][13]
                if l > 14:
                    proto_description.str_Data_15 = self.description[desckeys][14]
                if l > 15:
                    proto_description.str_Data_16 = self.description[desckeys][15]
            if DescriptonType[desckeys]['type'] == "float":
                if l > 0:
                    proto_description.f_Data_01 = self.description[desckeys][0]
                if l > 1:
                    proto_description.f_Data_02 = self.description[desckeys][1]
                if l > 2:
                    proto_description.f_Data_03 = self.description[desckeys][2]
                if l > 3:
                    proto_description.f_Data_04 = self.description[desckeys][3]
                if l > 4:
                    proto_description.f_Data_05 = self.description[desckeys][4]
                if l > 5:
                    proto_description.f_Data_06 = self.description[desckeys][5]
                if l > 6:
                    proto_description.f_Data_07 = self.description[desckeys][6]
                if l > 7:
                    proto_description.f_Data_08 = self.description[desckeys][7]
                if l > 8:
                    proto_description.f_Data_09 = self.description[desckeys][8]
                if l > 9:
                    proto_description.f_Data_10 = self.description[desckeys][9]
                if l > 10:
                    proto_description.f_Data_11 = self.description[desckeys][10]
                if l > 11:
                    proto_description.f_Data_12 = self.description[desckeys][11]
                if l > 12:
                    proto_description.f_Data_13 = self.description[desckeys][12]
                if l > 13:
                    proto_description.f_Data_14 = self.description[desckeys][13]
                if l > 14:
                    proto_description.f_Data_15 = self.description[desckeys][14]
                if l > 15:
                    proto_description.f_Data_16 = self.description[desckeys][15]
            proto_description.has_time_ticks=False
            binproto = proto_description.SerializeToString()
            binarymessage = b"DSCP"
            binarymessage = binarymessage + _VarintBytes(len(binproto)) + binproto
            self.socket.sendto(binarymessage, (self.targetIP, self.port))


if __name__=="__main__":
    Description = {
            'PHYSICAL_QUANTITY': ["X Tilt",   "Y Tilt", "X Voltage", "Y Voltage"],
            'UNIT': [r"\radian", r"\radian", r"\volt", r"\volt"],
            "RESOLUTION": [16777216, 16777216, 16777216, 16777216],
            "MIN_SCALE": [-0.0006086,- 0.0006618, -5.0, -5.0],
            "MAX_SCALE": [0.0006086, 0.0006618, 5.0, 5.0],
            "HIERARCHY": ["Tilt/0","Tilt/1","Voltage/0","Voltage/1"]
        }
    swSensor=SoftwareSensor("192.168.0.200",7654,0x13370100,"Simulated Tiltmeter",Description)
    while True:
        swSensor.sendDataMsg(np.arange(4))
        time.sleep(0.1)

import socket
import messages_pb2
import google.protobuf as pb
from google.protobuf.internal.encoder import _VarintBytes
import threading
import time
import numpy as np

class SensorSimulator:
    def __init__(self,updateratehz=1000,tagetip="127.0.0.1",port=7654,id=0x00000001,resolutionbit=8):
        self.flags = {"Networtinited": False}
        self.params = {"TargetIp": tagetip,
                       "Port": port,
                       "UpdateRateHz": updateratehz,
                       "ID":id,
                       "resolutionbit":resolutionbit}
        self.socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM)
        self.flags["Networtinited"] = True
        self.msgcount = 0
        self.lastTimestamp = 0
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name="Sensor Simulator", args=())
        self.packetssend=0
        self.thread.start()


    def run(self):
        firsttime=time.time()
        DeltaT=1/self.params["UpdateRateHz"]
        nexttime=firsttime+DeltaT

        while not self._stop_event.is_set():
            # TODO improve time scheduling
            if nexttime-time.time()<0:
                self.sendDataMsg()
                self.sendDescription()
                nexttime=nexttime+DeltaT

            #+geting actual time


    def stop(self):
        print("Stopping SensorSimulator")
        self._stop_event.set()


    def sendDataMsg(self):
        tmptime = time.time()
        secs = int(np.floor(tmptime))
        nsecs = int((tmptime - secs) * 1e9)
        # setting up the proto message
        protodata = messages_pb2.DataMessage()
        protodata.id = self.params["ID"]
        protodata.sample_number = self.packetssend
        protodata.unix_time = secs
        protodata.unix_time_nsecs = nsecs
        protodata.time_uncertainty = 1000000
        res=2**self.params["resolutionbit"]
        tmp=(nsecs/1e9)*(res-1)-res/2
        qunatizedint=int(tmp)
        qunatizedfloat=qunatizedint/res
        protodata.Data_01 = np.sin(qunatizedfloat)
        protodata.Data_02 = np.cos(qunatizedfloat)
        protodata.Data_03 = (qunatizedfloat)
        protodata.Data_04 = abs(qunatizedfloat)
        binproto = protodata.SerializeToString()
        # add DATA peramble for data Pacet
        binarymessage = b"DATA"
        binarymessage = binarymessage + _VarintBytes(len(binproto)) + binproto
        self.socket.sendto(binarymessage, (self.params["TargetIp"], self.params["Port"]))
        self.packetssend = self.packetssend + 1
        if self.packetssend%self.params["UpdateRateHz"]==0:
            print(str(self.packetssend)+" Packets send")
        return

    def sendDescription(self):
        res=2**self.params["resolutionbit"]
        max=((res-1)-res/2)/res
        min=((res/2)-res)/res
        Descripton={
            0: ["Sin","Cos","Sawtooth","Triangle"],#0: "PHYSICAL_QUANTITY",
            1: ["A.U.","A.U.","A.U.","A.U."],# 1: "UNIT",
            3: [res,res,res,res],#3: "RESOLUTION",
            4: [np.sin(max),np.cos(max),max,abs(max)],
            5: [np.sin(min),np.cos(min),max,abs(min)]
        }
        DescriptionTypNames = {


            2: "UNCERTAINTY_TYPE",
            3: "RESOLUTION",
            4: "MIN_SCALE",
            5: "MAX_SCALE",
        }
        for desckeys in Descripton.keys():
            ProtoDescription = messages_pb2.DescriptionMessage()
            ProtoDescription.Sensor_name="Sensor Simulation"
            ProtoDescription.Description_Type=desckeys
            binproto = protodescription.SerializeToString()
            binarymessage = b"DSCP"
            binarymessage = binarymessage + _VarintBytes(len(binproto)) + binproto



if __name__ == "__main__":
    sensorsim = SensorSimulator()

import socket
import messages_pb2
import google.protobuf as pb
from google.protobuf.internal.encoder import _VarintBytes
import threading
import time
import numpy as np


class SensorSimulator:
    """
    Class for simulationg an MET4FoF compatible Sensor
    """

    def __init__(
        self,
        updateratehz=1000,
        tagetip="127.0.0.1",
        port=7654,
        id=0x00000001,
        resolutionbit=8,
        paramupdateratehz=0.5,
    ):
        """


        Parameters
        ----------
        updateratehz : integer, optional
            Data update frequency in Hz. The default is 1000.
        tagetip : sting, optional
            IP Adress of the DataReceiver. The default is "127.0.0.1".
        port : integer, optional
            UDP Port of the DataReceiver. The default is 7654.
        id : integer, optional
            ID of the simulated sensor. The default is 0x00000001.
        resolutionbit : integer, optional
            The simulated Sensor data are quantized in 2^resolutionbit  steps between min and max. The default is 8.
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
            "UpdateRateHz": updateratehz,
            "ParamUpdateRateHz": paramupdateratehz,
            "ID": id,
            "resolutionbit": resolutionbit,
        }
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.flags["Networtinited"] = True
        self.msgcount = 0
        self.lastTimestamp = 0
        self._stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self.run, name="Sensor Simulator", args=()
        )
        self.packetssend = 0
        self.thread.start()

    def run(self):
        """
        Starts the sensor simulator Task.

        Returns
        -------
        None.

        """
        firsttime = time.time()
        delta_t_data = 1 / self.params["UpdateRateHz"]
        next_time_data = firsttime + delta_t_data
        delta_t_dscp = 1 / self.params["ParamUpdateRateHz"]
        next_time_dscp = firsttime + delta_t_dscp

        while not self._stop_event.is_set():
            # TODO improve time scheduling
            if next_time_data - time.time() < 0:
                self.__sendDataMsg()
                next_time_data = next_time_data + delta_t_data
            if next_time_dscp - time.time() < 0:
                self.__sendDescription()
                print("description sent")
                next_time_dscp = next_time_dscp + delta_t_dscp

            # +geting actual time

    def stop(self):
        """
        Stops the sensor simulator Task.

        Returns
        -------
        None.

        """
        print("Stopping SensorSimulator")
        self._stop_event.set()

    def __sendDataMsg(self):
        """
        Sends out simulated data.

        Returns
        -------
        None.

        """
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
        res = 2 ** self.params["resolutionbit"]
        tmp = (nsecs / 1e9) * (res - 1) - res / 2
        qunatizedint = int(tmp)
        qunatizedfloat = qunatizedint / res * 2
        protodata.Data_01 = np.sin(qunatizedfloat * np.pi)
        protodata.Data_02 = np.cos(qunatizedfloat * np.pi)
        protodata.Data_03 = qunatizedfloat
        protodata.Data_04 = abs(qunatizedfloat)
        binproto = protodata.SerializeToString()
        # add DATA peramble for data Pacet
        binarymessage = b"DATA"
        binarymessage = binarymessage + _VarintBytes(len(binproto)) + binproto
        self.socket.sendto(
            binarymessage, (self.params["TargetIp"], self.params["Port"])
        )
        self.packetssend = self.packetssend + 1
        if self.packetssend % self.params["UpdateRateHz"] == 0:
            print(str(self.packetssend) + " Packets sent")
        return

    def __sendDescription(self):
        """
        Sends out an description.

        Returns
        -------
        None.

        """
        res = 2 ** self.params["resolutionbit"]
        max = ((res - 1) - res / 2) / res
        min = ((res / 2) - res) / res
        Description = {
            0: ["Sin", "Cos", "Sawtooth", "Triangle"],  # 0: "PHYSICAL_QUANTITY",
            1: ["A.U.", "A.U.", "A.U.", "A.U."],  # 1: "UNIT",
            3: [res, res, res, res],  # 3: "RESOLUTION",
            4: [np.sin(max), np.cos(max), max, abs(max)],  # 4: "MIN_SCALE",
            5: [np.sin(min), np.cos(min), max, abs(min)],  #'5: "MAX_SCALE",
        }
        DescriptonType = {0: "str", 1: "str", 3: "float", 4: "float", 5: "float"}
        for desckeys in Description.keys():
            proto_description = messages_pb2.DescriptionMessage()
            proto_description.Sensor_name = "Sensor Simulation"
            proto_description.id = self.params["ID"]
            proto_description.Description_Type = desckeys
            if DescriptonType[desckeys] == "str":
                proto_description.str_Data_01 = Description[desckeys][0]
                proto_description.str_Data_02 = Description[desckeys][1]
                proto_description.str_Data_03 = Description[desckeys][2]
                proto_description.str_Data_04 = Description[desckeys][3]
            if DescriptonType[desckeys] == "float":
                proto_description.f_Data_01 = Description[desckeys][0]
                proto_description.f_Data_02 = Description[desckeys][1]
                proto_description.f_Data_03 = Description[desckeys][2]
                proto_description.f_Data_04 = Description[desckeys][3]
            binproto = proto_description.SerializeToString()
            binarymessage = b"DSCP"
            binarymessage = binarymessage + _VarintBytes(len(binproto)) + binproto


if __name__ == "__main__":
    sensorsim = SensorSimulator()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:20:12 2019

Data receiver for Met4FoF Protobuff Data
@author: Benedikt.Seeger@ptb.de
"""

import sys
import traceback
import os
import socket
import threading

import warnings
from datetime import datetime
from multiprocessing import Queue
import time
import copy
import json

# for live plotting
import matplotlib.pyplot as plt
import numpy as np

# proptobuff message encoding
from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURR_DIR)
sys.path.append(CURR_DIR)
import messages_pb2


from uncertainties import ufloat
from uncertainties import unumpy
# matplotlib.use('Qt5Agg')


class DataReceiver:
    """Class for handlig the incomming UDP Packets and spwaning sensor Tasks and sending the Protobuff Messages over an queue to the Sensor Task

    .. image:: ../doc/DR_flow.png

    """

    def __init__(self, IP, Port=7654):
        """


        Parameters
        ----------
        IP : string
            Either an spefic IP Adress like "192.168.0.200" or "" for all interfaces.
        Port : intger
            UDP Port for the incoming data 7654 is default.

        Raises
        ------
        socket.error:[errno 99] cannot assign requested address and namespace in python
            The Set IP does not match any networkintrefaces ip.
        socket.error:[Errno 98] Address already in use
            an other task is using the set port and interface.
        Returns
        -------
        None.

        """
        self.flags = {"Networtinited": False}
        self.params = {"IP": IP, "Port": Port, "PacketrateUpdateCount": 10000}
        self.socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM  # Internet
        )  # UDP

        # Try to open the UDP connection
        try:
            self.socket.bind((IP, Port))
        except OSError as err:
            print("OS error: {0}".format(err))
            if err.errno == 99:
                print(
                    "most likely no network card of the system has the ip address"
                    + str(IP)
                    + " check this with >>> ifconfig on linux or with >>> ipconfig on Windows"
                )
            if err.errno == 98:
                print(
                    "an other task is blocking the connection on linux use >>> sudo netstat -ltnp | grep -w ':"
                    + str(Port)
                    + "' on windows use in PowerShell >>> Get-Process -Id (Get-NetTCPConnection -LocalPort "
                    + str(Port)
                    + ").OwningProcess"
                )
            raise (err)
            # we need to raise an exception to prevent __init__ from returning
            # otherwise a broken class instance will be created
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise ("Unexpected error:", sys.exc_info()[0])
        self.flags["Networtinited"] = True
        self.packestlosforsensor = {}
        self.AllSensors = {}
        self.msgcount = 0
        self.lastTimestamp = 0
        self.Datarate = 0
        self._stop_event = threading.Event()
        # start thread for data processing
        self.thread = threading.Thread(
            target=self.run, name="Datareceiver_thread", args=()
        )
        self.thread.start()
        print("Data receiver now running wating for Packates")

    def __repr__(self):
        """
        Prints IP and Port as well as list of all sensors (self.AllSensors).

        Returns
        -------
        None.

        """
        return (
            "Datareceiver liestening at ip "
            + str(self.params["IP"])
            + " Port "
            + str(self.params["Port"])
            + "\n Active Snesors are:"
            + str(self.AllSensors)
        )

    def stop(self):
        """
        Stops the Datareceiver task and closes the UDP socket.

        Returns
        -------
        None.

        """
        print("Stopping DataReceiver")
        self._stop_event.set()
        # wait 1 second to ensure that all ques are empty before closing them
        # other wise SIGPIPE is raised by os
        # IMPORVEMNT use signals for this
        time.sleep(1)
        for key in self.AllSensors:
            self.AllSensors[key].stop()
        self.socket.close()

    def run(self):
        """
        Spwans the Datareceiver task.

        Returns
        -------
        None.

        """
        # implement stop routine
        while not self._stop_event.is_set():
            data, addr = self.socket.recvfrom(1500)  # buffer size is 1024 bytes
            wasValidData = False
            wasValidDescription = False
            ProtoData = messages_pb2.DataMessage()
            ProtoDescription = messages_pb2.DescriptionMessage()
            SensorID = 0
            BytesProcessed = 4  # we need an offset of 4 sice
            if data[:4] == b"DATA":
                while BytesProcessed < len(data):
                    msg_len, new_pos = _DecodeVarint32(data, BytesProcessed)
                    BytesProcessed = new_pos

                    try:
                        msg_buf = data[new_pos : new_pos + msg_len]
                        ProtoData.ParseFromString(msg_buf)
                        wasValidData = True
                        SensorID = ProtoData.id
                        message = {"ProtMsg": copy.deepcopy(ProtoData), "Type": "Data"}
                        BytesProcessed += msg_len
                    except:
                        pass  # ? no exception for wrong data type !!
                    if not (wasValidData or wasValidDescription):
                        print("INVALID PROTODATA")
                        pass  # invalid data leave parsing routine

                    if SensorID in self.AllSensors:
                        try:
                            self.AllSensors[SensorID].buffer.put_nowait(message)
                        except:
                            tmp = self.packestlosforsensor[SensorID] = (
                                self.packestlosforsensor[SensorID] + 1
                            )
                            if tmp == 1:
                                print("!!!! FATAL PERFORMANCE PROBLEMS !!!!")
                                print(
                                    "FIRSTTIME packet lost for sensor ID:"
                                    + str(SensorID)
                                )
                                print(
                                    "DROP MESSAGES ARE ONLY PRINTETD EVERY 1000 DROPS FROM NOW ON !!!!!!!! "
                                )
                            if tmp % 1000 == 0:
                                print("oh no lost an other  thousand packets :(")
                    else:
                        self.AllSensors[SensorID] = Sensor(SensorID)
                        print(
                            "FOUND NEW SENSOR WITH ID=hex"
                            + hex(SensorID)
                            + "==>dec:"
                            + str(SensorID)
                        )
                        self.packestlosforsensor[
                            SensorID
                        ] = 0  # initing lost packet counter
                    self.msgcount = self.msgcount + 1

                    if self.msgcount % self.params["PacketrateUpdateCount"] == 0:
                        print(
                            "received "
                            + str(self.params["PacketrateUpdateCount"])
                            + " packets"
                        )
                        if self.lastTimestamp != 0:
                            timeDIFF = datetime.now() - self.lastTimestamp
                            timeDIFF = timeDIFF.seconds + timeDIFF.microseconds * 1e-6
                            self.Datarate = (
                                self.params["PacketrateUpdateCount"] / timeDIFF
                            )
                            print("Update rate is " + str(self.Datarate) + " Hz")
                            self.lastTimestamp = datetime.now()
                        else:
                            self.lastTimestamp = datetime.now()
            elif data[:4] == b"DSCP":
                while BytesProcessed < len(data):
                    msg_len, new_pos = _DecodeVarint32(data, BytesProcessed)
                    BytesProcessed = new_pos
                    try:
                        msg_buf = data[new_pos : new_pos + msg_len]
                        ProtoDescription.ParseFromString(msg_buf)
                        # print(msg_buf)
                        wasValidData = True
                        SensorID = ProtoDescription.id
                        message = {"ProtMsg": ProtoDescription, "Type": "Description"}
                        BytesProcessed += msg_len
                    except:
                        pass  # ? no exception for wrong data type !!
                    if not (wasValidData or wasValidDescription):
                        print("INVALID PROTODATA")
                        pass  # invalid data leave parsing routine

                    if SensorID in self.AllSensors:
                        try:
                            self.AllSensors[SensorID].buffer.put_nowait(message)
                        except:
                            print("packet lost for sensor ID:" + hex(SensorID))
                    else:
                        self.AllSensors[SensorID] = Sensor(SensorID)
                        print(
                            "FOUND NEW SENSOR WITH ID=hex"
                            + hex(SensorID)
                            + " dec==>:"
                            + str(SensorID)
                        )
                    self.msgcount = self.msgcount + 1

                    if self.msgcount % self.params["PacketrateUpdateCount"] == 0:
                        print(
                            "received "
                            + str(self.params["PacketrateUpdateCount"])
                            + " packets"
                        )
                        if self.lastTimestamp != 0:
                            timeDIFF = datetime.now() - self.lastTimestamp
                            timeDIFF = timeDIFF.seconds + timeDIFF.microseconds * 1e-6
                            self.Datarate = (
                                self.params["PacketrateUpdateCount"] / timeDIFF
                            )
                            print("Update rate is " + str(self.Datarate) + " Hz")
                            self.lastTimestamp = datetime.now()
                        else:
                            self.lastTimestamp = datetime.now()
            else:
                print("unrecognized packed preamble" + str(data[:5]))

    def __del__(self):
        """
        just for securtiy closes the socket if __del__ is called.

        Returns
        -------
        None.

        """
        self.socket.close()


### classes to proces sensor descriptions
class AliasDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.aliases = {}

    def __getitem__(self, key):
        return dict.__getitem__(self, self.aliases.get(key, key))

    def __setitem__(self, key, value):
        return dict.__setitem__(self, self.aliases.get(key, key), value)

    def add_alias(self, key, alias):
        self.aliases[alias] = key


class ChannelDescription:
    def __init__(self, CHID):
        """


        Parameters
        ----------
        CHID : intger
            ID of the channel startig with 1.

        Returns
        -------
        None.

        """
        self.Description = {
            "CHID": CHID,
            "PHYSICAL_QUANTITY": False,
            "UNIT": False,
            "RESOLUTION": False,
            "MIN_SCALE": False,
            "MAX_SCALE": False,
        }
        self._complete = False

    def __getitem__(self, key):
        # if key='SpecialKey':
        # self.Description['SpecialKey']
        return self.Description[key]

    def __setitem__(self, key,item):
        # if key='SpecialKey':
        # self.Description['SpecialKey']
        self.Description[key]=item

    def __repr__(self):
        """
        Prints the quantity and unit of the channel.
        """
        return (
            "Channel: "
            + str(self.Description["CHID"])
            + " ==>"
            + str(self.Description["PHYSICAL_QUANTITY"])
            + " in "
            + str(self.Description["UNIT"])
        )

    # todo override set methode
    def setDescription(self, key, value):
        """
        Sets an spefic key of an channel description.

        Parameters
        ----------
        key : string
            PHYSICAL_QUANTITY",UNIT,RESOLUTION,MIN_SCALE or MAX_SCALE.
        value : string or intger
            valuie coresponding to the key.
        Returns
        -------
        None.

        """
        self.Description[key] = value
        if (
            self.Description["PHYSICAL_QUANTITY"] != False
            and self.Description["UNIT"] != False
            and self.Description["RESOLUTION"] != False
            and self.Description["MIN_SCALE"] != False
            and self.Description["MAX_SCALE"] != False
        ):
            self._complete = True


class SensorDescription:
    """
    this class is holding the Sensor description.
    It's subscriptable by :
    1. inter number of the channel   eg.g. SensorDescription[1]
    2. Name of The physical quantity SensorDescription["Temperature"]
    3. Name of the data field SensorDescription["Data_01"]
    """

    def __init__(self, ID=0x00000000, SensorName="undefined", fromDict=None):
        """


        Parameters
        ----------
        ID : uint32
            ID of the Sensor.The default is 0x00000000
        SensorName : sting
            Name of the sensor.The default is "undefined".
        fromDict : dict
            If an Description dict is passed the Channel params will be set accordingly.
        Returns
        -------
        None.

        """
        self.ID = ID
        self.SensorName = SensorName
        self._complete = False
        self.Channels = AliasDict([])
        self.ChannelCount = 0
        self._ChannelsComplte = 0
        if type(fromDict) is dict:
            try:
                self.ID = fromDict["ID"]
            except KeyError:
                warnings.warn("ID not in Dict", RuntimeWarning)
            try:
                self.SensorName = fromDict["Name"]
            except KeyError:
                warnings.warn("Name not in Dict", RuntimeWarning)
            for i in range(16):
                try:
                    channelDict = fromDict[i]
                    for key in channelDict.keys():
                        if key == "CHID":
                            pass
                        else:
                            self.setChannelParam(
                                channelDict["CHID"], key, channelDict[key]
                            )
                    print("Channel " + str(i) + " read from dict")
                except KeyError:
                    #ok maybe the channels are coded as string
                    try:
                        channelDict = fromDict[str(i)]
                        for key in channelDict.keys():
                            if key == "CHID":
                                pass
                            else:
                                self.setChannelParam(
                                    channelDict["CHID"], key, channelDict[key]
                                )
                        print("Channel " + str(i) + " read from dict")
                    except KeyError:
                        pass

    def setChannelParam(self, CHID, key, value):
        """
        Set parametes for an specific channel.

        Parameters
        ----------
        CHID : intger
            ID of the channel startig with 1.
        key : string
            PHYSICAL_QUANTITY",UNIT,RESOLUTION,MIN_SCALE or MAX_SCALE.
        value : string or intger
            valuie coresponding to the key.

        Returns
        -------
        None.

        """
        wasComplete = False
        if CHID in self.Channels:
            wasComplete = self.Channels[
                CHID
            ]._complete  # read if channel was completed before
            self.Channels[CHID].setDescription(key, value)
            if key == "PHYSICAL_QUANTITY":
                self.Channels.add_alias(
                    CHID, value
                )  # make channels callable by their Quantity
        else:
            if key == "PHYSICAL_QUANTITY":
                self.Channels.add_alias(
                    CHID, value
                )  # make channels callable by their Quantity
            self.Channels[CHID] = ChannelDescription(CHID)
            self.Channels[CHID].setDescription(key, value)
            self.Channels.add_alias(
                CHID, "Data_" + "{:02d}".format(CHID)
            )  # make channels callable by ther Data_xx name
            self.ChannelCount = self.ChannelCount + 1
        if wasComplete == False and self.Channels[CHID]._complete:
            self._ChannelsComplte = self._ChannelsComplte + 1
            if self._ChannelsComplte == self.ChannelCount:
                self._complete = True
                print("Description completed")

    def __getitem__(self, key):
        """
        Reutrns the description for an channel callable by Channel ID eg 1, Channel name eg. Data_01 or Physical PHYSICAL_QUANTITY eg. Acceleration_x.

        Parameters
        ----------
        key : sting or int
            Channel ID eg 1, Channel name eg. "Data_01" or Physical PHYSICAL_QUANTITY eg. "X Acceleration".

        Returns
        -------
        ChannelDescription
            The description of the channel.

        """
        # if key='SpecialKey':
        # self.Description['SpecialKey']
        return self.Channels[key]


    def __repr__(self):
        return "Descripton of" + self.SensorName + hex(self.ID)

    def asDict(self):
        """
        ChannelDescription as dict.

        Returns
        -------
        ReturnDict : dict
            ChannelDescription as dict.

        """
        ReturnDict = {"Name": self.SensorName, "ID": self.ID}
        for key in self.Channels:
            print(self.Channels[key].Description)
            ReturnDict.update(
                {self.Channels[key]["CHID"]: self.Channels[key].Description}
            )
        return ReturnDict

    def getUnits(self):
        units = {}
        for Channel in self.Channels:
            if self.Channels[Channel]["UNIT"] in units:
                units[self.Channels[Channel]["UNIT"]].append(Channel)
            else:
                units[self.Channels[Channel]["UNIT"]] = [Channel]
        return units

    def getActiveChannelsIDs(self):
        return self.Channels.keys()

def doNothingCb():
    pass

class Sensor:
    """Class for Processing the Data from Datareceiver class. All instances of this class will be swaned in Datareceiver.AllSensors

    .. image:: ../doc/Sensor_loop.png

    """

    StrFieldNames = [
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
    FFieldNames = [
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
    DescriptionTypNames = {
        0: "PHYSICAL_QUANTITY",
        1: "UNIT",
        2: "UNCERTAINTY_TYPE",
        3: "RESOLUTION",
        4: "MIN_SCALE",
        5: "MAX_SCALE",
    }

    def __init__(self, ID, BufferSize=25e5):
        """
        Constructor for the Sensor class

        Parameters
        ----------
        ID : uint32
            ID of the Sensor.
        BufferSize : integer, optional
            Size of the Data Queue. The default is 25e5.

        Returns
        -------
        None.

        """
        self.Description = SensorDescription(ID, "Name not Set")
        self.buffer = Queue(int(BufferSize))
        self.buffersize = BufferSize
        self.flags = {
            "DumpToFile": False,
            "DumpToFileProto":False,
            "DumpToFileASCII": False,
            "PrintProcessedCounts": True,
            "callbackSet": False,
        }
        self.params = {"ID": ID, "BufferSize": BufferSize, "DumpFileName": ""}
        self.DescriptionsProcessed = AliasDict(
            {
                "PHYSICAL_QUANTITY": False,
                "UNIT": False,
                "UNCERTAINTY_TYPE": False,
                "RESOLUTION": False,
                "MIN_SCALE": False,
                "MAX_SCALE": False,
            }
        )
        for i in range(6):
            self.DescriptionsProcessed.add_alias(self.DescriptionTypNames[i], i)
        self._stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self.run, name="Sensor_" + str(ID) + "_thread", args=()
        )
        # self.thread.daemon = True
        self.thread.start()
        self.ProcessedPacekts = 0
        self.datarate = 0
        self.timeoutOccured=False
        self.timeSinceLastPacket = 0

    def __repr__(self):
        """
        prints the Id and sensor name.

        Returns
        -------
        None.

        """
        return hex(self.Description.ID) + " " + self.Description.SensorName

    def StartDumpingToFileASCII(self, filename=""):
        """
        Activate dumping Messages in a file ASCII encoded ; seperated.

        Parameters
        ----------
        filename : path
            path to the dumpfile.

        Returns
        -------
        None.

        """
        # check if the path is valid
        # if(os.path.exists(os.path.dirname(os.path.abspath('data/dump.csv')))):
        if filename == "":
            now = datetime.now()
            filename = (
                "data/"
                + now.strftime("%Y%m%d%H%M%S")
                + "_"
                + str(self.Description.SensorName).replace(" ", "_")
                + "_"
                + hex(self.Description.ID)
                + ".dump"
            )
        self.DumpfileASCII = open(filename, "a")
        json.dump(self.Description.asDict(), self.DumpfileASCII)
        self.DumpfileASCII.write("\n")
        self.DumpfileASCII.write(
            "id;sample_number;unix_time;unix_time_nsecs;time_uncertainty;Data_01;Data_02;Data_03;Data_04;Data_05;Data_06;Data_07;Data_08;Data_09;Data_10;Data_11;Data_12;Data_13;Data_14;Data_15;Data_16\n"
        )
        self.params["DumpFileNameASCII"] = filename
        self.flags["DumpToFileASCII"] = True

    def StopDumpingToFileASCII(self):
        """
        Stops dumping to file ASCII encoded.

        Returns
        -------
        None.

        """
        self.flags["DumpToFileASCII"] = False
        self.params["DumpFileNameASCII"] = ""
        self.DumpfileASCII.close()

    def StartDumpingToFileProto(self, filename=""):
        """
        Activate dumping Messages in a file ProtBuff encoded \\n seperated.

        Parameters
        ----------
        filename : path
            path to the dumpfile.


        Returns
        -------
        None.

        """
        # check if the path is valid
        # if(os.path.exists(os.path.dirname(os.path.abspath('data/dump.csv')))):
        if filename == "":
            now = datetime.now()
            filename = (
                "data/"
                + now.strftime("%Y%m%d%H%M%S")
                + "_"
                + str(self.Description.SensorName).replace(" ", "_")
                + "_"
                + hex(self.Description.ID)
                + ".protodump"
            )
        self.DumpfileProto = open(filename, "a")
        json.dump(self.Description.asDict(), self.DumpfileProto)
        self.DumpfileProto.write("\n")
        self.DumpfileProto = open(filename, "ab")
        self.params["DumpFileNameProto"] = filename
        self.flags["DumpToFileProto"] = True

    def StopDumpingToFileProto(self):
        """
        Stops dumping to file Protobuff encoded.

        Returns
        -------
        None.

        """
        self.flags["DumpToFileProto"] = False
        self.params["DumpFileNameProto"] = ""
        self.DumpfileProto.close()

    def run(self):
        """
        Starts the Sensor loop.
        -------
        None.

        """
        while not self._stop_event.is_set():
            # problem when we are closing the queue this function is waiting for data and raises EOF error if we delet the q
            # work around adding time out so self.buffer.get is returning after a time an thestop_event falg can be checked
            try:
                message = self.buffer.get(timeout=0.1)
                self.timeoutOccured = False
                self.ProcessedPacekts = self.ProcessedPacekts + 1
                if self.flags["PrintProcessedCounts"]:
                    if self.ProcessedPacekts % 10000 == 0:
                        print(
                            "processed 10000 packets in receiver for Sensor ID:"
                            + hex(self.params["ID"])
                            + " Packets in Que "
                            + str(self.buffer.qsize())
                            + " -->"
                            + str((self.buffer.qsize() / self.buffersize) * 100)
                            + "%"
                        )
                if message["Type"] == "Description":
                    Description = message["ProtMsg"]
                    try:
                        if (
                            not any(self.DescriptionsProcessed.values())
                            and Description.IsInitialized()
                        ):
                            # run only if no description packed has been procesed ever
                            # self.Description.SensorName=message.Sensor_name
                            print(
                                "Found new description "
                                + Description.Sensor_name
                                + " sensor with ID:"
                                + str(self.params["ID"])
                            )
                            # print(str(Description.Description_Type))
                        if (
                            self.DescriptionsProcessed[Description.Description_Type]
                            == False
                        ):

                            if self.Description.SensorName == "Name not Set":
                                self.Description.SensorName = Description.Sensor_name
                            # we havent processed thiss message before now do that
                            if Description.Description_Type in [
                                0,
                                1,
                                2,
                            ]:  # ["PHYSICAL_QUANTITY","UNIT","UNCERTAINTY_TYPE"]
                                # print(Description)
                                # string Processing

                                FieldNumber = 1
                                for StrField in self.StrFieldNames:
                                    if Description.HasField(StrField):
                                        self.Description.setChannelParam(
                                            FieldNumber,
                                            self.DescriptionTypNames[
                                                Description.Description_Type
                                            ],
                                            Description.__getattribute__(StrField),
                                        )
                                        # print(str(FieldNumber)+' '+Description.__getattribute__(StrField))
                                    FieldNumber = FieldNumber + 1

                                self.DescriptionsProcessed[
                                    Description.Description_Type
                                ] = True
                                # print(self.DescriptionsProcessed)
                            if Description.Description_Type in [
                                3,
                                4,
                                5,
                            ]:  # ["RESOLUTION","MIN_SCALE","MAX_SCALE"]
                                self.DescriptionsProcessed[
                                    Description.Description_Type
                                ] = True
                                FieldNumber = 1
                                for FloatField in self.FFieldNames:
                                    if Description.HasField(FloatField):
                                        self.Description.setChannelParam(
                                            FieldNumber,
                                            self.DescriptionTypNames[
                                                Description.Description_Type
                                            ],
                                            Description.__getattribute__(FloatField),
                                        )
                                        # print(str(FieldNumber)+' '+str(Description.__getattribute__(FloatField)))
                                    FieldNumber = FieldNumber + 1
                                # print(self.DescriptionsProcessed)
                                # string Processing
                    except Exception:
                        print(
                            " Sensor id:"
                            + hex(self.params["ID"])
                            + "Exception in user Description parsing:"
                        )
                        print("-" * 60)
                        traceback.print_exc(file=sys.stdout)
                        print("-" * 60)
                if self.flags["callbackSet"]:
                    if message["Type"] == "Data":
                        try:
                            self.callback(message["ProtMsg"], self.Description)
                        except Exception:
                            print(
                                " Sensor id:"
                                + hex(self.params["ID"])
                                + "Exception in user callback:"
                            )
                            print("-" * 60)
                            traceback.print_exc(file=sys.stdout)
                            print("-" * 60)
                            pass

                if self.flags["DumpToFileProto"]:
                    if message["Type"] == "Data":
                        try:
                            self.__dumpMsgToFileProto(message["ProtMsg"])
                        except Exception:
                            print(
                                " Sensor id:"
                                + hex(self.params["ID"])
                                + "Exception in user datadump:"
                            )
                            print("-" * 60)
                            traceback.print_exc(file=sys.stdout)
                            print("-" * 60)
                            pass
                if self.flags["DumpToFileASCII"]:
                    if message["Type"] == "Data":
                        try:
                            self.__dumpMsgToFileASCII(message["ProtMsg"])
                        except Exception:
                            print(
                                " Sensor id:"
                                + hex(self.params["ID"])
                                + "Exception in user datadump:"
                            )
                            print("-" * 60)
                            traceback.print_exc(file=sys.stdout)
                            print("-" * 60)
                            pass
            except Exception as inst:
                 if self.timeoutOccured == False:
                     self.timeoutOccured = True
                     self.timeSinceLastPacket=0
                 else:
                     self.timeSinceLastPacket+=0.1

    def SetCallback(self, callback):
        """
        Sets an callback function signature musste be: callback(message["ProtMsg"], self.Description)

        Parameters
        ----------
        callback : function
            callback function signature musste be: callback(message["ProtMsg"], self.Description).

        Returns
        -------
        None.

        """
        self.flags["callbackSet"] = True
        self.callback = callback

    def UnSetCallback(self,):
        """
        deactivates the callback.

        Returns
        -------
        None.

        """
        self.flags["callbackSet"] = False
        self.callback = doNothingCb

    def stop(self):
        """
        Stops the sensor task.

        Returns
        -------
        None.

        """
        print("Stopping Sensor " + hex(self.params["ID"]))
        self._stop_event.set()
        # sleeping until run function is exiting due to timeout
        time.sleep(0.2)
        # thrash all data in queue
        while not self.buffer.empty():
            try:
                self.buffer.get(False)
            except:
                pass
        self.buffer.close()

    def join(self, *args, **kwargs):
        """
        Call the stop function

        Parameters
        ----------
        *args : args
            args are discarded.
        **kwargs : kwargs
            kwargs are discarded.

        Returns
        -------
        None.

        """
        self.stop()

    def __dumpMsgToFileASCII(self, message):
        """
        private function to dump MSG as ASCII line \n for new line.

        Parameters
        ----------
        message : protobuff message
            Data to be dumped.

        Returns
        -------
        None.

        """
        self.DumpfileASCII.write(
            str(message.id)
            + ";"
            + str(message.sample_number)
            + ";"
            + str(message.unix_time)
            + ";"
            + str(message.unix_time_nsecs)
            + ";"
            + str(message.time_uncertainty)
            + ";"
            + str(message.Data_01)
            + ";"
            + str(message.Data_02)
            + ";"
            + str(message.Data_03)
            + ";"
            + str(message.Data_04)
            + ";"
            + str(message.Data_05)
            + ";"
            + str(message.Data_06)
            + ";"
            + str(message.Data_07)
            + ";"
            + str(message.Data_08)
            + ";"
            + str(message.Data_09)
            + ";"
            + str(message.Data_10)
            + ";"
            + str(message.Data_11)
            + ";"
            + str(message.Data_12)
            + ";"
            + str(message.Data_13)
            + ";"
            + str(message.Data_14)
            + ";"
            + str(message.Data_15)
            + ";"
            + str(message.Data_16)
            + "\n"
        )

    def __dumpMsgToFileProto(self, message):
        """
        private function to dump MSG as binaryblob \n for new data packet.

        Parameters
        ----------
        message : protobuff message
            Data to be dumped.

        Returns
        -------
        None.

        """
        size = message.ByteSize()
        self.DumpfileProto.write(_VarintBytes(size))
        self.DumpfileProto.write(message.SerializeToString())


# USAGE
# create Buffer instance with ExampleBuffer=genericPlotter:(1000)
# Bind Sensor Callback to Buffer PushData function
# DR.AllSensors[$IDOFSENSOR].SetCallback(ExampleBuffer.PushData)
# wait until buffer is Full
# Data can be acessed over the atribute ExampleBuffer.Buffer[0]
class genericPlotter:
    def __init__(self, BufferLength):
        """
        Creates an Datebuffer witch is plotting the Sensor data after the buffer is full, one Subplot for every unique physical unit [°C,deg/s,m/s^2,µT]. in the data stream  

        Parameters
        ----------
        BufferLength : integer
            Length of the Buffer should fit aprox 2 seconds of dat.

        Returns
        -------
        None.

        """
        self.BufferLength = BufferLength
        self.Buffer = [None] * BufferLength
        self.Datasetpushed = 0
        self.FullmesaggePrinted = False
        self.flags = {"callbackSet" : False}
        # TODO change to actual time values""
        self.x = np.arange(BufferLength)
        self.Y = np.zeros([16, BufferLength])
        self.figInited = False

    def setUpFig(self):
        """
        Sets up the figure with subplots and labels cant be called in init since this params are not knowen to init time.

        Returns
        -------
        None.

        """
        self.units = (
            self.Description.getUnits()
        )  # returns dict with DSI-unit Strings as keys and channelist of channels as value
        self.Numofplots = len(
            self.units
        )  # numer off different units for one unit one plot
        plt.ion()
        # setting up subplot
        self.fig, self.ax = plt.subplots(self.Numofplots, 1, sharex=True)
        for ax in self.ax:
            ax.set_xlim(0, self.BufferLength)
        self.fig.suptitle(
            "Life plot of "
            + self.Description.SensorName
            + " with ID "
            + hex(self.Description.ID),
            y=1.0025,
        )
        self.titles = []
        self.unitstr = []
        # parsing titles and unit from the description
        for unit in self.units:
            self.unitstr.append(unit)
            title = ""
            for channel in self.units[unit]:
                title = title + self.Description[channel]["PHYSICAL_QUANTITY"] + " "
            self.titles.append(title)
            for i in range(len(self.titles)):
                self.ax[i].set_title(self.titles[i])
        plt.show()

    # TODO make convDict external
    def __getShortunitStr(self, unitstr):
        """
        converts the log DSI compatible unit sting to shorter ones for matplotlib plotting.
        e.g. '\\metre\\second\\tothe{-2}'--> "m/s^2".

        Parameters
        ----------
        unitstr : string
            DSi compatible string.

        Returns
        -------
        result : string
            Short string for matplotlib plotting.

        """
        convDict = {
            "\\degreecelsius": "deg C",
            "\\micro\\tesla": "uT",
            "\\radian\\second\\tothe{-1}": "rad/s",
            "\\metre\\second\\tothe{-2}": "m/s^2",
        }
        try:
            result = convDict[unitstr]
        except KeyError:
            result = unitstr
        return result

    def PushData(self, message, Description):
        """
        Pushes an block of data in to the buffer. This function is set as Sensor callback with the function :Sensor.SetCallback`

        Parameters
        ----------
        message : protobuff message
            Message to be pushed in the buffer.
        Description SensorDescription:
            SensorDescription is discarded.

        Returns
        -------
        None.

        """
        if self.Datasetpushed == 0:
            self.Description = copy.deepcopy(Description)
            # ok fig was not inited do it now
            if self.figInited == False:
                self.setUpFig()
                self.figInited = True
        if self.Datasetpushed < self.BufferLength:
            # Pushing data in to the numpy array for convinience
            i = self.Datasetpushed
            self.Buffer[i] = message
            self.Y[0, i] = self.Buffer[i].Data_01
            self.Y[1, i] = self.Buffer[i].Data_02
            self.Y[2, i] = self.Buffer[i].Data_03
            self.Y[3, i] = self.Buffer[i].Data_04
            self.Y[4, i] = self.Buffer[i].Data_05
            self.Y[5, i] = self.Buffer[i].Data_06
            self.Y[6, i] = self.Buffer[i].Data_07
            self.Y[7, i] = self.Buffer[i].Data_08
            self.Y[8, i] = self.Buffer[i].Data_09
            self.Y[9, i] = self.Buffer[i].Data_10
            self.Y[10, i] = self.Buffer[i].Data_11
            self.Y[11, i] = self.Buffer[i].Data_12
            self.Y[12, i] = self.Buffer[i].Data_13
            self.Y[13, i] = self.Buffer[i].Data_14
            self.Y[14, i] = self.Buffer[i].Data_15
            self.Y[15, i] = self.Buffer[i].Data_16
            self.Datasetpushed = self.Datasetpushed + 1
        else:
            # ok the buffer is full---> do some plotting now

            # flush the axis
            for ax in self.ax:
                ax.clear()
            # set titles and Y labels
            for i in range(len(self.titles)):
                self.ax[i].set_title(self.titles[i])
                self.ax[i].set_ylabel(self.__getShortunitStr(self.unitstr[i]))
            # actual draw
            i = 0
            for unit in self.units:
                for channel in self.units[unit]:
                    self.ax[i].plot(self.x, self.Y[channel - 1])
                i = i + 1
            # self.line1.set_ydata(self.y1)
            self.fig.canvas.draw()
            time=np.zeros(self.BufferLength)
            time_uncer = np.zeros(self.BufferLength)

            #_______ Peprare Data reshaping for agent comunication ________
            #                     generate time index
            for i in range(self.BufferLength):
                time[i]=self.Buffer[i].unix_time+self.Buffer[i].unix_time_nsecs*10e-9
                time_uncer[i]=self.Buffer[i].time_uncertainty*10e-9
            self.index=unumpy.uarray(time,time_uncer)
            activeChannels=self.Description.getActiveChannelsIDs()
            OutDataDescripton={}
            for ac in activeChannels:
                OutDataDescripton[ac-1]=self.Description[ac]
            coppyMask=np.array(list(activeChannels))
            timeDescription = {
                'PHYSICAL_QUANTITY' : "Time",
                'UNIT' : "unixSeconds",
                "UNCERTAINTY_TYPE": "2sigma convidence",
            }
            OutDescription={"Index":[timeDescription],"Data":OutDataDescripton,"TimeStamp":self.index[0]}
            coppyMask =coppyMask-1
            if self.flags["callbackSet"]:
                try:
                    self.callback(Index=self.index, Data = self.Y[coppyMask, :], Descripton = OutDescription)
                except Exception:
                    print(
                        " Generic Plotter for  id:"
                        + hex(self.Description.ID)
                        + " Exception in user callback:"
                    )
                    print("-" * 60)
                    traceback.print_exc(file=sys.stdout)
                    print("-" * 60)
                    pass
            # flush Buffer
            self.Buffer = [None] * self.BufferLength
            self.Datasetpushed = 0

    def SetCallback(self, callback):
        """
        Sets an callback function signature musste be: callback(message["ProtMsg"], self.Description)

        Parameters
        ----------
        callback : function
            callback function signature musste be: callback(message["ProtMsg"], self.Description).

        Returns
        -------
        None.

        """
        self.flags["callbackSet"] = True
        self.callback = callback

    def UnSetCallback(self,):
        """
        deactivates the callback.

        Returns
        -------
        None.

        """
        self.flags["callbackSet"] = False
        self.callback = doNothingCb

class RealFFTNodeCore:
    def __init__(self,Name):
        self.parmas={"Name":Name}

    def pushData(self,Index, Data, Descripton):
        self.Data=Data
        self.Index=Index
        self.Description=Descripton
        self.doRFFT()

    def doRFFT(self):
        self.outData=np.fft.rfft(self.Data,axis=0)
        #TODO add FTT scalfactor right to have power spectral density
        FFTScalfactor=1
        self.outData=self.outData*FFTScalfactor
        deltaT=np.mean(np.diff(self.Index))
        self.OutIndex=np.fft.rfftfreq(self.Data.shape[0],d=deltaT)
        #TODO generate description
        #think abou how to convert unit to fft units
        for DataChannels in self.Description["Data"]:
            candesc=self.Description["Data"][DataChannels]
            candesc["PHYSICAL_QUANTITY"]= candesc["PHYSICAL_QUANTITY"]+" power spectraldensity"
            candesc["UNIT"]="FFT UNIT" #INUIT^2/sqrt(HZ),
            candesc["UNCERTAINTY_TYPE"]= False
            candesc["RESOLUTION"]= candesc["RESOLUTION"]*self.Data.shape[0]
            candesc["MAX_SCALE"]: np.sqrt(2)*candesc["MAX_SCALE"]-candesc["MIN_SCALE"]# Peak to peak efective value is maximum for an fft bin
            candesc["MIN_SCALE"]: -1.0*candesc["MAX_SCALE"]
        freqDescription = {
        'PHYSICAL_QUANTITY': "Time frequency",
        'UNIT': "//Herz",
        "RESOLUTION":self.outData.shape[0],
        "MIN_SCALE" : self.Index[0],
        "MAX_scale" : self.Index[-1]
        }
        self.Description["Index"]=freqDescription
        print(self.parmas["Name"])
        print("___RFFT DONE !!! ____")
        print("Index " + str(self.OutIndex))
        print("Description " + str(self.Description))
        print("Data " + str(self.outData))






def ExampleDataPrinter(Index, Data, Descripton):
    #set breakpoint below this line to examine data structure
    print("___DATA PRINTER ____")
    print("Index "+str(Index))
    print("Description "+str(Descripton))
    print("Data "+str(Data))

# Example for DSCP Messages
# Quant b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x00"\x0eX Acceleration*\x0eY Acceleration2\x0eZ Acceleration:\x12X Angular velocityB\x12Y Angular velocityJ\x12Z Angular velocityR\x17X Magnetic flux densityZ\x17Y Magnetic flux densityb\x17Z Magnetic flux densityj\x0bTemperature'
# Unit  b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x01"\x17\\metre\\second\\tothe{-2}*\x17\\metre\\second\\tothe{-2}2\x17\\metre\\second\\tothe{-2}:\x18\\radian\\second\\tothe{-1}B\x18\\radian\\second\\tothe{-1}J\x18\\radian\\second\\tothe{-1}R\x0c\\micro\\teslaZ\x0c\\micro\\teslab\x0c\\micro\\teslaj\rdegreecelsius'
# Res   b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x03\xa5\x01\x00\x00\x80G\xad\x01\x00\x00\x80G\xb5\x01\x00\x00\x80G\xbd\x01\x00\x00\x80G\xc5\x01\x00\x00\x80G\xcd\x01\x00\x00\x80G\xd5\x01\x00\xf0\x7fG\xdd\x01\x00\xf0\x7fG\xe5\x01\x00\xf0\x7fG\xed\x01\x00\x00\x80G'
# Min   b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x04\xa5\x01\x16\xea\x1c\xc3\xad\x01\x16\xea\x1c\xc3\xb5\x01\x16\xea\x1c\xc3\xbd\x01\xe3\xa0\x0b\xc2\xc5\x01\xe3\xa0\x0b\xc2\xcd\x01\xe3\xa0\x0b\xc2\xd5\x01\x00\x00\x00\x80\xdd\x01\x00\x00\x00\x80\xe5\x01\x00\x00\x00\x80\xed\x01\xf3j\x9a\xc2'
# Max   b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x05\xa5\x01\xdc\xe8\x1cC\xad\x01\xdc\xe8\x1cC\xb5\x01\xdc\xe8\x1cC\xbd\x01\xcc\x9f\x0bB\xc5\x01\xcc\x9f\x0bB\xcd\x01\xcc\x9f\x0bB\xd5\x01\x00\x00\x00\x00\xdd\x01\x00\x00\x00\x00\xe5\x01\x00\x00\x00\x00\xed\x01\x02)\xeeB'
if __name__ == "__main__":
    DR = DataReceiver("192.168.0.200", 7654)
    time.sleep(5)
    firstSensorId = list(DR.AllSensors.keys())[0]
    secondSensorId = list(DR.AllSensors.keys())[0]
    print(
        "First sensor is"
        + str(DR.AllSensors[firstSensorId])
        + " binding generic plotter"
    )
    GP = genericPlotter(2000)
    DR.AllSensors[firstSensorId].SetCallback(GP.PushData)
    RFFTNode=RealFFTNodeCore("Simple Test Node")
    GP.SetCallback(RFFTNode.pushData)
# func_stats = yappi.get_func_stats()
# func_stats.save('./callgrind.out.', 'CALLGRIND')

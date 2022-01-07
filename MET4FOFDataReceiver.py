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
import datetime
import copy
import json
import h5py

# from mpi4py import MPI #for multi threaded hdf writing on Windows MSMPI needs to be installed https://www.microsoft.com/en-us/download/details.aspx?id=57467


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

        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)# socket can be resued instantly for debugging
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
        print("Data receiver now running wating for packets")

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
            + "\n Active Sensors are:"
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
                            timeDIFF = time.monotonic() - self.lastTimestamp
                            self.Datarate = (
                                self.params["PacketrateUpdateCount"] / timeDIFF
                            )
                            print("Update rate is " + str(self.Datarate) + " Hz")
                            self.lastTimestamp = time.monotonic()
                        else:
                            self.lastTimestamp = time.monotonic()
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
                            timeDIFF = time.monotonic() - self.lastTimestamp
                            self.Datarate = (
                                self.params["PacketrateUpdateCount"] / timeDIFF
                            )
                            print("Update rate is " + str(self.Datarate) + " Hz")
                            self.lastTimestamp = time.monotonic()
                        else:
                            self.lastTimestamp = time.monotonic()
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

    def StartDumpingAllSensorsASCII(
        self, folder="data", filenamePrefix="", splittime=86400, force=False
    ):
        AllDscsCompleete = True
        for SensorID in self.AllSensors:
            if self.AllSensors[SensorID].Description._complete == False:
                print(
                    "Description incompelte for sensor "
                    + str(self.AllSensors[SensorID])
                )
                if force != True:
                    AllDscsCompleete = False
        if AllDscsCompleete == False:
            raise RuntimeError(
                "not all descriptions are complete dumping not started."
                " Wait until descriptions are complete or use function argument force=true to start anyway"
            )
        if folder != "":
            if not os.path.exists(folder):
                os.makedirs(folder)
        filenamePrefixwFolder = os.path.join(folder, filenamePrefix)
        for SensorID in self.AllSensors:
            self.AllSensors[SensorID].StartDumpingToFileASCII(
                filenamePrefix=filenamePrefixwFolder, splittime=splittime
            )

    def StopDumpingAllSensorsASCII(
        self,
    ):
        for SensorID in self.AllSensors:
            self.AllSensors[SensorID].StopDumpingToFileASCII()


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
            "PHYSICAL_QUANTITY": None,
            "UNIT": None,
            "RESOLUTION": None,
            "MIN_SCALE": None,
            "MAX_SCALE": None,
            "HIERARCHY": None,
        }
        self._complete = False

    def __getitem__(self, key):
        # if key='SpecialKey':
        # self.Description['SpecialKey']
        return self.Description[key]

    def __setitem__(self, key, item):
        # if key='SpecialKey':
        # self.Description['SpecialKey']
        self.Description[key] = item

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
            self.Description["PHYSICAL_QUANTITY"] != None
            and self.Description["UNIT"] != None
            and self.Description["RESOLUTION"] != None
            and self.Description["MIN_SCALE"] != None
            and self.Description["MAX_SCALE"] != None
            and self.Description["HIERARCHY"] != None
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
        self.has_time_ticks = False  # do the data contain a 64 bit raw timestamp
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
                    # ok maybe the channels are coded as string
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

        ReturnsDR.
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

    def gethieracyasdict(self):
        self.hiracydict = {}
        channelsperdatasetcount = {}
        # loop over all channels to extract gropus and count elemnts perfomance dosent matter since only a few channels (16 max) are expected per description
        for Channel in self.Channels:
            splittedhieracy = self.Channels[Channel]["HIERARCHY"].split("/")
            if len(splittedhieracy) != 2:
                print(self.Channels[Channel]["HIERARCHY"])
                raise ValueError(
                    "HIERACY "
                    + Channel["HIERARCHY"]
                    + " is invalide since it was no split in two parts"
                )
            try:
                splittedhieracy[1] = int(splittedhieracy[1])
            except ValueError:
                raise ValueError(
                    "HIERACY "
                    + Channel["HIERARCHY"]
                    + "is invalide since last part is not an integer"
                )
            if splittedhieracy[0] in channelsperdatasetcount:
                channelsperdatasetcount[splittedhieracy[0]] = (
                    channelsperdatasetcount[splittedhieracy[0]] + 1
                )
            else:
                channelsperdatasetcount[splittedhieracy[0]] = 1
        print(channelsperdatasetcount)
        for key in channelsperdatasetcount.keys():
            self.hiracydict[key] = {
                "copymask": np.zeros(channelsperdatasetcount[key]).astype(int)
            }
            self.hiracydict[key]["PHYSICAL_QUANTITY"] = [
                None
            ] * channelsperdatasetcount[key]
            self.hiracydict[key]["RESOLUTION"] = np.zeros(channelsperdatasetcount[key])
            self.hiracydict[key]["MIN_SCALE"] = np.zeros(channelsperdatasetcount[key])
            self.hiracydict[key]["MAX_SCALE"] = np.zeros(channelsperdatasetcount[key])

        # print(self.hiracydict)
        # loop a second time infecient but don't care error check no nessary since done before
        # no align chann
        for Channel in self.Channels:
            splittedhieracy = self.Channels[Channel]["HIERARCHY"].split("/")
            self.hiracydict[splittedhieracy[0]]["copymask"][int(splittedhieracy[1])] = (
                self.Channels[Channel]["CHID"] - 1
            )
            self.hiracydict[splittedhieracy[0]]["MIN_SCALE"][
                int(splittedhieracy[1])
            ] = self.Channels[Channel]["MIN_SCALE"]
            self.hiracydict[splittedhieracy[0]]["MAX_SCALE"][
                int(splittedhieracy[1])
            ] = self.Channels[Channel]["MAX_SCALE"]
            self.hiracydict[splittedhieracy[0]]["RESOLUTION"][
                int(splittedhieracy[1])
            ] = self.Channels[Channel]["RESOLUTION"]
            self.hiracydict[splittedhieracy[0]]["PHYSICAL_QUANTITY"][
                int(splittedhieracy[1])
            ] = self.Channels[Channel]["PHYSICAL_QUANTITY"]
            self.hiracydict[splittedhieracy[0]]["UNIT"] = self.Channels[Channel][
                "UNIT"
            ]  # tehy ned to have the same unit by definition so we will over write it mybe some times but will not change anny thing
        print(self.hiracydict)
        return self.hiracydict


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
        6: "HIERARCHY",
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
            "DumpToFileProto": False,
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
                "HIERARCHY": False,
            }
        )
        for i in range(7):
            self.DescriptionsProcessed.add_alias(self.DescriptionTypNames[i], i)
        self._stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self.run, name="Sensor_" + str(ID) + "_thread", args=()
        )
        # self.thread.daemon = True
        self.thread.start()
        self.ProcessedPacekts = 0
        self.datarate = 0
        self.timeoutOccured = False
        self.timeSinceLastPacket = 0
        self.ASCIIDumpStartTime = None
        self.ASCIIDumpFileCount = 0
        self.ASCIIDumpFilePrefix = ""
        self.ASCIIDumpSplittime = 86400
        self.ASCIIDumpNextSplittime = 0

    def __repr__(self):
        """
        prints the Id and sensor name.

        Returns
        -------
        None.

        """
        return hex(self.Description.ID) + " " + self.Description.SensorName

    def StartDumpingToFileASCII(self, filenamePrefix="", splittime=86400):
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
        self.ASCIIDumpFilePrefix = filenamePrefix
        self.ASCIIDumpStartTime = time.monotonic()
        self.ASCIIDumpStartTimeLocal = datetime.datetime.now()
        self.flags["DumpToFileASCII"] = True
        if splittime > 0:
            self.ASCIIDumpSplittime = splittime
        self.initNewASCIIFile()

    def initNewASCIIFile(self):
        try:
            self.DumpfileASCII.close()
        except:
            pass  # the file is not opend or existing and there fore cant be closed
        filename = os.path.join(
            self.ASCIIDumpFilePrefix,
            self.ASCIIDumpStartTimeLocal.strftime("%Y%m%d%H%M%S")
            + "_"
            + str(self.Description.SensorName).replace(" ", "_")
            + "_"
            + hex(self.Description.ID)
            + "_"
            + str(self.ASCIIDumpFileCount).zfill(5)
            + ".dump",
        )
        print("created new dumpfile " + filename)
        self.DumpfileASCII = open(filename, "a")
        json.dump(self.Description.asDict(), self.DumpfileASCII)
        self.DumpfileASCII.write("\n")
        self.DumpfileASCII.write(
            "id;sample_number;unix_time;unix_time_nsecs;time_uncertainty;Data_01;Data_02;Data_03;Data_04;Data_05;Data_06;Data_07;Data_08;Data_09;Data_10;Data_11;Data_12;Data_13;Data_14;Data_15;Data_16\n"
        )
        self.params["DumpFileNameASCII"] = filename
        self.ASCIIDumpFileCount = self.ASCIIDumpFileCount + 1
        self.ASCIIDumpNextSplittime = (
            self.ASCIIDumpStartTime + self.ASCIIDumpSplittime * self.ASCIIDumpFileCount
        )

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
        self.ASCIIDumpFileCount = 0

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
                            if(Description.has_time_ticks==True):
                                print("Raw tick detected for " +Description.Sensor_name
                                + " sensor with ID:"
                                + str(self.params["ID"]))
                                self.Description.has_time_ticks =True
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
                                6,
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
                        if time.monotonic() > self.ASCIIDumpNextSplittime:
                            # TODO remove bug in this line
                            self.initNewASCIIFile()
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
                    self.timeSinceLastPacket = 0
                else:
                    self.timeSinceLastPacket += 0.1

    def donothingcb(self, message, Description):
        pass

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

    def UnSetCallback(
        self,
    ):
        """
        deactivates the callback.

        Returns
        -------
        None.

        """
        self.flags["callbackSet"] = False
        self.callback = self.donothingcb

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
            + ";"
            + str(message.time_ticks)
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


class HDF5Dumper:
    def __init__(self, dscp, file, hdfffilelock, chunksize=2048,correcttimeglitches=True,ignoreMissmatchErrors=True):
        self.dscp=dscp
        self.hdflock = hdfffilelock
        self.pushlock = threading.Lock()
        self.dataframindexoffset = 4
        self.chunksize = chunksize
        self.buffer = np.zeros([16, self.chunksize])
        self.time_buffer = np.zeros([4,self.chunksize], dtype=np.uint64)
        self.ticks_buffer = np.zeros( self.chunksize,dtype=np.uint64)
        self.chunkswritten = 0
        self.msgbufferd = 0
        self.lastdatatime=0
        self.timeoffset = 0
        self.uncerpenaltyfortimeerrorns=10000
        self.hieracy = dscp.gethieracyasdict()
        self.startimewritten = False
        self.correcttimeglitches=correcttimeglitches
        if isinstance(file, str):
            self.f = h5py.File(file, "a")
        elif isinstance(file, h5py._hl.files.File):
            self.f = file
        else:
            raise TypeError(
                "file needs to be either str or h5py._hl.files.File not "
                + str(type(file))
            )
        self.Datasets = {}
        # chreate self.groups
        with self.hdflock:
            try:
                self.group = self.f[
                    "RAWDATA/" + hex(dscp.ID) + "_" + dscp.SensorName.replace(" ", "_")
                ]
                warnings.warn(
                    "GROUP RAWDATA/"
                    + hex(dscp.ID)
                    + "_"
                    + dscp.SensorName.replace(" ", "_")
                    + " existed allready !"
                )

                self.Datasets["Absolutetime"] = self.group["Absolutetime"]
                self.Datasets["Absolutetime_uncertainty"] = self.group[
                    "Absolutetime_uncertainty"
                ]
                if(self.dscp.has_time_ticks):
                    self.Datasets["Time_ticks"] = self.group["Time_ticks"]
                self.Datasets["Sample_number"] = self.group["Sample_number"]
                if (self.Datasets["Absolutetime"].shape[1] / self.chunksize) / int(
                    self.Datasets["Absolutetime"].shape[1] / self.chunksize
                ) != 1:
                    warnings.warn(
                        "CHECK Chunksize Actual datasize is not an multiple of the chunksize set",
                        RuntimeWarning,
                    )
                self.chunkswritten = int(
                    self.Datasets["Absolutetime"].shape[1] / self.chunksize
                )
                for groupname in self.hieracy:
                    #TODO add loop over dict with error mesaages for unmatched parirs this will save at least 50 lines code and will be way better readable
                    self.Datasets[groupname] = self.group[groupname]
                    if (
                        not self.Datasets[groupname].attrs["Unit"]
                        == self.hieracy[groupname]["UNIT"]
                    ):
                        if ignoreMissmatchErrors:
                            raise RuntimeWarning(
                                "Unit missmatch !"
                                + self.Datasets[groupname].attrs["Unit"]
                                + " "
                                + self.hieracy[groupname]["UNIT"]
                            )
                        else:
                            raise RuntimeError(
                                "Unit missmatch !"
                                + self.Datasets[groupname].attrs["Unit"]
                                + " "
                                + self.hieracy[groupname]["UNIT"]
                            )

                    if not (
                        self.Datasets[groupname].attrs["Physical_quantity"]
                        == self.hieracy[groupname]["PHYSICAL_QUANTITY"]
                    ).all():
                        if ignoreMissmatchErrors:
                            raise RuntimeWarning(
                                "Physical_quantity missmatch !"
                                + self.Datasets[groupname].attrs["Physical_quantity"]
                                + " "
                                + self.hieracy[groupname]["PHYSICAL_QUANTITY"]
                            )
                        else:
                            raise RuntimeError(
                                "Physical_quantity missmatch !"
                                + self.Datasets[groupname].attrs["Physical_quantity"]
                                + " "
                                + self.hieracy[groupname]["PHYSICAL_QUANTITY"]
                            )

                    if not (
                        np.nan_to_num(self.Datasets[groupname].attrs["Resolution"])
                        == np.nan_to_num(self.hieracy[groupname]["RESOLUTION"])
                    ).all():
                        if ignoreMissmatchErrors:
                            raise RuntimeWarning(
                                "Resolution  missmatch !"
                                + str(self.Datasets[groupname].attrs["Resolution"])
                                + " "
                                + str(self.hieracy[groupname]["RESOLUTION"])
                            )
                        else:
                            raise RuntimeError(
                                "Resolution  missmatch !"
                                + str(self.Datasets[groupname].attrs["Resolution"])
                                + " "
                                + str(self.hieracy[groupname]["RESOLUTION"])
                            )

                    if not (
                        np.nan_to_num(self.Datasets[groupname].attrs["Max_scale"])
                        == np.nan_to_num(self.hieracy[groupname]["MAX_SCALE"])
                    ).all():
                        if ignoreMissmatchErrors:
                            raise RuntimeWarning(
                                "Max scale missmatch !"
                                + str(self.Datasets[groupname].attrs["Max_scale"])
                                + " "
                                + str(self.hieracy[groupname]["MAX_SCALE"])
                            )
                        else:
                            raise RuntimeError(
                                "Max scale missmatch !"
                                + str(self.Datasets[groupname].attrs["Max_scale"])
                                + " "
                                + str(self.hieracy[groupname]["MAX_SCALE"])
                            )

                    if not (
                        np.nan_to_num(self.Datasets[groupname].attrs["Min_scale"])
                        == np.nan_to_num(self.hieracy[groupname]["MIN_SCALE"])
                    ).all():
                        if ignoreMissmatchErrors:
                            raise RuntimeWarning(
                                "Min scale missmatch !"
                                + str(self.Datasets[groupname].attrs["Min_scale"])
                                + " "
                                + str(self.hieracy[groupname]["MIN_SCALE"])
                            )
                        else:
                            raise RuntimeError(
                                "Min scale missmatch !"
                                + str(self.Datasets[groupname].attrs["Min_scale"])
                                + " "
                                + str(self.hieracy[groupname]["MIN_SCALE"])
                            )
            except KeyError:
                self.group = self.f.create_group(
                    "RAWDATA/" + hex(dscp.ID) + "_" + dscp.SensorName.replace(" ", "_")
                )
                self.group.attrs["Data_description_json"] = json.dumps(dscp.asDict())
                self.group.attrs["Sensor_name"] = dscp.SensorName
                self.group.attrs["Sensor_ID"] = dscp.ID
                self.group.attrs["Data_description_json"] = json.dumps(dscp.asDict())
                self.group.attrs["Data_point_number"]=0
                if (self.dscp.has_time_ticks):
                    self.Datasets["Time_Ticks"] = self.group.create_dataset(
                        "Time_Ticks",
                        ([1, chunksize]),
                        maxshape=(1, None),
                        dtype="uint64",
                        compression="gzip",
                        shuffle=True,
                    )
                    self.Datasets["Time_Ticks"]
                    self.Datasets["Time_Ticks"].attrs["Unit"] = "\\one"
                    self.Datasets["Time_Ticks"].attrs[
                        "Physical_quantity"
                    ] = "CPU Ticks since System Start"
                    self.Datasets["Time_Ticks"].attrs["Resolution"] = np.exp2(64)
                    self.Datasets["Time_Ticks"].attrs["Max_scale"] = np.exp2(64)
                    self.Datasets["Time_Ticks"].attrs["Min_scale"] = 0

                self.Datasets["Absolutetime"] = self.group.create_dataset(
                    "Absolutetime",
                    ([1, chunksize]),
                    maxshape=(1, None),
                    dtype="uint64",
                    compression="gzip",
                    shuffle=True,
                )
                self.Datasets["Absolutetime"].make_scale("Absoluitetime")
                self.Datasets["Absolutetime"].attrs["Unit"] = "\\nano\\seconds"
                self.Datasets["Absolutetime"].attrs[
                    "Physical_quantity"
                ] = "Uinix_time_in_nanoseconds"
                self.Datasets["Absolutetime"].attrs["Resolution"] = np.exp2(64)
                self.Datasets["Absolutetime"].attrs["Max_scale"] = np.exp2(64)
                self.Datasets["Absolutetime"].attrs["Min_scale"] = 0
                self.Datasets["Absolutetime_uncertainty"] = self.group.create_dataset(
                    "Absolutetime_uncertainty",
                    ([1, chunksize]),
                    maxshape=(1, None),
                    dtype="uint32",
                    compression="gzip",
                    shuffle=True,
                )
                self.Datasets["Absolutetime_uncertainty"].attrs[
                    "Unit"
                ] = "\\nano\\seconds"
                self.Datasets["Absolutetime_uncertainty"].attrs[
                    "Physical_quantity"
                ] = "Uinix_time_uncertainty_in_nanosconds"
                self.Datasets["Absolutetime_uncertainty"].attrs["Resolution"] = np.exp2(
                    32
                )
                self.Datasets["Absolutetime_uncertainty"].attrs["Max_scale"] = np.exp2(
                    32
                )
                self.Datasets["Absolutetime_uncertainty"].attrs["Min_scale"] = 0.0

                self.Datasets["Sample_number"] = self.group.create_dataset(
                    "Sample_number",
                    ([1, chunksize]),
                    maxshape=(1, None),
                    dtype="uint32",
                    compression="gzip",
                    shuffle=True,
                )
                self.Datasets["Sample_number"].attrs["Unit"] = "\\one"
                self.Datasets["Sample_number"].attrs[
                    "Physical_quantity"
                ] = "Sample_number"
                self.Datasets["Sample_number"].attrs["Resolution"] = np.exp2(32)
                self.Datasets["Sample_number"].attrs["Max_scale"] = np.exp2(32)
                self.Datasets["Sample_number"].attrs["Min_scale"] = 0
                for groupname in self.hieracy:
                    vectorlength = len(self.hieracy[groupname]["copymask"])
                    self.Datasets[groupname] = self.group.create_dataset(
                        groupname,
                        ([vectorlength, chunksize]),
                        maxshape=(3, None),
                        dtype="float32",
                        compression="gzip",
                        shuffle=True,
                    )  # compression="gzip",shuffle=True,
                    self.Datasets[groupname].dims[0].label = "Absoluitetime"
                    self.Datasets[groupname].dims[0].attach_scale(
                        self.Datasets["Absolutetime"]
                    )
                    self.Datasets[groupname].attrs["Unit"] = self.hieracy[groupname][
                        "UNIT"
                    ]
                    self.Datasets[groupname].attrs["Physical_quantity"] = self.hieracy[
                        groupname
                    ]["PHYSICAL_QUANTITY"]
                    self.Datasets[groupname].attrs["Resolution"] = self.hieracy[
                        groupname
                    ]["RESOLUTION"]
                    self.Datasets[groupname].attrs["Max_scale"] = self.hieracy[
                        groupname
                    ]["MAX_SCALE"]
                    self.Datasets[groupname].attrs["Min_scale"] = self.hieracy[
                        groupname
                    ]["MIN_SCALE"]
                self.f.flush()

    def __str__(self):
        return str(self.f)+" "+hex(self.dscp.ID) + "_" + self.dscp.SensorName.replace(" ", "_")+' '+str(self.msgbufferd +self.chunksize * self.chunkswritten)+' msg received'
    def pushmsg(self, message, Description):
        with self.pushlock:
            time=message.unix_time*1e9+message.unix_time_nsecs
            if self.correcttimeglitches:
                if(self.msgbufferd==0 and self.chunkswritten==0):
                    self.lastdatatime = message.unix_time * 1e9 + message.unix_time_nsecs  # store fist time as last timestamp to have an difference of 0ns for fisrt sample
                deltat=time-self.lastdatatime
                if deltat<-2.5e8:
                    deltains=np.rint((deltat)/1e9)
                    self.timeoffset=self.timeoffset-deltains
                    warnings.warn("Time difference is negative in Sensor "+self.dscp.SensorName+' ID '+hex(self.dscp.ID)+" at IDX "+str(self.msgbufferd+self.chunksize * self.chunkswritten)+"with timme difference "+str(time-self.lastdatatime)+" ns "+str(deltains)+" in seconds "+str(self.timeoffset)+' accumulated deltat in s',UserWarning)
                if deltat > 2.5e8:
                    if self.timeoffset<0:
                        deltains=np.rint((deltat)/1e9)
                        self.timeoffset = self.timeoffset-deltains
                        if self.timeoffset!=0:
                            warnings.warn("Time difference is large positive in Sensor "+self.dscp.SensorName+' ID '+hex(self.dscp.ID)+"at IDX "+str(self.msgbufferd+self.chunksize * self.chunkswritten)+"with timme difference "+str(time-self.lastdatatime)+" ns "+str(deltains)+" in seconds "+str(self.timeoffset)+' accumulated deltat in seconds. Accumulated deltat will be set to 0',UserWarning)
                        self.timeoffset=0
            self.lastdatatime=message.unix_time*1e9+message.unix_time_nsecs#store last timestamp for consysty check of time
            self.buffer[:, self.msgbufferd] = np.array(
                [
                    message.Data_01,
                    message.Data_02,
                    message.Data_03,
                    message.Data_04,
                    message.Data_05,
                    message.Data_06,
                    message.Data_07,
                    message.Data_08,
                    message.Data_09,
                    message.Data_10,
                    message.Data_11,
                    message.Data_12,
                    message.Data_13,
                    message.Data_14,
                    message.Data_15,
                    message.Data_16,
                ]
            )
            self.time_buffer[:, self.msgbufferd] = np.array([
                message.sample_number,
                message.unix_time + self.timeoffset,
                message.unix_time_nsecs,
                message.time_uncertainty + self.uncerpenaltyfortimeerrorns * self.timeoffset])
            self.ticks_buffer[self.msgbufferd] = message.time_ticks
            self.msgbufferd = self.msgbufferd + 1
            if self.msgbufferd == self.chunksize:
                #print(hex(self.dscp.ID)+"waiting for lock "+str(self.hdflock))
                with self.hdflock:
                    #print(hex(self.dscp.ID)+"Aquired lock " + str(self.hdflock))
                    startIDX = self.chunksize * self.chunkswritten
                    #print("Start index is " + str(startIDX))
                    self.Datasets["Absolutetime"].resize([1, startIDX + self.chunksize])
                    time = (
                        self.time_buffer[1, :] * 1000000000
                        + self.time_buffer[2, : startIDX + self.chunksize]
                    )
                    self.Datasets["Absolutetime"][:, startIDX:] = time
                    if (self.dscp.has_time_ticks):
                        self.Datasets["Time_Ticks"].resize([1, startIDX + self.chunksize])
                        self.Datasets["Time_Ticks"][:, startIDX:] = self.ticks_buffer

                    self.Datasets["Absolutetime_uncertainty"].resize(
                        [1, startIDX + self.chunksize]
                    )
                    Absolutetime_uncertainty = self.time_buffer[3, :].astype(np.uint32)
                    self.Datasets["Absolutetime_uncertainty"][
                        :, startIDX:
                    ] = Absolutetime_uncertainty
                    if not self.startimewritten:
                        self.group.attrs["Start_time"] = time[0]
                        self.group.attrs[
                            "Start_time_uncertainty"
                        ] = Absolutetime_uncertainty[0]
                        self.startimewritten = True
                    self.Datasets["Sample_number"].resize(
                        [1, startIDX + self.chunksize]
                    )
                    samplenumbers = self.time_buffer[0, :].astype(np.uint32)
                    self.Datasets["Sample_number"][:, startIDX:] = samplenumbers

                    for groupname in self.hieracy:
                        vectorlength = len(self.hieracy[groupname]["copymask"])
                        self.Datasets[groupname].resize(
                            [vectorlength, startIDX + self.chunksize]
                        )
                        data = self.buffer[
                            (
                                self.hieracy[groupname]["copymask"]
                                #+ self.dataframindexoffset
                            ),
                            :,
                        ].astype("float32")
                        self.Datasets[groupname][:, startIDX:] = data
                    # self.f.flush()
                    self.msgbufferd = 0
                    self.chunkswritten = self.chunkswritten + 1
                    self.group.attrs["Data_point_number"] = self.chunkswritten*self.chunksize
                    self.buffer.fill(np.NaN)
                    self.ticks_buffer.fill(np.NaN)
                    self.buffer[0:4,:]=np.zeros([4,self.chunksize])


    def wirteRemainingToHDF(self):
        warnings.warn('WARNING will generate zeros in the end of the data file if callback is active there will be an gap in the file ')
        with self.hdflock:
            startIDX = self.chunksize * self.chunkswritten
            # print("Start index is " + str(startIDX))
            self.Datasets["Absolutetime"].resize([1, startIDX + self.chunksize])
            time = (
                    self.time_buffer[1, :] * 1000000000
                    + self.time_buffer[2, : startIDX + self.chunksize]
            ).astype(np.uint64)
            self.Datasets["Absolutetime"][:, startIDX:] = time
            if (self.dscp.has_time_ticks):
                self.Datasets["Time_Ticks"].resize([1, startIDX + self.chunksize])
                self.Datasets["Time_Ticks"][:, startIDX:] = self.ticks_buffer

            self.Datasets["Absolutetime_uncertainty"].resize(
                [1, startIDX + self.chunksize]
            )
            Absolutetime_uncertainty = self.time_buffer[3, :].astype(np.uint32)
            self.Datasets["Absolutetime_uncertainty"][
            :, startIDX:
            ] = Absolutetime_uncertainty
            if not self.startimewritten:
                self.group.attrs["Start_time"] = time[0]
                self.group.attrs[
                    "Start_time_uncertainty"
                ] = Absolutetime_uncertainty[0]
                self.startimewritten = True
            self.Datasets["Sample_number"].resize(
                [1, startIDX + self.chunksize]
            )
            samplenumbers = self.buffer[0, :].astype(np.uint32)
            self.Datasets["Sample_number"][:, startIDX:] = samplenumbers

            for groupname in self.hieracy:
                vectorlength = len(self.hieracy[groupname]["copymask"])
                self.Datasets[groupname].resize(
                    [vectorlength, startIDX + self.chunksize]
                )
                data = self.buffer[
                       (
                               self.hieracy[groupname]["copymask"]
                               #+ self.dataframindexoffset
                       ),
                       :,
                       ].astype("float32")
                self.Datasets[groupname][:, startIDX:] = data
            self.f.flush()
            self.group.attrs["Data_point_number"] = self.group.attrs["Data_point_number"] + self.msgbufferd
            self.msgbufferd = 0
            self.chunkswritten = self.chunkswritten + 1

def startdumpingallsensorshdf(filename):
    hdfdumplock = threading.Lock()
    hdfdumpfile = h5py.File(filename, "a")
    hdfdumper = []
    for SensorID in DR.AllSensors:
        hdfdumper.append(
            HDF5Dumper(DR.AllSensors[SensorID].Description, hdfdumpfile, hdfdumplock,chunksize=1024,correcttimeglitches=False)
        )
        DR.AllSensors[SensorID].SetCallback(hdfdumper[-1].pushmsg)
    return hdfdumper, hdfdumpfile


def stopdumpingallsensorshdf(dumperlist, dumpfile):
    for SensorID in DR.AllSensors:
        DR.AllSensors[SensorID].UnSetCallback()
    for dumper in dumperlist:
        print("closing"+str(dumper))
        dumper.wirteRemainingToHDF()
        dumper.f.flush()
        del dumper
    dumpfile.close()


    #for dumper in dumperlist:
    #    print(str(dumper))



if __name__ == "__main__":
    DR = DataReceiver("192.168.0.200", 7654)
    #time.sleep(10)
    #dumperlist,file=startdumpingallsensorshdf("tetratest_2.hfd5")
    #time.sleep(15)
    #stopdumpingallsensorshdf(dumperlist,file)
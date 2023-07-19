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
import asyncio

import warnings
from datetime import datetime
from multiprocessing import Queue
from multiprocessing import shared_memory
import multiprocessing
import time
import datetime
import copy
import json
import h5py

# from mpi4py import MPI #for multi threaded hdf writing on Windows MSMPI needs to be installed https://www.microsoft.com/en-us/download/details.aspx?id=57467


# for live plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas

# proptobuff message encoding
from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURR_DIR)
sys.path.append(CURR_DIR)
import messages_pb2

import pandas as pd
from bokeh.server.server import Server
from bokeh.models import TabPanel, Tabs,FileInput, Dropdown,ColumnDataSource, Ellipse, DataTable,TableColumn,Div,Spacer,NumericInput,RadioGroup,Button,TextAreaInput,MultiChoice,CheckboxGroup,PreText,HTMLTemplateFormatter
from bokeh.layouts import column,row
from bokeh.plotting import curdoc,figure, show
from tornado.ioloop import IOLoop
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.server.server import Server
import logging
from hdf_sensor_data_file_format.src.HDF5DataFiles import *
# matplotlib.use('Qt5Agg')

from filelock import Timeout, FileLock
bufferDtype=np.dtype([('absTime','u8'),
                      ('abstimeUncer','u4'),
                      ('sampleNumber','u4'),
                      ('Data_01','f4'),
                      ('Data_02','f4'),
                      ('Data_03','f4'),
                      ('Data_04','f4'),
                      ('Data_05','f4'),
                      ('Data_06','f4'),
                      ('Data_07','f4'),
                      ('Data_08','f4'),
                      ('Data_09','f4'),
                      ('Data_10','f4'),
                      ('Data_11','f4'),
                      ('Data_12','f4'),
                      ('Data_13','f4'),
                      ('Data_14','f4'),
                      ('Data_15','f4'),
                      ('Data_16','f4')])
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
        self.mpManager=multiprocessing.Manager()
        self.mpSensorDict=self.mpManager.dict()
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
                        #TODO performace improvement don't clall  ProtoData.ParseFromString(msg_buf) just get the id out of the str since its fixed position and length
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
                            self.AllSensors[SensorID].msgQueue.put_nowait(message)
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
                        self.AllSensors[SensorID] = Sensor(SensorID,self.mpSensorDict)
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
                            self.AllSensors[SensorID].msgQueue.put_nowait(message)
                        except:
                            print("packet lost for sensor ID:" + hex(SensorID))
                    else:
                        self.AllSensors[SensorID] = Sensor(SensorID,self.sharedMemorySensorList)
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
        self.stop()
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
        return "Descripton of " + self.SensorName + hex(self.ID)

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

    def asDataFrame(self):
        pdSeries=[]
        for key in self.Channels:
                pdSeries.append(pd.Series(self.Channels[key].Description,name=self.Channels[key].Description["CHID"]))
        df=pandas.DataFrame(pdSeries)
        print(df)
        return df


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

    def __init__(self, ID,mpSensorDict, msgQueSize=25e5,bufferSize=32768):
        """
        Constructor for the Sensor class

        Parameters
        ----------
        ID : uint32
            ID of the Sensor.
        msgQueSize : integer, optional
            Size of the Data Queue. The default is 25e5.

        Returns
        -------
        None.

        """
        self.Description = SensorDescription(ID, "Name not Set")
        self.msgQueue = Queue(int(msgQueSize))
        self.msgQueSize = msgQueSize
        self.flags = {
            "PrintProcessedCounts": True,
            "callbackSet": False,
        }
        self.processData=True
        self.params = {"ID": ID, "BufferSize": msgQueSize, "DumpFileName": ""}
        self.bufferSize=bufferSize
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
        self.smBufferName=str(self.params['ID'])
        # self.thread.daemon = True
        try:
            self.sharedMemoryBuffer=shared_memory.SharedMemory(name=self.smBufferName, create=True, size=self.bufferSize*bufferDtype.itemsize)
        except:
            self.sharedMemoryBuffer = shared_memory.SharedMemory(name=self.smBufferName, create=False,
                                                                 size=self.bufferSize * bufferDtype.itemsize)
        self.sharedmemoryArray=np.ndarray((self.bufferSize,), dtype=bufferDtype, buffer=self.sharedMemoryBuffer.buf)
        self.mpSensorDict=mpSensorDict
        self.golbalDescriptionUpdated=False
        self.thread.start()
        self.ProcessedPacekts = 0
        self.ProcessedPacektsLastDataRateupdate = 0
        self.timeLastDataRateupdate=time.monotonic()
        self.dataRate = np.NaN
        self.timeoutOccured = False
        self.timeSinceLastPacket = 0

    def __repr__(self):
        """
        prints the Id and sensor name.

        Returns
        -------
        None.

        """
        return hex(self.Description.ID) + " " + self.Description.SensorName

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
                if self.processData:
                    message = self.msgQueue.get(timeout=0.1)
                    self.timeoutOccured = False
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
                            if self.Description._ChannelsComplte and self.golbalDescriptionUpdated==False:
                                self.mpSensorDict[self.params['ID']]={'descriptionDict': self.Description.asDict(),'smBuffer':{'name':self.smBufferName,'size':self.bufferSize}}
                                self.golbalDescriptionUpdated=True
                        except Exception:
                            print(
                                " Sensor id:"
                                + hex(self.params["ID"])
                                + "Exception in user Description parsing:"
                            )
                            print("-" * 60)
                            traceback.print_exc(file=sys.stdout)
                            print("-" * 60)
                    if message["Type"] == "Data":
                        msg=message['ProtMsg']
                        self.sharedmemoryArray[self.ProcessedPacekts%self.bufferSize]=np.array((np.uint64(np.uint64(msg.unix_time) * np.uint64(1000000000)) + np.uint64(msg.unix_time_nsecs),
                            msg.time_uncertainty,
                            msg.sample_number,
                            msg.Data_01,
                            msg.Data_02,
                            msg.Data_03,
                            msg.Data_04,
                            msg.Data_05,
                            msg.Data_06,
                            msg.Data_07,
                            msg.Data_08,
                            msg.Data_09,
                            msg.Data_10,
                            msg.Data_11,
                            msg.Data_12,
                            msg.Data_13,
                            msg.Data_14,
                            msg.Data_15,
                            msg.Data_16),dtype=bufferDtype)
                        self.ProcessedPacekts = self.ProcessedPacekts + 1
                        if self.flags["PrintProcessedCounts"]:
                            if self.ProcessedPacekts % 100 == 0:
                                self.updateDatarate()
                            if self.ProcessedPacekts % 10000 == 0:
                                print(
                                    + hex(self.params["ID"])
                                    + str(self.msgQueue.qsize())
                                    + " ->"
                                    + "{:.2f}".format((self.msgQueue.qsize() / self.msgQueSize) * 100)
                                    + "%"
                                    + " "
                                    + "{:.2f}".format(self.dataRate)
                                    + " Hz"
                                )
                        if self.flags["callbackSet"]:
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
            except Exception as inst:
                if self.timeoutOccured == False:
                    self.timeoutOccured = True
                    self.timeSinceLastPacket = 0
                else:
                    self.timeSinceLastPacket += 0.1

    def updateDatarate(self):
        newTime=time.monotonic()
        deltaT=newTime-self.timeLastDataRateupdate
        deltapacekts=self.ProcessedPacekts-self.ProcessedPacektsLastDataRateupdate
        self.dataRate=deltapacekts/deltaT
        self.ProcessedPacektsLastDataRateupdate=self.ProcessedPacekts
        self.timeLastDataRateupdate=newTime
        return self.dataRate
    def stopDataProcessing(self):
        self.processData=False
    def startDataProcessing(self):
        self.processData=True
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
        while not self.msgQueue.empty():
            try:
                self.msgQueue.get(False)
            except:
                pass
        self.msgQueue.close()

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

class HDF5Dumper:
    def __init__(self, dscp, file,timeSliceGPRName, hdfffilelock, bufferLen=2048, ignoreMissmatchErrors=True):
        self.dscp=dscp
        self.hdflock = hdfffilelock
        self.pushlock = threading.Lock()
        self.dataframindexoffset = 4
        self.bufferLen = bufferLen
        self.buffer = np.zeros([16, self.bufferLen])
        self.ticks_buffer = np.zeros(self.bufferLen, dtype=np.uint64)
        self.absTimeBuffer = np.zeros(self.bufferLen, dtype=np.uint64)
        self.absTimeUncerSampleNumberBuffer = np.zeros([2,self.bufferLen], dtype=np.uint32)
        self.chunkswritten = 0
        self.msgbufferd = 0
        self.lastdatatime=0
        self.hieracy = dscp.gethieracyasdict()
        self.startimewritten = False
        self.dataFile=file
        self.Datasets = {}
        # chreate self.groups
        with self.hdflock:
            try:
                self.rawDataGPR = self.dataFile.initRawDataGPR(groupAttrs={"creationSoftware":"Met4FoFDataReceiver"})
                self.timeSliceGPR=self.rawDataGPR.initTimeSliceGroup(timeSliceGPRName)
                name='{0:0{1}X}'.format(self.dscp.ID,8)+'_'+str(self.dscp.SensorName)
                self.sourceGPR=self.timeSliceGPR.initSourceGroup(name)
                names=[]
                rows=[]
                arrayMapping={}
                dsetAttrs=[]
                for groupname in self.hieracy.keys():
                    dSetDSCP=self.hieracy[groupname]
                    names.append(str(groupname))
                    rows.append(len(dSetDSCP['copymask']))
                    arrayMapping[str(groupname)]=np.array(dSetDSCP['copymask'])
                    ProtobufTOHDFKeyMapping={'pysicalQuantity':'PHYSICAL_QUANTITY','unit':'UNIT','resolution':'RESOLUTION',"minRange":'MIN_SCALE',"maxRange":'MAX_SCALE'} #TODO move to HDF5Datafile.py
                    attrs={}
                    for hdfKey,protoKey in ProtobufTOHDFKeyMapping.items():
                        attrs[hdfKey]=dSetDSCP[protoKey]
                    dsetAttrs.append(attrs)
                self.sourceGPR.createMultiplerawDataSets(names, rows, arrayMapping,dsetAttrs,createAbsTimeBase=True)
            except Exception as E:
                print(E)

    def __str__(self):
        return str(self.dataFile)+" "+hex(self.dscp.ID) + "_" + self.dscp.SensorName.replace(" ", "_")+' '+str(self.msgbufferd + self.bufferLen * self.chunkswritten)+ ' msg received'
    def pushmsg(self, message, Description):
        test=10
        with self.pushlock:
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
            self.absTimeBuffer[self.msgbufferd]=np.uint64(np.uint64(message.unix_time) * np.uint64(1000000000)) + np.uint64(message.unix_time_nsecs)
            self.absTimeUncerSampleNumberBuffer[:,self.msgbufferd]=np.array([np.uint32(message.time_uncertainty),np.uint32(message.sample_number)])
            self.msgbufferd+=1
            if self.msgbufferd == self.bufferLen-1:
                with self.hdflock:
                    self.sourceGPR.pushArray(self.buffer,absTimeArray=self.absTimeBuffer,absTimeUncerSampleNumberArray=self.absTimeUncerSampleNumberBuffer)
                    self.msgbufferd=0
    def wirteRemainingToHDF(self):
        with self.hdflock:
            self.sourceGPR.pushArray(self.buffer[:,:self.msgbufferd],absTimeArray=self.absTimeBuffer[:self.msgbufferd],absTimeUncerSampleNumberArray=self.absTimeUncerSampleNumberBuffer[:,:self.msgbufferd])
            self.msgbufferd = 0
            self.sourceGPR.finishArrayPushing()


def startdumpingallsensorshdf(dataFile,timeSliceGPRName):
    hdfdumplock = threading.Lock()
    hdfdumpers = []
    for SensorID in DR.AllSensors:
        hdfdumpers.append(
            HDF5Dumper(DR.AllSensors[SensorID].Description,dataFile,timeSliceGPRName, hdfdumplock,bufferLen=int(32000))
        )
        DR.AllSensors[SensorID].SetCallback(hdfdumpers[-1].pushmsg)
    return hdfdumpers, dataFile

def stopdumpingallsensorshdf(dumperlist, dumpfile):
    for SensorID in DR.AllSensors:
        DR.AllSensors[SensorID].UnSetCallback()
    for dumper in dumperlist:
        print("closing"+str(dumper))
        dumper.wirteRemainingToHDF()
        dumper.dataFile.flush()
        del dumper


    #for dumper in dumperlist:
    #    print(str(dumper))

class fileDumper:

    def __init__(self,DR,fileName=None,bufferLen=int(32000)):
        self.DR=DR
        self.dumpers=[]
        self.bufferLen=bufferLen
        self.groupName=None
        self.lock=threading.Lock()
        self.state="idle"
        if fileName!=None:
            self.openFile(fileName)
    def openFile(self,fileName):
        if self.state == "idle":
            self.fileLock= FileLock(fileName+".lock")
            self.fileLock.acquire()
            self.file=HDF5DataFile(fileName,'w')
            self.state="fileNoDump"#stats are "idle","fileNoDUmp","fileDump"

    def closeFile(self):
        if self.state=="fileDump":
            self.stopDumpingAllSensors()
            self.file.flush()
            self.file.close()
            self.fileLock.rlease()
        elif self.state=="fileNoDump":
            self.file.flush()
            self.file.close()
            self.fileLock.rlease()
        elif self.state=="idle":
            warnings.warn("Dumper has no File to close!",RuntimeWarning)
    def startDumpingAllSensors(self,timeSliceGPRName):
        if self.state=="fileNoDump":
            for SensorID in self.DR.AllSensors:
                self.dumpers.append(HDF5Dumper(DR.AllSensors[SensorID].Description, self.file, timeSliceGPRName, self.lock,bufferLen=self.bufferLen))
                self.DR.AllSensors[SensorID].SetCallback(self.dumpers[-1].pushmsg)
                self.DR.AllSensors[SensorID].startDataProcessing()
            self.state="fileDump"
        self.groupName = timeSliceGPRName
    def switchTimeSliceGPR(self,timeSliceGPRName):
        if self.state=="fileDump":
            self.stopDumpingAllSensors()
            self.state == "fileNoDump"
            self.startDumpingAllSensors(timeSliceGPRName)
        if self.state=="fileNoDump":
            self.startDumpingAllSensors(timeSliceGPRName)#we havent been dumping so far so just start dumping

    def stopDumpingAllSensors(self):
        if self.state=="fileDump":
            for SensorID in self.DR.AllSensors:
                self.DR.AllSensors[SensorID].stopDataProcessing()
            for SensorID in DR.AllSensors:
                DR.AllSensors[SensorID].UnSetCallback()
            for dumper in self.dumpers:
                print("closing" + str(dumper))
                dumper.wirteRemainingToHDF()
                del dumper
            del self.dumpers
            self.dumpers=[]
            self.file.flush()
            self.groupName = None
            self.state="fileNoDump"
        else:
            warnings.warn("Not dumping stopDumpingAllSensors is pointless",RuntimeWarning)

class page:
    class SensorBokehWidget:
        def __init__(self, page, DR, sensorID):
            self.page = page
            self.DR = DR
            self.sensorID = sensorID
            self.sensor = self.DR.AllSensors[self.sensorID]
            self.descriptionDF = self.sensor.Description.asDataFrame()
            source = ColumnDataSource(self.descriptionDF)
            vars = list(self.descriptionDF.columns)
            columns = [TableColumn(field=Ci, title=Ci) for Ci in vars]  # bokeh columns
            self.data_table = DataTable(source=source, columns=columns, width=1800,
                                        height=30 * self.descriptionDF.shape[0])
            self.descriptionDiv=Div(text="""<h2 style="color:#1f77b4";>"""+" {:08X}".format(self.sensorID)+" "+ str(
                DR.AllSensors[self.sensorID].Description.SensorName)+" datarate {:.4f}".format(DR.AllSensors[self.sensorID].dataRate) +""" Hz </h2> """, height=10)
            self.widget = column(self.descriptionDiv, Spacer(height=10),
                                 self.data_table)
        def update(self):
            self.descriptionDiv.text="""<h2 style="color:#1f77b4";>"""+" {:08X}".format(self.sensorID)+" "+ str(
                DR.AllSensors[self.sensorID].Description.SensorName)+" datarate {:.4f}".format(DR.AllSensors[self.sensorID].dataRate) +""" Hz </h2> """
    class DumperBokehWidget:
        def __init__(self, page, DR,dumper):
            self.page=page
            self.DR=DR
            self.dumper=dumper
            self.dumping = False
            self.fileNameInput=TextAreaInput(value="filename.hdf5", title="Enter Filename")
            self.lastFileName=""
            self.lastSourceGroupName=""
            self.sourceGroupInput=TextAreaInput(value="TEST0000", title="Enter Source Group Name")
            self.startStopButton=Button(button_type="primary",label="Start Dumping")
            self.startStopButton.on_click(self.toggleDumping)
            self.dumperStateText=PreText(text=self.dumper.state)
            self.widget=row(self.dumperStateText,self.fileNameInput,self.sourceGroupInput,self.startStopButton)

        def toggleDumping(self):
            if self.dumping == False:
                self.filename=self.fileNameInput.value
                if self.dumper.state=="idle":
                    self.dumper.openFile(self.filename)
                    self.lastFileName=self.filename
                self.sourceGroupName=self.sourceGroupInput.value
                if self.sourceGroupName!=self.lastSourceGroupName:
                    if self.dumper.state=="fileNoDump":
                        self.dumper.startDumpingAllSensors(self.sourceGroupName)
                    self.lastSourceGroupName=self.sourceGroupName
                self.startStopButton.button_type="danger"
                self.startStopButton.label = "Stop Dumping"
                self.dumping = True
            else:
                self.dumper.stopDumpingAllSensors()
                self.startStopButton.button_type="success"
                self.startStopButton.label = "Start Dumping"
                self.dumping = False
        def update(self):
            self.dumperStateText.text=self.dumper.state

    def __init__(self,DR,dumper):
        self.DR=DR
        self.dumper=dumper
        self.descriptionDiv=Div(text="""<h2 style="color:#1f77b4";>Met4FoF Datareceiver IP:"""+str(self.DR.params['IP'])+":"+str(self.DR.params['Port'])+" datarate {:.2f}".format(self.DR.Datarate)+" packets/s"+"""</h2> """,height=14)
        self.page=column(self.descriptionDiv)
        self.bokehSensorWidgets=[]
        print("Creating Sensor Widgets")
        for sensorID in sorted(self.DR.AllSensors):
            print("ID"+str(sensorID))
            self.bokehSensorWidgets.append(self.SensorBokehWidget(self,self.DR,sensorID))
            self.page.children.append(self.bokehSensorWidgets[-1].widget)
        self.DumperWidget=self.DumperBokehWidget(self,self.DR,self.dumper)
        self.page.children.append(self.DumperWidget.widget)

    def update(self):
        self.descriptionDiv.text="""<h2 style="color:#1f77b4";>Met4FoF Datareceiver IP:"""+str(self.DR.params['IP'])+":"+str(self.DR.params['Port'])+" datarate {:.2f}".format(self.DR.Datarate)+" packets/s"+"""</h2> """
        for sensorWidget in self.bokehSensorWidgets:
            sensorWidget.update()
        self.DumperWidget.update()


def make_doc(doc):
        LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
        file_handler = logging.FileHandler(filename='./bokeh.log', mode='w')
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger = logging.getLogger(__name__)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.info('This is Datareceiver Viewer ...')
        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        logger.info("Opening Bokeh application on http://" + str(hostname) + ":5003/")
        logger.info("Opening Bokeh application on http://" + str(IPAddr) + ":5003/")
        myPage = page(DR,dumper)
        doc.add_root(myPage.page)
        doc.title = "DR view"
        doc.add_periodic_callback(myPage.update, 1000)
        print("Done")

class viewServerPage:
    def __init__(self,mpSensorDict):
        self.mpSensorDict=mpSensorDict
        self.page = column(Div(text="""<h2 style="color:#1f77b4";>Test</h2>"""))
        print("Done")
    def update(self):
        print("Debug")
        sensorID=self.mpSensorDict.keys()[0]
        existing_shm = shared_memory.SharedMemory(name=str(sensorID))
        # Note that a.shape is (6,) and a.dtype is np.int64 in this example
        c = np.ndarray((32768,), dtype=bufferDtype, buffer=existing_shm.buf)
        #self.mpSensorDict.keys
        #self.sharedMemoryBuffer = shared_memory.SharedMemory(name=self.smBufferName, create=False,
        #                                                         size=self.bufferSize * bufferDtype.itemsize)
        #self.sharedmemoryArray=np.ndarray((self.bufferSize,), dtype=bufferDtype, buffer=self.sharedMemoryBuffer.buf)
        pass


def sharedMembokehDataViewThread(mpSensorDict):
    print(mpSensorDict)
    def make_viewServerdoc(doc):
        LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
        file_handler = logging.FileHandler(filename='./bokehViewServer.log', mode='w')
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger = logging.getLogger(__name__)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.info('This is dataViewer ...')
        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        logger.info("Opening Bokeh application on http://" + str(hostname) + ":5011/")
        logger.info("Opening Bokeh application on http://" + str(IPAddr) + ":5011/")
        vSPage = viewServerPage(mpSensorDict)
        doc.add_root(vSPage.page)
        doc.title = "DR viewServer"
        doc.add_periodic_callback(vSPage.update, 1000)
        print("Done")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bokehPort = 5146
    dataViewerIo_loop = IOLoop.current()
    dataViewerBokeh_app = Application(FunctionHandler(make_viewServerdoc))
    viewServer = Server(
        {"/": dataViewerBokeh_app},  # list of Bokeh applications
        io_loop=dataViewerIo_loop,  # Tornado IOLoop
        http_server_kwargs={'max_buffer_size': 900000000},
        websocket_max_message_size=500000000,
        port=bokehPort,
        allow_websocket_origin=['*']
    )
    # start timers and services and immediately return
    viewServer.start()
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    print("Opening Bokeh application on http://" + str(hostname) + ":" + str(bokehPort) + "/")
    print("Opening Bokeh application on http://" + str(IPAddr) + ":" + str(bokehPort) + "/")
    dataViewerIo_loop.add_callback(viewServer.show, "/")
    dataViewerIo_loop.start()

if __name__ == "__main__":
    try:

        DR = DataReceiver("192.168.0.200", 7654)
        bokehPort=5046
        time.sleep(10)
        dumper=fileDumper(DR)
        """
        dumper.openFile(str(time.time()).replace('.','_')+".hdf5")
        dumper.startDumpingAllSensors("Test0000")
        time.sleep(30)
        dumper.stopDumpingAllSensors()
        """
        DataViewServerThread=threading.Thread(target=sharedMembokehDataViewThread, args=(DR.mpSensorDict,))
        DataViewServerThread.start()
        time.sleep(10)
        dataReceiverIo_loop = IOLoop()
        dataReceiverBokeh_app = Application(FunctionHandler(make_doc))
        server = Server(
            {"/": dataReceiverBokeh_app},  # list of Bokeh applications
            io_loop=dataReceiverIo_loop,  # Tornado IOLoop
            http_server_kwargs={'max_buffer_size': 900000000},
            websocket_max_message_size=500000000,
            port=bokehPort,
            allow_websocket_origin=['*']
        )
        # start timers and services and immediately return
        server.start()
        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        print("Opening Bokeh application on http://" + str(hostname) + ":" + str(bokehPort) + "/")
        print("Opening Bokeh application on http://" + str(IPAddr) + ":" + str(bokehPort) + "/")
        dataReceiverIo_loop.add_callback(server.show, "/")
        dataReceiverIo_loop.start()
    finally:
        server.stop()
        dataReceiverBokeh_app.stop()
        DR.stop()
        del DR
        #del bokeh_app

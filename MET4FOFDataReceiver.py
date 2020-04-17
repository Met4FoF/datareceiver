#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:20:12 2019

Data receiver for Met4FoF Protobuff Data
@author: seeger01
"""

import sys
import traceback
import os
import socket
import threading
import messages_pb2
import google.protobuf as pb
from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32
from datetime import datetime
import threading
import time
from multiprocessing import Queue
import copy
import json

# for live plotting
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

#matplotlib.use('Qt5Agg')


class DataReceiver:
    def __init__(self, IP, Port):
        """
        

        Parameters
        ----------
        IP : TYPE
            DESCRIPTION.
        Port : TYPE
            DESCRIPTION.

        Raises
        ------
        
            DESCRIPTION.
        an
            DESCRIPTION.

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
        self.packestlosforsensor={}
        self.AllSensors = {}
        self.ActiveSensors = {}
        self.msgcount = 0
        self.lastTimestamp = 0
        self.Datarate = 0
        self._stop_event = threading.Event()
        #start thread for data processing
        self.thread = threading.Thread(target=self.run, name="Datareceiver_thread", args=())
        self.thread.start()
        print("Data receiver now running wating for Packates")
    def __repr__(self):
        """
        

        Returns
        -------
        None.

        """
        return('Datareceiver liestening at ip '+str(self.params["IP"])+' Port '+str(self.params["Port"])+'\n Active Snesors are:'+str(self.AllSensors))

    def stop(self):
        """
        

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
                            tmp=self.packestlosforsensor[SensorID]=self.packestlosforsensor[SensorID]+1
                            if tmp==1:
                                print("!!!! FATAL PERFORMANCE PROBLEMS !!!!")
                                print("FIRSTTIME packet lost for sensor ID:" + str(SensorID))
                                print("DROP MESSAGES ARE ONLY PRINTETD EVERY 1000 DROPS FROM NOW ON !!!!!!!! ")
                            if tmp%1000==0:
                                print("oh no lost an other  thousand packets :(")
                    else:
                        self.AllSensors[SensorID] = Sensor(SensorID)
                        print(
                            "FOUND NEW SENSOR WITH ID=hex"
                            + hex(SensorID)
                            + "==>dec:"
                            + str(SensorID)
                        )
                        self.packestlosforsensor[SensorID]=0#initing lost packet counter
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
        CHID : TYPE
            DESCRIPTION.

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
        """
        

        Parameters
        ----------
        key : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # if key='SpecialKey':
        # self.Description['SpecialKey']
        return self.Description[key]

    def __repr__(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

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
        

        Parameters
        ----------
        key : TYPE
            DESCRIPTION.
        value : TYPE
            DESCRIPTION.

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
    def __init__(self, ID, SensorName):
        """
        

        Parameters
        ----------
        ID : TYPE
            DESCRIPTION.
        SensorName : TYPE
            DESCRIPTION.

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

    def setChannelParam(self, CHID, key, value):
        """
        

        Parameters
        ----------
        CHID : TYPE
            DESCRIPTION.
        key : TYPE
            DESCRIPTION.
        value : TYPE
            DESCRIPTION.

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
        

        Parameters
        ----------
        key : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # if key='SpecialKey':
        # self.Description['SpecialKey']
        return self.Channels[key]

    def asDict(self):
        """
        

        Returns
        -------
        RetunDict : TYPE
            DESCRIPTION.

        """
        RetunDict = {"Name": self.SensorName}
        for key in self.Channels:
            print(self.Channels[key].Description)
            RetunDict.update(
                {self.Channels[key]["CHID"]: self.Channels[key].Description}
            )
        return RetunDict


class Sensor:
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
    # TODO implement multi therading and callbacks
    def __init__(self, ID, BufferSize=25e5):
        """
        

        Parameters
        ----------
        ID : TYPE
            DESCRIPTION.
        BufferSize : TYPE, optional
            DESCRIPTION. The default is 25e5.

        Returns
        -------
        None.

        """
        self.Description = SensorDescription(ID, "Name not Set")
        self.buffer = Queue(int(BufferSize))
        self.buffersize = BufferSize
        self.flags = {
            "DumpToFile": False,
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
        self.lastPacketTimestamp = datetime.now()
        self.deltaT = (
            self.lastPacketTimestamp - datetime.now()
        )  # will b 0 but has deltaTime type witch is intended
        self.datarate = 0
    def __repr__(self):
        """
        

        Returns
        -------
        None.

        """
        return(hex(self.Description.ID)+' '+self.Description.SensorName)
    def StartDumpingToFileASCII(self, filename=""):
        """
        

        Parameters
        ----------
        filename : TYPE, optional
            DESCRIPTION. The default is "".

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
        

        Returns
        -------
        None.

        """
        self.flags["DumpToFileASCII"] = False
        self.params["DumpFileNameASCII"] = ""
        self.DumpfileASCII.close()

    def StartDumpingToFileProto(self, filename=""):
        """
        

        Parameters
        ----------
        filename : TYPE, optional
            DESCRIPTION. The default is "".

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
        

        Returns
        -------
        None.

        """
        self.flags["DumpToFileProto"] = False
        self.params["DumpFileNameProto"] = ""
        self.DumpfileProto.close()

    def run(self):
        """
        

        Returns
        -------
        None.

        """
        while not self._stop_event.is_set():
            # problem when we are closing the queue this function is waiting for data and raises EOF error if we delet the q
            # work around adding time out so self.buffer.get is returning after a time an thestop_event falg can be checked
            try:
                message = self.buffer.get(timeout=0.1)
                # self.deltaT = (
                #    tmpTime - self.lastPacketTimestamp
                # )  # will b 0 but has deltaTime type witch is intended
                # self.datarate = 1 / (self.deltaT.seconds + 1e-6 * self.deltaT.microseconds)
                # self.lastPacketTimestamp = datetime.now()
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
                                "Found new "
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
                     if(message['Type']=='Data'):
                         try:
                             self.__dumpMsgToFileProto(message['ProtMsg'])
                         except Exception:
                             print (" Sensor id:"+hex(self.params["ID"])+"Exception in user datadump:")
                             print('-'*60)
                             traceback.print_exc(file=sys.stdout)
                             print('-'*60)
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
            except Exception:
                pass

    def SetCallback(self, callback):
        """
        

        Parameters
        ----------
        callback : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.flags["callbackSet"] = True
        self.callback = callback

    def UnSetCallback(self,):
        """
        

        Returns
        -------
        None.

        """
        self.flags["callbackSet"] = False
        self.callback = doNothingCb

    def stop(self):
        """
        

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
        

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.stop()

    def __dumpMsgToFileASCII(self, message):
        """
        

        Parameters
        ----------
        message : TYPE
            DESCRIPTION.

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
        

        Parameters
        ----------
        message : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        size = message.ByteSize()
        self.DumpfileProto.write(_VarintBytes(size))
        self.DumpfileProto.write(message.SerializeToString())

def doNothingCb():
    """
    

    Returns
    -------
    None.

    """
    pass


# USAGE
# create Buffer instance with ExampleBuffer=DataBuffer(1000)
# Bind Sensor Callback to Buffer PushData function
# DR.AllSensors[$IDOFSENSOR].SetCallback(ExampleBuffer.PushData)
# wait until buffer is Full
# Data can be acessed over the atribute ExampleBuffer.Buffer[0]
class DataBuffer:
    def __init__(self, BufferLength):
        """
        

        Parameters
        ----------
        BufferLength : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.BufferLength = BufferLength
        self.Buffer = [None] * BufferLength
        self.Datasetpushed = 0
        self.FullmesaggePrinted = False
        self.x = np.arange(BufferLength)
        self.y1 = np.zeros(BufferLength)
        self.y2 = np.zeros(BufferLength)
        self.y3 = np.zeros(BufferLength)
        self.y4 = np.zeros(BufferLength)
        self.y5 = np.zeros(BufferLength)
        self.y6 = np.zeros(BufferLength)
        self.y7 = np.zeros(BufferLength)
        self.y8 = np.zeros(BufferLength)
        self.y9 = np.zeros(BufferLength)
        self.y10 = np.zeros(BufferLength)
        self.y11 = np.zeros(BufferLength)
        self.y12 = np.zeros(BufferLength)
        self.y13 = np.zeros(BufferLength)
        plt.ion()
        self.fig, self.ax = plt.subplots(5, 1, sharex=True)
        self.ax[0].set_xlim(0, self.BufferLength)
        self.ax[1].set_xlim(0, self.BufferLength)
        self.ax[2].set_xlim(0, self.BufferLength)
        self.ax[3].set_xlim(0, self.BufferLength)
        self.ax[4].set_xlim(0, self.BufferLength)
        self.ax[0].set_ylabel("Acceleration in m/s²")
        self.ax[1].set_ylabel("Rotational speed in rad/s")
        self.ax[2].set_ylabel("Mag. flux dens in µT")
        self.ax[3].set_ylabel("Temp. in °C")
        self.ax[4].set_ylabel("Analog V in V")
        # self.line1, = self.ax[0].plot(self.x,np.zeros(BufferLength))
        # self.line1.set_xdata(self.x)
        # self.ax.set_ylim(-160,160)
        plt.show()

    def PushData(self, message, Description):
        """
        

        Parameters
        ----------
        message : TYPE
            DESCRIPTION.
        Description : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.Datasetpushed == 0:
            self.Description = copy.deepcopy(Description)
        if self.Datasetpushed < self.BufferLength:
            i = self.Datasetpushed
            self.Buffer[i] = message
            self.y1[i] = self.Buffer[i].Data_01
            self.y2[i] = self.Buffer[i].Data_02
            self.y3[i] = self.Buffer[i].Data_03
            self.y4[i] = self.Buffer[i].Data_04
            self.y5[i] = self.Buffer[i].Data_05
            self.y6[i] = self.Buffer[i].Data_06
            self.y7[i] = self.Buffer[i].Data_07
            self.y8[i] = self.Buffer[i].Data_08
            self.y9[i] = self.Buffer[i].Data_09
            self.y10[i] = self.Buffer[i].Data_10
            self.y11[i] = self.Buffer[i].Data_11
            self.y12[i] = self.Buffer[i].Data_12
            self.y13[i] = self.Buffer[i].Data_13
            self.Datasetpushed = self.Datasetpushed + 1
        else:
            self.ax[0].clear()
            self.ax[1].clear()
            self.ax[2].clear()
            self.ax[3].clear()
            self.ax[4].clear()
            self.ax[0].set_ylabel("Acceleration in m/s²")
            self.ax[1].set_ylabel("Rotational speed in rad/s")
            self.ax[2].set_ylabel("Mag. flux dens in µT")
            self.ax[3].set_ylabel("Temp. in °C")
            self.ax[4].set_ylabel("Analog V in V")
            # self.line1.set_ydata(self.y1)
            self.ax[0].plot(self.x, self.y1)
            self.ax[0].plot(self.x, self.y2)
            self.ax[0].plot(self.x, self.y3)
            self.ax[1].plot(self.x, self.y4)
            self.ax[1].plot(self.x, self.y5)
            self.ax[1].plot(self.x, self.y6)
            self.ax[2].plot(self.x, self.y7)
            self.ax[2].plot(self.x, self.y8)
            self.ax[2].plot(self.x, self.y9)
            self.ax[3].plot(self.x, self.y10)
            self.ax[4].plot(self.x, self.y11)
            self.ax[4].plot(self.x, self.y12)
            self.ax[4].plot(self.x, self.y13)
            self.fig.canvas.draw()
            # flush Buffer
            self.Buffer = [None] * self.BufferLength
            self.Datasetpushed = 0


# Example for DSCP Messages
# Quant b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x00"\x0eX Acceleration*\x0eY Acceleration2\x0eZ Acceleration:\x12X Angular velocityB\x12Y Angular velocityJ\x12Z Angular velocityR\x17X Magnetic flux densityZ\x17Y Magnetic flux densityb\x17Z Magnetic flux densityj\x0bTemperature'
# Unit  b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x01"\x17\\metre\\second\\tothe{-2}*\x17\\metre\\second\\tothe{-2}2\x17\\metre\\second\\tothe{-2}:\x18\\radian\\second\\tothe{-1}B\x18\\radian\\second\\tothe{-1}J\x18\\radian\\second\\tothe{-1}R\x0c\\micro\\teslaZ\x0c\\micro\\teslab\x0c\\micro\\teslaj\rdegreecelsius'
# Res   b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x03\xa5\x01\x00\x00\x80G\xad\x01\x00\x00\x80G\xb5\x01\x00\x00\x80G\xbd\x01\x00\x00\x80G\xc5\x01\x00\x00\x80G\xcd\x01\x00\x00\x80G\xd5\x01\x00\xf0\x7fG\xdd\x01\x00\xf0\x7fG\xe5\x01\x00\xf0\x7fG\xed\x01\x00\x00\x80G'
# Min   b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x04\xa5\x01\x16\xea\x1c\xc3\xad\x01\x16\xea\x1c\xc3\xb5\x01\x16\xea\x1c\xc3\xbd\x01\xe3\xa0\x0b\xc2\xc5\x01\xe3\xa0\x0b\xc2\xcd\x01\xe3\xa0\x0b\xc2\xd5\x01\x00\x00\x00\x80\xdd\x01\x00\x00\x00\x80\xe5\x01\x00\x00\x00\x80\xed\x01\xf3j\x9a\xc2'
# Max   b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x05\xa5\x01\xdc\xe8\x1cC\xad\x01\xdc\xe8\x1cC\xb5\x01\xdc\xe8\x1cC\xbd\x01\xcc\x9f\x0bB\xc5\x01\xcc\x9f\x0bB\xcd\x01\xcc\x9f\x0bB\xd5\x01\x00\x00\x00\x00\xdd\x01\x00\x00\x00\x00\xe5\x01\x00\x00\x00\x00\xed\x01\x02)\xeeB'
if __name__ == "__main__":
    DR=DataReceiver("",7654)
# func_stats = yappi.get_func_stats()
# func_stats.save('./callgrind.out.', 'CALLGRIND')

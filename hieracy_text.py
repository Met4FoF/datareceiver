import numpy as np
import pandas
import h5py
import json
import os

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
            PHYSICAL_QUANTITY",UNIT,RESOLUTION,MIN_SCALE, MAX_SCALE,HIERACY.
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


def convertdumpfile(dumpfile):
    with open(dumpfile) as f:
        jsonstring = f.readline()
        dscp = json.loads(jsonstring)
    print(dscp)
    df = pandas.read_csv(dumpfile, sep=";", header=1)
    hdf5filename = dumpfile.replace("csv", "hdf5")
    # hdffile = h5py.File(hdf5filename, 'a')
    # group=hdffile.create_group(hex(dscp[ID]))
    return dscp, df


if __name__ == "__main__":

    dscpstr, df = convertdumpfile(
        r"D:\datareceiver\data\20201012133232_MPU_9250_0x60ad0100_00000.dump"
    )
    dscp = SensorDescription(fromDict=dscpstr)
    hieracy = dscp.gethieracyasdict()
    os.remove("test2.hdf5")
    f = h5py.File(r"test2.hdf5", "a")
    rawvalues = df.values.transpose()
    dataframindexoffset = 5
    Datasets = {}
    chunksize = 1000
    group = f.create_group(hex(dscp.ID) + "_" + dscp.SensorName.replace(" ", "_"))
    group.attrs["Data_description_json"] = json.dumps(dscp.asDict())
    group.attrs["Sensor_name"] = dscp.SensorName
    group.attrs["Sensor_ID"] = dscp.ID
    group.attrs["Data_description_json"] = json.dumps(dscp.asDict())
    Datasets["Absolutetime"] = group.create_dataset(
        "Absolutetime",
        ([1, chunksize]),
        maxshape=(1, None),
        dtype="float64",
        compression="gzip",
        shuffle=True,
    )
    Datasets["Absolutetime"].resize([1, rawvalues.shape[1]])
    time = rawvalues[2] + rawvalues[3] * 1e-9
    Datasets["Absolutetime"][...] = time.astype(np.float64)
    Datasets["Absolutetime"].make_scale("Absoluitetime")
    Datasets["Sample_number"] = group.create_dataset(
        "Sample_number",
        ([1, chunksize]),
        maxshape=(1, None),
        dtype="float64",
        compression="gzip",
        shuffle=True,
    )
    Datasets["Sample_number"].resize([1, rawvalues.shape[1]])
    Datasets["Sample_number"][...] = rawvalues[1].astype(np.uint32)
    for Groupname in hieracy:
        vectorlength = len(hieracy[Groupname]["copymask"])
        Datasets[Groupname] = group.create_dataset(
            Groupname,
            ([vectorlength, chunksize]),
            maxshape=(3, None),
            dtype="float32",
            compression="gzip",
            shuffle=True,
        )  # compression="gzip",shuffle=True,
        Datasets[Groupname].resize([vectorlength, rawvalues.shape[1]])
        Datasets[Groupname][...] = rawvalues[
            hieracy[Groupname]["copymask"] + dataframindexoffset
        ].astype("float32")
        Datasets[Groupname].dims[0].label = "Absoluitetime"
        Datasets[Groupname].dims[0].attach_scale(Datasets["Absolutetime"])
        Datasets[Groupname].attrs["Unit"] = hieracy[Groupname]["UNIT"]
        Datasets[Groupname].attrs["Physical_quantity"] = hieracy[Groupname][
            "PHYSICAL_QUANTITY"
        ]
        Datasets[Groupname].attrs["Resolution"] = hieracy[Groupname]["RESOLUTION"]
        Datasets[Groupname].attrs["Max_scale"] = hieracy[Groupname]["MAX_SCALE"]
        Datasets[Groupname].attrs["Min_scale"] = hieracy[Groupname]["MIN_SCALE"]
    f.flush()
    f.close()

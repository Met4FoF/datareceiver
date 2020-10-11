import numpy as np
descriptiondict={'Name': 'MPU_9250',
 'ID': 1621950464,
 1: {'CHID': 1, 'PHYSICAL_QUANTITY': 'X Acceleration', 'UNIT': '\\metre\\second\\tothe{-2}', 'RESOLUTION': 65536.0, 'MIN_SCALE': -156.91439819335938, 'MAX_SCALE': 156.90960693359375, 'HIERARCHY': 'Acceleration/0'},
 2: {'CHID': 2, 'PHYSICAL_QUANTITY': 'Y Acceleration', 'UNIT': '\\metre\\second\\tothe{-2}', 'RESOLUTION': 65536.0, 'MIN_SCALE': -156.91439819335938, 'MAX_SCALE': 156.90960693359375, 'HIERARCHY': 'Acceleration/1'},
 3: {'CHID': 3, 'PHYSICAL_QUANTITY': 'Z Acceleration', 'UNIT': '\\metre\\second\\tothe{-2}', 'RESOLUTION': 65536.0, 'MIN_SCALE': -156.91439819335938, 'MAX_SCALE': 156.90960693359375, 'HIERARCHY': 'Acceleration/2'},
 4: {'CHID': 4, 'PHYSICAL_QUANTITY': 'X Angular velocity', 'UNIT': '\\radian\\second\\tothe{-1}', 'RESOLUTION': 65536.0, 'MIN_SCALE': -34.9071159362793, 'MAX_SCALE': 34.90605163574219, 'HIERARCHY': 'Angular_velocity/0'},
 5: {'CHID': 5, 'PHYSICAL_QUANTITY': 'Y Angular velocity', 'UNIT': '\\radian\\second\\tothe{-1}', 'RESOLUTION': 65536.0, 'MIN_SCALE': -34.9071159362793, 'MAX_SCALE': 34.90605163574219, 'HIERARCHY': 'Angular_velocity/1'},
 6: {'CHID': 6, 'PHYSICAL_QUANTITY': 'Z Angular velocity', 'UNIT': '\\radian\\second\\tothe{-1}', 'RESOLUTION': 65536.0, 'MIN_SCALE': -34.9071159362793, 'MAX_SCALE': 34.90605163574219, 'HIERARCHY': 'Angular_velocity/2'},
 7: {'CHID': 7, 'PHYSICAL_QUANTITY': 'X Magnetic flux density', 'UNIT': '\\micro\\tesla', 'RESOLUTION': 65520.0, 'MIN_SCALE': -5890.5625, 'MAX_SCALE': 5890.5625, 'HIERARCHY': 'Magnetic_flux_density/0'},
 8: {'CHID': 8, 'PHYSICAL_QUANTITY': 'Y Magnetic flux density', 'UNIT': '\\micro\\tesla', 'RESOLUTION': 65520.0, 'MIN_SCALE': -5890.5625, 'MAX_SCALE': 5890.5625, 'HIERARCHY': 'Magnetic_flux_density/1'},
 9: {'CHID': 9, 'PHYSICAL_QUANTITY': 'Z Magnetic flux density', 'UNIT': '\\micro\\tesla', 'RESOLUTION': 65520.0, 'MIN_SCALE': -5890.5625, 'MAX_SCALE': 5890.5625, 'HIERARCHY': 'Magnetic_flux_density/2'},
 10: {'CHID': 10, 'PHYSICAL_QUANTITY': 'Temperature', 'UNIT': '\\degreecelsius', 'RESOLUTION': 65536.0, 'MIN_SCALE': -77.2088851928711, 'MAX_SCALE': 119.08009338378906, 'HIERARCHY': 'Temperature/0'}
 }

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
        self.hiracydict={}
        channelsperdatasetcount={}
        #loop over all channels to extract gropus and count elemnts perfomance dosent matter since only a few channels (16 max) are expected per description
        for Channel in self.Channels:
            splittedhieracy=self.Channels[Channel]["HIERARCHY"].split('/')
            if len(splittedhieracy) !=2:
                raise ValueError("HIERACY "+Channel["HIERARCHY"]+" is invalide since it was no split in two parts")
            try:
                splittedhieracy[1]=int(splittedhieracy[1])
            except ValueError:
                raise ValueError("HIERACY "+Channel["HIERARCHY"]+"is invalide since last part is not an integer")
            if splittedhieracy[0] in channelsperdatasetcount:
                channelsperdatasetcount[splittedhieracy[0]]=channelsperdatasetcount[splittedhieracy[0]]+1
            else:
                channelsperdatasetcount[splittedhieracy[0]] =1
        print(channelsperdatasetcount)
        for key in channelsperdatasetcount.keys():
            self.hiracydict[key] = {'copymask':np.zeros(channelsperdatasetcount[key]).astype(int)}
            self.hiracydict[key]['PHYSICAL_QUANTITY'] = [None] * channelsperdatasetcount[key]
            self.hiracydict[key]['RESOLUTION'] = np.zeros(channelsperdatasetcount[key])
            self.hiracydict[key]['MIN_SCALE'] = np.zeros(channelsperdatasetcount[key])
            self.hiracydict[key]['MAX_SCALE'] = np.zeros(channelsperdatasetcount[key])

        #print(self.hiracydict)
        # loop a second time infecient but don't care error check no nessary since done before
        # no align chann
        for Channel in self.Channels:
            splittedhieracy=self.Channels[Channel]["HIERARCHY"].split('/')
            self.hiracydict[splittedhieracy[0]]['copymask'][int(splittedhieracy[1])] = self.Channels[Channel]["CHID"] - 1
            self.hiracydict[splittedhieracy[0]]['MIN_SCALE'][int(splittedhieracy[1])]=self.Channels[Channel]["MIN_SCALE"]
            self.hiracydict[splittedhieracy[0]]['MAX_SCALE'][int(splittedhieracy[1])] = self.Channels[Channel]["MAX_SCALE"]
            self.hiracydict[splittedhieracy[0]]['RESOLUTION'][int(splittedhieracy[1])] = self.Channels[Channel]["RESOLUTION"]
            self.hiracydict[splittedhieracy[0]]['PHYSICAL_QUANTITY'][int(splittedhieracy[1])] = self.Channels[Channel]["PHYSICAL_QUANTITY"]
            self.hiracydict[splittedhieracy[0]]['UNIT'] = self.Channels[Channel]["UNIT"]# tehy ned to have the same unit by definition so we will over write it mybe some times but will not change anny thing
        print(self.hiracydict)




if __name__ == "__main__":
    DSCP=SensorDescription(fromDict=descriptiondict)
    DSCP.gethieracyasdict()
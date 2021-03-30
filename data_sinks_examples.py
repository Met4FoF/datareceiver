# USAGE
# create Buffer instance with ExampleBuffer=genericPlotter:(1000)
# Bind Sensor Callback to Buffer PushData function
# DR.AllSensors[$IDOFSENSOR].SetCallback(ExampleBuffer.PushData)
# wait until buffer is Full
# Data can be acessed over the atribute ExampleBuffer.Buffer[0]
class genericPlotter:
    def __init__(self, BufferLength, pushDevider=1):
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
        self.pushDevider = pushDevider
        self.Datasetpushed = 0
        self.Devidercount = 0
        self.FullmesaggePrinted = False
        self.flags = {"callbackSet": False}
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
        self.Plots = [None] * self.Numofplots
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
        if self.Devidercount % self.pushDevider == 0:
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
                time = np.zeros(self.BufferLength)
                time_uncer = np.zeros(self.BufferLength)

                # _______ Peprare Data reshaping for agent comunication ________
                #                     generate time index
                for i in range(self.BufferLength):
                    time[i] = (
                        self.Buffer[i].unix_time
                        + self.Buffer[i].unix_time_nsecs * 10e-9
                    )
                    time_uncer[i] = self.Buffer[i].time_uncertainty * 10e-9
                self.index = np.array(time)
                activeChannels = self.Description.getActiveChannelsIDs()
                OutDataDescripton = {}
                for ac in activeChannels:
                    OutDataDescripton[ac - 1] = self.Description[ac]
                coppyMask = np.array(list(activeChannels))
                timeDescription = {
                    "PHYSICAL_QUANTITY": "Time",
                    "UNIT": "unixSeconds",
                    "UNCERTAINTY_TYPE": "2sigma convidence",
                }
                OutDescription = {
                    "Index": [timeDescription],
                    "Data": OutDataDescripton,
                    "TimeStamp": self.index[0],
                }
                coppyMask = coppyMask - 1
                if self.flags["callbackSet"]:
                    try:
                        self.callback(
                            Index=self.index,
                            Data=self.Y[coppyMask, :],
                            Descripton=OutDescription,
                        )
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
        self.Devidercount += 1

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
        self.callback = doNothingCb


class RealFFTNodeCore:
    def __init__(self, Name):
        self.parmas = {"Name": Name}

    def pushData(self, Index, Data, Descripton):
        self.Data = Data
        self.Index = Index
        self.Description = Descripton
        self.doRFFT()

    def doRFFT(self):
        self.outData = np.fft.rfft(self.Data, axis=0)
        # TODO add FTT scalfactor right to have power spectral density
        FFTScalfactor = 1
        self.outData = self.outData * FFTScalfactor
        deltaT = np.mean(np.diff(self.Index))
        self.OutIndex = np.fft.rfftfreq(self.Data.shape[0], d=deltaT)
        # TODO generate description
        # think abou how to convert unit to fft units
        for DataChannels in self.Description["Data"]:
            candesc = self.Description["Data"][DataChannels]
            candesc["PHYSICAL_QUANTITY"] = (
                candesc["PHYSICAL_QUANTITY"] + " power spectraldensity"
            )
            candesc["UNIT"] = "FFT UNIT"  # INUIT^2/sqrt(HZ),
            candesc["UNCERTAINTY_TYPE"] = False
            candesc["RESOLUTION"] = candesc["RESOLUTION"] * self.Data.shape[0]
            candesc["MAX_SCALE"]: np.sqrt(2) * candesc["MAX_SCALE"] - candesc[
                "MIN_SCALE"
            ]  # Peak to peak efective value is maximum for an fft bin
            candesc["MIN_SCALE"]: -1.0 * candesc["MAX_SCALE"]
        freqDescription = {
            "PHYSICAL_QUANTITY": "Time frequency",
            "UNIT": "//Herz",
            "RESOLUTION": self.outData.shape[0],
            "MIN_SCALE": self.Index[0],
            "MAX_SCALE": self.Index[-1],
        }
        self.Description["Index"] = freqDescription
        print(self.parmas["Name"])
        print("___RFFT DONE !!! ____")
        print("Index " + str(self.OutIndex))
        print("Description " + str(self.Description))
        print("Data " + str(self.outData))


def ExampleDataPrinter(Index, Data, Descripton):
    # set breakpoint below this line to examine data structure
    print("___DATA PRINTER ____")
    print("Index " + str(Index))
    print("Description " + str(Descripton))
    print("Data " + str(Data))


# Example for DSCP Messages
# Quant b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x00"\x0eX Acceleration*\x0eY Acceleration2\x0eZ Acceleration:\x12X Angular velocityB\x12Y Angular velocityJ\x12Z Angular velocityR\x17X Magnetic flux densityZ\x17Y Magnetic flux densityb\x17Z Magnetic flux densityj\x0bTemperature'
# Unit  b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x01"\x17\\metre\\second\\tothe{-2}*\x17\\metre\\second\\tothe{-2}2\x17\\metre\\second\\tothe{-2}:\x18\\radian\\second\\tothe{-1}B\x18\\radian\\second\\tothe{-1}J\x18\\radian\\second\\tothe{-1}R\x0c\\micro\\teslaZ\x0c\\micro\\teslab\x0c\\micro\\teslaj\rdegreecelsius'
# Res   b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x03\xa5\x01\x00\x00\x80G\xad\x01\x00\x00\x80G\xb5\x01\x00\x00\x80G\xbd\x01\x00\x00\x80G\xc5\x01\x00\x00\x80G\xcd\x01\x00\x00\x80G\xd5\x01\x00\xf0\x7fG\xdd\x01\x00\xf0\x7fG\xe5\x01\x00\xf0\x7fG\xed\x01\x00\x00\x80G'
# Min   b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x04\xa5\x01\x16\xea\x1c\xc3\xad\x01\x16\xea\x1c\xc3\xb5\x01\x16\xea\x1c\xc3\xbd\x01\xe3\xa0\x0b\xc2\xc5\x01\xe3\xa0\x0b\xc2\xcd\x01\xe3\xa0\x0b\xc2\xd5\x01\x00\x00\x00\x80\xdd\x01\x00\x00\x00\x80\xe5\x01\x00\x00\x00\x80\xed\x01\xf3j\x9a\xc2'
# Max   b'\x08\x80\x80\xac\xe6\x0b\x12\x08MPU 9250\x18\x05\xa5\x01\xdc\xe8\x1cC\xad\x01\xdc\xe8\x1cC\xb5\x01\xdc\xe8\x1cC\xbd\x01\xcc\x9f\x0bB\xc5\x01\xcc\x9f\x0bB\xcd\x01\xcc\x9f\x0bB\xd5\x01\x00\x00\x00\x00\xdd\x01\x00\x00\x00\x00\xe5\x01\x00\x00\x00\x00\xed\x01\x02)\xeeB'

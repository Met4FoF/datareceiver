import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
import time
import multiprocessing
from MET4FOFDataReceiver import SensorDescription


def getplotableunitstring(unitstr, Latex=False):
    if not Latex:
        convDict = {
            "\\degreecelsius": "°C",
            "\\degree": "°",
            "\\micro\\tesla": "µT",
            "\\radian\\second\\tothe{-1}": "rad/s",
            "\\metre\\second\\tothe{-2}": "m/s^2",
            "\\volt": "v",
            "\\hertz": "Hz",
        }
    else:
        convDict = {
            "\\degreecelsius": "$^\circ C$",
            "\\degree": "$^\circ$",
            "\\micro\\tesla": "$\micro T$",
            "\\radian\\second\\tothe{-1}": "$\\frac{m}{s}$",
            "\\metre\\second\\tothe{-2}": "$\\frac{m}{s^2}",
            "\\volt": "v",
            "\\hertz": "Hz",
        }
    try:
        result = convDict[unitstr]
    except KeyError:
        result = unitstr
    return result

class hdfmet4fofdatafile:
    def __init__(self,hdffile):
        self.hdffile=hdffile
        self.senorsnames=list(self.hdffile['RAWDATA'].keys())
        self.sensordatasets={}
        for name in self.senorsnames:
            datasets=list(self.hdffile['RAWDATA/'+name].keys())
            keystocheckandremove=['Absolutetime', 'Absolutetime_uncertainty','Sample_number']
            for key in keystocheckandremove:
                try:
                    datasets.remove(key)
                except ValueError:
                    raise RuntimeWarning(str(name)+" doese not contain "+str(key)+" dataset is maybe corrupted!")
            self.sensordatasets[name]=datasets
        print("INIT DONE")
        print("RAW DataGroups are "+str(self.senorsnames))
        print("RAW Datasets are " + str(self.sensordatasets))

    def calcblockwiesestd(self,dataset,blocksize=100):
        start = time.time()
        blockcount=int(np.floor(dataset.size / blocksize))
        std=np.zeros(blockcount)
        split=np.split(dataset[:blocksize*blockcount], blockcount, axis=0)
        std=np.std(split,axis=1)
        end = time.time()
        print("bwstd for dataset "+str(dataset)+"took "+str(end - start)+ " secs")
        return std

    def detectnomovment(self,sensorname,quantitiy,treshold=0.05,blocksinrow=5,blocksize=100):
        tmpData = np.squeeze(self.hdffile['RAWDATA/'+sensorname + '/' + quantitiy])
        tmpTime = np.squeeze(self.hdffile['RAWDATA/'+sensorname + '/' + 'Absolutetime'])# fetch data from hdffile
        mag = np.linalg.norm(tmpData, axis=0)
        std = self.calcblockwiesestd(mag, blocksize=blocksize)
        wasvalide=0
        nomovementtidx=[]
        nomovementtimes = []
        for i in range(std.size):
            if std[i]<treshold:
                wasvalide=wasvalide+1
            else:
                if wasvalide>blocksinrow:
                    startidx=(i-wasvalide)*blocksize
                    stopidx = (i) * blocksize
                    nomovementtidx.append([startidx,stopidx])
                    nomovementtimes.append([tmpTime[startidx], tmpTime[stopidx]])
                wasvalide = 0
        nomovementidx=np.array(nomovementtidx)
        return nomovementidx,np.array(nomovementtimes)

    def detectmovment(self,sensorname,quantitiy,treshold=0.5,blocksinrow=5,blocksize=100):
        tmpData = np.squeeze(self.hdffile['RAWDATA/'+sensorname + '/' + quantitiy])
        tmpTime = np.squeeze(self.hdffile['RAWDATA/'+sensorname + '/' + 'Absolutetime'])# fetch data from hdffile
        mag = np.linalg.norm(tmpData, axis=0)
        std = self.calcblockwiesestd(mag, blocksize=blocksize)
        wasvalide=0
        movementtidx=[]
        movementtimes = []
        for i in range(std.size):
            if std[i]>treshold:
                wasvalide=wasvalide+1
            else:
                if wasvalide>blocksinrow:
                    startidx=(i-wasvalide)*blocksize
                    stopidx = (i) * blocksize
                    movementtidx.append([startidx,stopidx])
                    movementtimes.append([tmpTime[startidx], tmpTime[stopidx]])
                wasvalide = 0
        movementidx=np.array(movementtidx)
        return movementidx,np.array(movementtimes)

    def getnearestidxs(self,sensorname,time,blocksize=1000):
        absolutimegroupname='RAWDATA/'+sensorname + '/' + 'Absolutetime'
        absolutetimes= np.squeeze(self.hdffile[absolutimegroupname])
        subsampledtimes=absolutetimes[0::blocksize]
        originalshape=time.shape
        falttned=time.flatten()
        def getnearestsingelval(timeval):
            t=timeval
            timediff=abs(subsampledtimes-t)
            tmp=np.argmin(timediff)
            if(tmp==0):#value is in first block
                idx = np.argmin(abs(absolutetimes[:blocksize] - t))
            elif(tmp==len(subsampledtimes)):#value is in first block
                tmpidx = np.argmin(abs(absolutetimes[len((absolutetimes)-blocksize):] - t))
                idx= tmpidx+len((absolutetimes)-blocksize)
            else:
                subtimediff=abs(absolutetimes[(blocksize*(tmp-1)):(blocksize*(tmp+1))]-t)
                relidx=np.argmin(subtimediff)
                idx=relidx+(blocksize*(tmp-1))
            return int(idx)

        for i in np.arange(falttned.size):
            falttned[i]=getnearestsingelval(falttned[i])
        falttned.resize(originalshape)
        return falttned.astype(int)

class experiment():
    def __init__(self,hdfmet4fofdatafile,times):
        self.met4fofdatafile=hdfmet4fofdatafile
        self.timepoints=times
        self.idxs={}
        self.Data={}
        for name in self.met4fofdatafile.senorsnames:
            self.idxs[name]=self.met4fofdatafile.getnearestidxs(name,self.timepoints)
            self.Data[name] = {}
            for dataset in self.met4fofdatafile.sensordatasets[name]:
                self.Data[name][dataset] = {}
        print(self.idxs)

    def plotall(self,absolutetime=False):
        cols=len(self.met4fofdatafile.sensordatasets)#one colum for every sensor
        datasetspersensor=[]
        for sensors in self.met4fofdatafile.sensordatasets:
            datasetspersensor.append(len(self.met4fofdatafile.sensordatasets[sensors]))
        rows=np.max(datasetspersensor)
        fig,axs=plt.subplots(cols,rows, sharex='all')
        icol=0
        for sensor in self.met4fofdatafile.sensordatasets:
            irow=0
            idxs=self.idxs[sensor]
            axs[icol,0].annotate(sensor.replace('_',' '), xy=(0, 0.5), xytext=(-axs[icol,0].yaxis.labelpad - 5, 0),
                        xycoords=axs[icol,0].yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center',rotation=90)

            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                dsetattrs=self.met4fofdatafile.hdffile['RAWDATA/'+sensor+ '/' + dataset ].attrs
                time=self.met4fofdatafile.hdffile['RAWDATA/'+sensor + '/' + 'Absolutetime'][0,idxs[0]:idxs[1]]
                if not absolutetime:
                    time=time-self.timepoints[0]
                time=time/1e9
                print(dsetattrs.keys())
                axs[icol, irow].set_ylabel(getplotableunitstring(dsetattrs['Unit']))
                if not absolutetime:
                    axs[icol, irow].set_xlabel("Relative time in s")
                else :
                    axs[icol, irow].set_xlabel("Unixtime in s")
                axs[icol, irow].set_title(dataset.replace('_',' '))
                for i in np.arange(self.met4fofdatafile.hdffile['RAWDATA/'+sensor + '/' + dataset ].shape[0]):
                    label=dsetattrs['Physical_quantity'][i]
                    data=self.met4fofdatafile.hdffile['RAWDATA/'+sensor + '/' + dataset ][i,idxs[0]:idxs[1]]
                    axs[icol,irow].plot(time,data,label=label)
                    axs[icol,irow].legend()
                irow=irow+1
            icol=icol+1
        fig.show()

class sineexcitation(experiment):
    def __init__(self,hdfmet4fofdatafile,times):
        super().__init__(hdfmet4fofdatafile,times)

    def dofft(self):
        for sensor in self.met4fofdatafile.sensordatasets:
            idxs = self.idxs[sensor]
            points=idxs[1]-idxs[0]
            time = self.met4fofdatafile.hdffile['RAWDATA/' + sensor + '/' + 'Absolutetime'][0, idxs[0]:idxs[1]]
            reltime = time - self.timepoints[0]
            self.Data[sensor]['Mean Delta T']=np.mean(np.diff(reltime/1e9))
            self.Data[sensor]['RFFT Frequencys']=np.fft.rfftfreq(points,self.Data[sensor]['Mean Delta T'])
            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                data=self.met4fofdatafile.hdffile['RAWDATA/'+sensor + '/' + dataset ][:,idxs[0]:idxs[1]]
                self.Data[sensor][dataset]['RFFT'] =np.fft.rfft(data,axis=1)
                self.Data[sensor][dataset]['FFT_max_freq']=self.Data[sensor]['RFFT Frequencys'][np.argmax(abs(np.sum(self.Data[sensor][dataset]['RFFT'],axis=0)))]
                print(self.Data[sensor][dataset]['FFT_max_freq'])


    def dosinefits(self,sensorname,quantitiy,axis="Mag"):
        pass

def add1dsinereferencedatatohdffile(csvfilename,hdffile,axis=2):
    refcsv= pd.read_csv(csvfilename, delimiter=";",comment='#')
    hdffile=hdffile
    isaccelerationreference1d=False
    with open(csvfilename, "r") as file:
        first_line = file.readline()
        second_line = file.readline()
        third_line= file.readline()
        if r'loop;frequency;ex_amp;ex_amp_std;phase;phase_std' in first_line and r'#Number;Hz;m/s^2;m/s^2;deg;deg' in third_line:
            isaccelerationreference1d = True
            print("1D Accelerationrefference fiele given creating hdf5 data set")
        else:
            if not r'loop;frequency;ex_amp;ex_amp_std;phase;phase_std' in first_line:
                raise RuntimeError("Looking for >>>loop;frequency;ex_amp;ex_amp_std;phase;phase_std<<< in csvfile first row got"+first_line)
            if not r'#Number;Hz;m/s^2;m/s^2;deg;deg' in third_line:
                raise RuntimeError("Looking for >>>loop;frequency;ex_amp;ex_amp_std;phase;phase_std<<< in csvfile first row got"+third_line)
    if isaccelerationreference1d:
        Datasets = {}
        REFDATA = hdffile.create_group("REFENCEDATA")
        group = REFDATA.create_group("Acceleration_refference")
        group.attrs['Refference_name'] = "PTB HF acceleration standard"
        group.attrs['Sensor_name'] = group.attrs['Refference_name']
        group.attrs['Refference_type'] = "1D acceleration"
        Datasets['Frequency'] = group.create_dataset('Frequency', ([1, refcsv.shape[0]]),
                                                    dtype='float64')
        Datasets['Frequency'].make_scale("Frequency")
        Datasets['Frequency'].attrs['Unit'] = "/hertz"
        Datasets['Frequency'].attrs['Physical_quantity'] = "Excitation frequency"
        Datasets['Frequency'][:]=refcsv['frequency'].to_numpy()
        Datasets['Repetition count'] = group.create_dataset('repetition count', ([1, refcsv.shape[0]]),
                                                    dtype='int32')
        Datasets['Repetition count'].attrs['Unit'] = "/one"
        Datasets['Repetition count'].attrs['Physical_quantity'] = "Repetition count"
        Datasets['Repetition count'][:]=refcsv['loop'].to_numpy()
        Datasets['Repetition count'].dims[0].label = 'Frequency'
        Datasets['Repetition count'].dims[0].attach_scale(Datasets['Frequency'])
        uncerval = np.dtype([("value", np.float), ("uncertainty", np.float)])
        Datasets['Excitation amplitude']=group.create_dataset('Excitation amplitude', ([3, refcsv.shape[0]]),
                                                    dtype=uncerval)
        Datasets['Excitation amplitude'].attrs['Unit'] = "\\metre\\second\\tothe{-2}"
        Datasets['Excitation amplitude'].attrs['Physical_quantity'] = ["X Excitation amplitude",
                                                                       "Y Excitation amplitude",
                                                                       "Z Excitation amplitude"]
        Datasets['Excitation amplitude'].attrs['UNCERTAINTY_TYPE'] = "95% coverage gausian"
        Datasets['Excitation amplitude'][:]=np.empty([3, refcsv.shape[0]])
        Datasets['Excitation amplitude'][:]=np.nan
        Datasets['Excitation amplitude'][axis, :,"value"] = refcsv['ex_amp']
        Datasets['Excitation amplitude'][axis, :,"uncertainty"] = refcsv['ex_amp_std']
        Datasets['Excitation amplitude'].dims[0].label = 'Frequency'
        Datasets['Excitation amplitude'].dims[0].attach_scale(Datasets['Frequency'])

        Datasets['Phase']=group.create_dataset('Phasee', ([3, refcsv.shape[0]]),
                                                    dtype=uncerval)
        Datasets['Phase'].attrs['Unit'] = "\\degree"
        Datasets['Phase'].attrs['Physical_quantity'] = ["X Phase",
                                                        "Y Phase",
                                                        "Z Phase"]
        Datasets['Phase'].attrs['UNCERTAINTY_TYPE'] = "95% coverage gausian"
        Datasets['Phase'][:]=np.empty([3, refcsv.shape[0]])
        Datasets['Phase'][:]=np.nan
        Datasets['Phase'][axis, :,"value"] = refcsv['phase']
        Datasets['Phase'][axis, :,"uncertainty"] = refcsv['phase_std']
        Datasets['Phase'].dims[0].label = 'Frequency'
        Datasets['Phase'].dims[0].attach_scale(Datasets['Frequency'])
        hdffile.flush()


def processdata(i):
        mpdata['Experiemnts'][i] = sineexcitation(mpdata['hdfinstance'], mpdata['movementtimes'][i])
        mpdata['Experiemnts'][i].dofft()

if __name__ == "__main__":
    hdffilename = r"D:\data\2020-09-07 Messungen MPU9250_SN31_Zweikanalig\WDH3\20200907160043_MPU_9250_0x1fe40000_metallhalter_sensor_sensor_SN31_WDH3.hdf5"
    datafile = h5py.File(hdffilename, 'r+')
    test=hdfmet4fofdatafile(datafile)
    #nomovementidx,nomovementtimes=test.detectnomovment('0x1fe40000_MPU_9250', 'Acceleration')
    movementidx,movementtimes=test.detectmovment('0x1fe40000_MPU_9250', 'Acceleration')
    manager = multiprocessing.Manager()
    mpdata=manager.dict()
    mpdata['hdfinstance']=test
    mpdata['movementtimes']=movementtimes
    mpdata['Experiemnts']=[None]*movementtimes.shape[0]
    i=np.arange(movementtimes.shape[0])

    process1 = multiprocessing.Process(target=processdata, args=[i[0::2]])
    process2 = multiprocessing.Process(target=processdata, args=[i[1::2]])
    process1.start()
    process2.start()
    process1.join()
    process2.join()


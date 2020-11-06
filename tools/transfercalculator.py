import h5py
import numpy as np
import matplotlib.pyplot as plt
from MET4FOFDataReceiver import SensorDescription


def getplotableunitstring(unitstr, Latex=False):
    if not Latex:
        convDict = {
            "\\degreecelsius": "°C",
            "\\micro\\tesla": "µT",
            "\\radian\\second\\tothe{-1}": "rad/s",
            "\\metre\\second\\tothe{-2}": "m/s^2",
            "\\volt": "v",
        }
    else:
        convDict = {
            "\\degreecelsius": "$^\circ C$",
            "\\micro\\tesla": "$\micro T$",
            "\\radian\\second\\tothe{-1}": "$\frac{m}{s}$",
            "\\metre\\second\\tothe{-2}": "\frac{m}{s^2}",
            "\\volt": "v",
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

    def calcblockwiesestd(self,dataset,blocksize=1000):
        blockcount=int(np.floor(dataset.size / blocksize))
        std=np.zeros(blockcount)
        for i in range(blockcount):
            startIDX=i*blocksize
            stopIDX=(i+1)*blocksize
            std[i]=np.std(dataset[startIDX:stopIDX])
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

class Experiment():
    def __init__(self,hdfmet4fofdatafile,times):
        self.met4fofdatafile=hdfmet4fofdatafile
        self.timepoints=times
        self.idxs={}
        for name in self.met4fofdatafile.senorsnames:
            self.idxs[name]=self.met4fofdatafile.getnearestidxs(name,self.timepoints)
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


if __name__ == "__main__":
    hdffilename = r"/home/seeger01/Schreibtisch/20200907160043_MPU_9250_0x1fe40000_metallhalter_sensor_sensor_SN31_WDH3.hdf5"
    datafile = h5py.File(hdffilename, 'r+')
    test=hdfmet4fofdatafile(datafile)
    nomovementidx,nomovementtimes=test.detectnomovment('0x1fe40000_MPU_9250', 'Acceleration')
    movementidx,movementtimes=test.detectmovment('0x1fe40000_MPU_9250', 'Acceleration')
    experiment=Experiment(test,movementtimes[0])
    experiment2 = Experiment(test, movementtimes[1])
    experiment.plotall()
    experiment2.plotall()
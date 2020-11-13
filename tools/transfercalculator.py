#import h5py
import h5pickle as h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import multiprocessing
import sys
import time
import sinetools.SineTools as st
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


# code from https://stackoverflow.com/questions/23681948/get-index-of-closest-value-with-binary-search  answered Dec 6 '14 at 23:51 Yaguang
def binarySearch(data, val):
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break

        # check if data[mid] is closer to val than data[best_ind]
        dtype=data.dtype
        if data.dtype==np.uint64:
            absmidint=abs(data[mid].astype(np.int64) - val.astype(np.int64))
            absbestind = abs(data[best_ind].astype(np.int64) - val.astype(np.int64))
        else:
            absmidint=abs(data[mid] - val)
            absbestind = abs(data[best_ind]- val)
        if absmidint<absbestind:
            best_ind = mid

    return best_ind


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

    def getnearestidxs(self,sensorname,time):
        absolutimegroupname='RAWDATA/'+sensorname + '/' + 'Absolutetime'
        absolutetimes= np.squeeze(self.hdffile[absolutimegroupname])
        idxs=np.copy(time)
        with np.nditer(idxs, op_flags=['readwrite']) as it:
            for x in it:
                x[...]=binarySearch(absolutetimes ,x)
        return idxs



class experiment():
    def __init__(self,hdfmet4fofdatafile,times):
        self.met4fofdatafile=hdfmet4fofdatafile
        self.timepoints=times
        self.idxs={}
        self.Data={}
        for name in self.met4fofdatafile.senorsnames:
            start = time.time()
            self.idxs[name]=self.met4fofdatafile.getnearestidxs(name,self.timepoints)
            end = time.time()
            print("Nearest IDX Time " + str(end - start))
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
                    time=time.astype('int64') - self.timepoints[0].astype('int64')
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
            reltime = time.astype('int64') - self.timepoints[0].astype('int64')
            self.Data[sensor]['Mean Delta T']=np.mean(np.diff(reltime/1e9))
            self.Data[sensor]['RFFT Frequencys']=np.fft.rfftfreq(points,self.Data[sensor]['Mean Delta T'])
            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                data=self.met4fofdatafile.hdffile['RAWDATA/'+sensor + '/' + dataset ][:,idxs[0]:idxs[1]]
                self.Data[sensor][dataset]['RFFT'] =np.fft.rfft(data,axis=1)
                self.Data[sensor][dataset]['FFT_max_freq']=self.Data[sensor]['RFFT Frequencys'][np.argmax(abs(np.sum(self.Data[sensor][dataset]['RFFT'][:,1:],axis=0)))+1]
                print(self.Data[sensor][dataset]['FFT_max_freq'])


    def do3paramsinefits(self,freqs):
        for sensor in self.met4fofdatafile.sensordatasets:
            idxs = self.idxs[sensor]
            points = idxs[1] - idxs[0]
            time = self.met4fofdatafile.hdffile['RAWDATA/' + sensor + '/' + 'Absolutetime'][0, idxs[0]:idxs[1]]
            reltime = time.astype('int64') - self.timepoints[0].astype('int64')
            reltime=reltime/1e9
            excitationfreqs=freqs
            uniquexfreqs = np.sort(np.unique(excitationfreqs))
            idxs = self.idxs[sensor]
            for dataset in self.met4fofdatafile.sensordatasets[sensor]:
                try:
                    fftmaxfreq=self.Data[sensor][dataset]['FFT_max_freq']
                except NameError:
                    self.dofft()
                freqidx=binarySearch(uniquexfreqs, fftmaxfreq)
                f0=uniquexfreqs[freqidx]
                self.Data[sensor][dataset]['Sin_Fit_freq'] =f0
                datasetrows=self.met4fofdatafile.hdffile['RAWDATA/' + sensor + '/' +dataset].shape[0]
                #calc first row and create output array[:,idxs[0]:idxs[1]]
                sineparams=st.seq_threeparsinefit(self.met4fofdatafile.hdffile['RAWDATA/' + sensor + '/' +dataset][0, idxs[0]:idxs[1]], reltime , f0,periods=10)
                self.Data[sensor][dataset]['SinPOpt']=np.zeros([datasetrows,4])
                self.Data[sensor][dataset]['SinPCov'] = np.zeros([datasetrows,4, 4])
                self.Data[sensor][dataset]['SinParams'] = np.zeros([datasetrows, sineparams.shape[0], 3])
                self.Data[sensor][dataset]['SinParams'][0]=sineparams
                for i in np.arange(1,datasetrows):
                    sineparams = st.seq_threeparsinefit(
                        self.met4fofdatafile.hdffile['RAWDATA/' + sensor + '/' + dataset][i, idxs[0]:idxs[1]], reltime, f0,periods=10)
                    self.Data[sensor][dataset]['SinParams'][i] = sineparams
                for i in np.arange(datasetrows):
                    sineparams = self.Data[sensor][dataset]['SinParams'][i]
                    Complex = sineparams[:, 1] + 1j * sineparams[:, 0]
                    DC = sineparams[:, 2]
                    Freq = np.ones(sineparams.shape[0])*f0
                    self.Data[sensor][dataset]['SinPOpt'][i,:] = [
                        np.mean(abs(Complex)),
                        np.mean(DC),
                        np.mean(Freq),
                        np.mean(np.unwrap(np.angle(Complex))),
                    ]
                    CoVarData = np.stack(
                        (abs(Complex), DC, Freq, np.unwrap(np.angle(Complex))), axis=0
                    )
                    self.Data[sensor][dataset]['SinPCov'][i,:] = np.cov(
                        CoVarData, bias=True
                    )  # bias=True Nomation With N like np.std


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
        print(i)
        sys.stdout.flush()

        times=mpdata['movementtimes'][i]
        times[0] += 1000000000
        times[1] -= 1000000000
        experiment = sineexcitation(mpdata['hdfinstance'], times)
        sys.stdout.flush()
        print(experiment)
        sys.stdout.flush()
        start = time.time()
        experiment.dofft()
        end = time.time()
        print("FFT Time "+str(end - start))
        start = time.time()
        experiment.do3paramsinefits(mpdata['uniquexfreqs'])
        end = time.time()
        print("Sin Fit Time "+str(end - start))
        sys.stdout.flush()
        return experiment



if __name__ == "__main__":
    start = time.time()
    hdffilename = r"/home/seeger01/Schreibtisch/20200907160043_MPU_9250_0x1fe40000_metallhalter_sensor_sensor_SN31_WDH3.hdf5"
    revcsv = r"/home/seeger01/Schreibtisch/20200907160043_MPU_9250_0x1fe40000_metallhalter_sensor_sensor_SN31_WDH3_Ref_TF.csv"
    datafile = h5py.File(hdffilename, 'r+',driver='core')
    #add1dsinereferencedatatohdffile(revcsv, datafile)
    test=hdfmet4fofdatafile(datafile)
    #nomovementidx,nomovementtimes=test.detectnomovment('0x1fe40000_MPU_9250', 'Acceleration')
    movementidx,movementtimes=test.detectmovment('0x1fe40000_MPU_9250', 'Acceleration')
    manager = multiprocessing.Manager()
    mpdata=manager.dict()
    mpdata['hdfinstance']=test
    mpdata['movementtimes']=movementtimes
    mpdata['uniquexfreqs'] = np.unique(test.hdffile['REFENCEDATA/Acceleration_refference/Frequency'][0, :], axis=0)
    i=np.arange(movementtimes.shape[0])
    #i=np.arange(2)
    with multiprocessing.Pool(4) as p:
        results=p.map(processdata,i)
    end = time.time()
    print(end - start)



#PTB Pc
#Single Core 105.88655972480774
#Dual Core    59.0755774974823
#Triple Core  39.963184118270874
#Quad Core    31.353542804718018

#Bene Pc
#Single Core 103.87189602851868
#Dual Core    58.934141635894775
#Triple Core  42.016947507858276
#Quad Core    32.152546405792236
#Penta Core    26.92629075050354
#hexa core  24.332046270370483
#hepta core  21.610567331314087
#octa core   19.724497318267822

#dodeca cora 15.849191665649414
#15 core     13.566272497177124
#16 core     14.062445878982544
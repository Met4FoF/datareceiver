import os
import random
import numpy as np
import copy
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.ticker import AutoMinorLocator, MultipleLocator,LogLocator
from brokenaxes import brokenaxes
import sinetools.SineTools as st
import scipy.stats
from scipy.stats import vonmises
from multiprocessing import Pool
from contextlib import closing
import multiprocessing as mp
import tqdm
from tqdm.contrib.concurrent import process_map
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.signal import correlate
from scipy.signal import correlation_lags
from scipy.ndimage import gaussian_filter
import scipy as sp
import uncertainties
from tools.figPickel import saveImagePickle
#from mpi4py import MPI #multiprocessing read
import h5py as h5py
#import h5pickle as h5py
import functools
#import allantools
import sineTools2 as st2
import hashlib
#____________________ GLobal config begin_____________
# jitterGensForSimulations=manager.list()
jitterGensForSimulations = []
jitterSimuLengthInS=1.0
localFreqqCorr=True
askforFigPickelSave=False

lineSyles=[
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

tubscolors=[(0/255,112/255,155/255),(250/255,110/255,0/255), (109/255,131/255,0/255), (81/255,18/255,70/255),(102/255,180/255,211/255),(255/255,200/255,41/255),(172/255,193/255,58/255),(138/255,48/255,127/255)]
plt.rcParams['axes.prop_cycle'] = colorCycler=plt.cycler(color=tubscolors) #TUBS Blue,Orange,Green,Violet,Light Blue,Light Orange,Lieght green,Light Violet
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\boldmath'
LANG='DE'
if LANG=='DE':
    import locale
    trueFalseAnAus = {True: 'An', False: 'Aus'}
    locale.setlocale(locale.LC_NUMERIC,"de_DE.utf8")
    locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    plt.rcParams['text.latex.preamble'] = r'\usepackage{icomma}\usepackage{amsmath}\boldmath' # remove nasty Space behind comma in de_DE.utf8 locale https://stackoverflow.com/questions/50657326/matplotlib-locale-de-de-latex-space-btw-decimal-separator-and-number
    plt.rcParams['axes.formatter.use_locale'] = True
    # Customize formatting to not include thousand separators
    locale._override_localeconv.update({'grouping': [], 'thousands_sep': ''})
else:
    import locale
    trueFalseAnAus = {True: 'On', False: 'Off'}
    plt.rcParams['text.latex.preamble'] = r'\usepackage{icomma}\usepackage{amsmath}\boldmath' # remove nasty Space behind comma in de_DE.utf8 locale https://stackoverflow.com/questions/50657326/matplotlib-locale-de-de-latex-space-btw-decimal-separator-and-number
    locale.setlocale(locale.LC_NUMERIC,"en_US.utf8")
    locale.setlocale(locale.LC_ALL,"en_US.utf8")
    plt.rcParams['axes.formatter.use_locale'] = True
#plt.rcParams['mathtext.fontset'] = 'custom'
#plt.rcParams['mathtext.rm'] = 'NexusProSans'
#plt.rcParams['mathtext.it'] = 'NexusProSans:italic'
#plt.rcParams['mathtext.bf'] = 'NexusProSans:bold'
#plt.rcParams['mathtext.tt'] = 'NexusProSans:monospace'
plt.rcParams['svg.fonttype'] = 'none'  # This stores text as text in SVG files, not paths
plt.rc('text', usetex=True)
plt.rc("figure", figsize=[16,9])  # fontsize of the figure title
plt.rc("figure", dpi=300)
PLTSCALFACTOR = 1.66
SMALL_SIZE = 9 * PLTSCALFACTOR
MEDIUM_SIZE = 12 * PLTSCALFACTOR
BIGGER_SIZE = 15 * PLTSCALFACTOR
plt.rc("font", weight='bold') # controls default text sizes
plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
figSaveCounter = 30
SAVEFOLDER = './imagesPresentation'
SHOW=False


def gaus(x,a,sigma):
    x0=0
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def logGisticPICaped(x,k,x0,L):
    return L/(1+np.exp(-k*(x-x0)))

def align_yaxis(ax1, v1, ax2, v2):
    #adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1
    #https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin/10482477#10482477

    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)



class realWordJitterGen:
    def __rpr__(self):
        return str(self.title)+' fs= '+str(self.fs)+' Hz'
    def __init__(self,HDfFile,sensorName,title,nominalfreq=0,offset=[40000,10000],pollingFreq=None):
        try:
            self.floatType=np.float64
        except AttributeError as e:
            print ("Your system moast likely Windows does not support float128 with numpy switching back to float64")
            print(e)
            self.floatType=np.float64
        datafile=HDfFile
        self.title=title
        self.Dataset=datafile['RAWDATA/'+sensorName+'/Absolutetime'][0]
        self.dataPoints=datafile['RAWDATA/'+sensorName].attrs['Data_point_number']# use only valide points
        self.AbsoluteTime = self.Dataset[ 0 + offset[0]:self.dataPoints - offset[1]] - self.Dataset[ offset[0]]
        self.timeData=((self.AbsoluteTime-self.AbsoluteTime[0])/1e9).astype(self.floatType)
        #self.timeData=(self.Dataset[0,0+offset[0]:self.dataPoints-offset[1]]-self.Dataset[0,offset[0]])/1e9# substract first point to avoid precisionlos with f64 the divide by 1e9 to ahve seconds
        self.relSampleNumber = (datafile['RAWDATA/'+sensorName+'/Sample_number'][0, 0 + offset[0]:self.dataPoints-offset[1]] - datafile['RAWDATA/'+sensorName+'/Sample_number'][0, offset[0]])
        if nominalfreq==0:
            self.fs =  (self.relSampleNumber[-1].astype(self.floatType))/self.timeData[-1].astype(self.floatType)  # calculate smaple freq
        else:
            self.fs = nominalfreq
        self.fs_std=np.NaN
        #self.deltaT=self.length/(self.dataPoints-1)
        self.deltaT = 1.0/self.fs
        print("Sample frequency is "+str(self.fs)+' Hz')
        self.expectedTime=self.relSampleNumber.astype(self.floatType)*self.deltaT
        self.deviationFromNominal=self.timeData-self.expectedTime#calulate deviation from Expected Mean
        if pollingFreq!=None:
            self.PollDT=1.0/pollingFreq
            self.pollingTimes=np.arange(int(self.timeData[-1]/self.PollDT))*self.PollDT
            self.pollingTimeDiffFromNom=np.zeros_like(self.pollingTimes)
            lastDataIdx=0
            lastTime=0
            for i in range(self.pollingTimes.size-1):
                while self.timeData[lastDataIdx]-self.pollingTimes[i]<0:
                    lastDataIdx+=1
                self.pollingTimeDiffFromNom[i]=self.pollingTimes[i]-self.timeData[lastDataIdx-1]
            self.pollingTimeDiffFromNom[0]=0.0
            self.deviationFromNominal=self.pollingTimeDiffFromNom
            self.expectedTime=self.pollingTimes

            self.fs=pollingFreq

            """
            if pollingFreq<self.fs:
                raise RuntimeError("Polling Freq smaller than sensor Frequency this is not supportet yet!")
            else:
                self.pollingDT=1/pollingFreq
                self.deviationFromNominal=(self.timeData % self.pollingDT)
            """

        self.meandeviationFromNominal=np.mean(self.deviationFromNominal)
        self.std = np.std(self.deviationFromNominal)
        # INTERPOLATE MISSING DATA WITH NEAREST NIGBOUR
        self.interpolatedDeviationFromNominal=np.ones(self.relSampleNumber[-1])*self.meandeviationFromNominal

        diff=np.diff(self.relSampleNumber)
        jumpIDX=np.array(np.argwhere(diff>1))
        #jumpIDX=self.relSampleNumber[jumpIDX]
        jumpIDX = np.insert(jumpIDX, 0, 0)
        coppyOffset=0
        if jumpIDX.size>1:
            for i in range(jumpIDX.size-2):
                startIDX=jumpIDX[i]
                stopIDX=jumpIDX[i+1]
                tmp=np.copy(self.deviationFromNominal[startIDX:stopIDX])
                self.interpolatedDeviationFromNominal[(startIDX+coppyOffset):(stopIDX+coppyOffset)]=tmp
                coppyOffset+=diff[stopIDX]
                print(coppyOffset)
                del tmp
        else:
            self.interpolatedDeviationFromNominal=np.copy(self.deviationFromNominal)

        print("Test")

    def plotDeviation(self, fig=None, axs=None, lengthInS=None, show=False, lw=PLTSCALFACTOR, correctLinFreqDrift=True,
                      plotInSamples=False, save=False, unit='ns', yLims=None, plotInSamplesAxis=False, alpha=1,
                      color=None, maxSegments=1e4, animate=False):
        """
        Plot time deviation with optional animation support.

        Parameters:
        -----------
        animate : float or False
            If False (default), create static plot. If float, create animation with duration in seconds.
        """
        salefactorFromeUnit = {'ns': 1e9, r'\textmu s': 1e6, 'ms': 1e3, 's': 1}
        yScaleFactor = salefactorFromeUnit[unit]

        # For animation, always create new figure
        if animate:
            fig, ax = plt.subplots(figsize=(16, 9), dpi=120)  # FHD 16:9 aspect ratio
            axs = [ax]
            if plotInSamples or plotInSamplesAxis:
                ax2 = ax.twinx()
                axs = [ax, ax2]
        else:
            # Original static plot logic
            if fig is None and axs is None:
                fig, ax = plt.subplots()
                axs = [ax]
                if plotInSamples or plotInSamplesAxis:
                    ax2 = ax.twinx()
                    axs = [ax, ax2]
            else:
                if isinstance(axs, list):
                    axs = axs
                else:
                    if isinstance(axs, plt.Axes):
                        ax2 = axs.twinx()
                        axOld = axs
                        axs = [axOld, ax2]
                    else:
                        raise ValueError("No valid axis object")

        segment_length = int(lengthInS * self.fs) if lengthInS is not None else self.expectedTime.size
        total_possible_segments = int(np.ceil(self.expectedTime.size / segment_length))

        if color:
            all_colors = total_possible_segments * [color]
        else:
            cmap = plt.get_cmap('rainbow')
            all_colors = [cmap(i / total_possible_segments) for i in range(total_possible_segments)]

        # For animation, use all segments; for static, apply maxSegments logic
        if animate:
            segment_indices = range(total_possible_segments)
        else:
            if total_possible_segments > maxSegments:
                step_size = total_possible_segments // maxSegments
                segment_indices = range(0, step_size * maxSegments, step_size)
            else:
                segment_indices = range(total_possible_segments)

        # Prepare data for all segments
        segments_data = []
        slopes = np.zeros(len(segment_indices))

        for segment_idx, segment_num in enumerate(segment_indices):
            segment_start = segment_num * segment_length
            segment_end = min((segment_num + 1) * segment_length, self.expectedTime.size)
            timeDev = self.deviationFromNominal[segment_start:segment_end]
            times = self.expectedTime[segment_start:segment_end] - self.expectedTime[segment_start]

            segment_info = {
                'times': times,
                'timeDev': timeDev,
                'color': all_colors[segment_num],
                'segment_idx': segment_idx,
                'segment_num': segment_num
            }

            if correctLinFreqDrift:
                slope = (timeDev[-1] - timeDev[0]) / times[-1] if times[-1] != 0 else 0
                slopes[segment_idx] = slope
                correctedTimeDev = timeDev - slope * times - timeDev[0]
                segment_info['correctedTimeDev'] = correctedTimeDev
                segment_info['slope'] = slope

            segments_data.append(segment_info)

        # Set up axes labels and formatting
        if LANG == 'EN':
            axs[0].set_xlabel(r"\textbf{Relative time in s}")
            axs[0].set_ylabel(r"\textbf{Time Interval Error (TIE) in " + unit + "}")
        if LANG == 'DE':
            axs[0].set_xlabel(r"\textbf{Relative Zeit in s}")
            axs[0].set_ylabel(r"\textbf{Zeitabweichung (TIE) in " + unit + "}")

        for ax in axs:
            ax.ticklabel_format(axis='both', style='plain')

        if plotInSamples or plotInSamplesAxis:
            if LANG == 'EN':
                axs[1].set_ylabel(r"\textbf{TIE in sampling intervals} $\Delta t=\frac{1}{\overline{f_\text{s}}}$")
            if LANG == 'DE':
                axs[1].set_ylabel(r"\textbf{Zeitabweichung in} $\Delta t=\frac{1}{\overline{f_\text{s}}}$")

        axs[0].grid()

        if yLims is not None:
            axs[0].set_ylim(yLims)

        # Animation logic
        if animate:
            fps = 60
            total_frames = int(animate * fps)
            segments_per_frame = np.ceil((len(segments_data)-1) / total_frames)

            # Adjust total frames if needed due to ceiling
            actual_frames = int(np.ceil((len(segments_data)-1) / segments_per_frame))

            # Storage for line objects
            lines_main = []
            lines_secondary = []
            line_alphas = []
            line_widths = []

            # Alpha decay parameters
            alpha_decay = 0.1
            alpha_target = 0.3

            # Line width parameters
            lw_initial = lw * 3  # Start at 3x line width
            lw_target = lw  # End at 1x line width
            lw_decay_rate = (lw_initial - lw_target) / ((1.0 - alpha_target) / alpha_decay)  # Scale with alpha decay

            def init():
                axs[0].set_xlim(0, lengthInS if lengthInS else self.expectedTime[-1])
                if not yLims:
                    # Estimate y limits from all data
                    all_devs = []
                    for seg in segments_data:
                        if correctLinFreqDrift:
                            all_devs.extend(seg['correctedTimeDev'] * yScaleFactor)
                        else:
                            all_devs.extend(seg['timeDev'] * yScaleFactor)
                    if all_devs:
                        y_margin = 0.1 * (max(all_devs) - min(all_devs))
                        axs[0].set_ylim(min(all_devs) - y_margin, max(all_devs) + y_margin)

                if plotInSamplesAxis:
                    timeUnitsPerDT = self.deltaT * yScaleFactor
                    axs[1].set_ylim(axs[0].get_ylim()[0] / timeUnitsPerDT, axs[0].get_ylim()[1] / timeUnitsPerDT)

                return []

            def animate_frame(frame):
                # Calculate which segments to draw in this frame
                start_idx = int(frame * segments_per_frame)
                end_idx = int(min((frame + 1) * segments_per_frame, len(segments_data)))

                # Draw new segments
                for idx in range(start_idx, end_idx):
                    seg = segments_data[idx]

                    if correctLinFreqDrift:
                        line_main, = axs[0].plot(seg['times'], seg['correctedTimeDev'] * yScaleFactor,
                                                 label=self.title if idx == 0 else "", lw=lw_initial,
                                                 color=seg['color'], alpha=1.0)
                        lines_main.append(line_main)

                        if plotInSamples:
                            line_sec, = axs[1].plot(seg['times'], seg['correctedTimeDev'] / self.deltaT,
                                                    label=self.title if idx == 0 else "", lw=lw_initial,
                                                    color=seg['color'], ls=':', alpha=1.0)
                            lines_secondary.append(line_sec)
                    else:
                        line_main, = axs[0].plot(seg['times'], seg['timeDev'] * yScaleFactor,
                                                 label=self.title if idx == 0 else "", lw=lw_initial,
                                                 color=seg['color'], alpha=1.0)
                        lines_main.append(line_main)

                        if plotInSamples:
                            line_sec, = axs[1].plot(seg['times'], seg['timeDev'] / self.deltaT,
                                                    label=self.title if idx == 0 else "", lw=lw_initial,
                                                    color=seg['color'], ls=':', alpha=1.0)
                            lines_secondary.append(line_sec)

                    line_alphas.append(1.0)
                    line_widths.append(lw_initial)

                # Update alphas only for lines that need fading (from previous frames)
                if frame > 0:
                    # Calculate how many frames back we need to check for alpha updates
                    frames_to_fade = int(np.ceil((1.0 - alpha_target) / alpha_decay))

                    # Start checking from the frame that's 'frames_to_fade' frames back
                    check_start_frame = max(0, frame - frames_to_fade)

                    for check_frame in range(check_start_frame, frame):
                        # Calculate indices for lines drawn in this previous frame
                        check_start_idx = int(check_frame * segments_per_frame)
                        check_end_idx = int(min((check_frame + 1) * segments_per_frame, len(segments_data)))

                        # Update alpha for these lines
                        for idx in range(check_start_idx, check_end_idx):
                            if idx < len(line_alphas):
                                line_alphas[idx] = max(line_alphas[idx] - alpha_decay, alpha_target)
                                line_widths[idx] = max(line_widths[idx] - lw_decay_rate, lw_target)

                                lines_main[idx].set_alpha(line_alphas[idx])
                                lines_main[idx].set_linewidth(line_widths[idx])  # <-- HERE

                                if plotInSamples and idx < len(lines_secondary):
                                    lines_secondary[idx].set_alpha(line_alphas[idx])
                                    lines_secondary[idx].set_linewidth(line_widths[idx])  # <-- AND HERE

                # Update legend only once
                if frame == 0 and self.title:
                    axs[0].legend(loc='upper left', ncol=2)

                return lines_main + lines_secondary

            # Create animation
            anim = animation.FuncAnimation(fig, animate_frame, init_func=init,
                                           frames=actual_frames, interval=1000 / fps,
                                           blit=True, repeat=False)

            # Save animation
            if save:
                filename = os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' + str(
                    lengthInS) + 's_' + 'TimeDev_Corr_' + trueFalseAnAus[
                                            correctLinFreqDrift] + '_TimeDeviations_animated.webm')

                writer = animation.FFMpegWriter(fps=fps, codec='libvpx-vp9',
                                                bitrate=5000,  # 5 Mbps for good quality
                                                extra_args=['-quality', 'good', '-cpu-used', '0'])

                anim.save(filename, writer=writer)
                print(f"Animation saved to: {filename}")
                globals()['figSaveCounter'] += 1

            if show:
                plt.show()

            return fig, axs

        # Static plot logic (original code)
        else:
            for segment_idx, segment_num in enumerate(segment_indices):
                seg = segments_data[segment_idx]

                if correctLinFreqDrift:
                    axs[0].plot(seg['times'], seg['correctedTimeDev'] * yScaleFactor,
                                label=self.title if segment_idx == 0 else "", lw=lw,
                                color=seg['color'], alpha=alpha)
                    if plotInSamples:
                        axs[1].plot(seg['times'], seg['correctedTimeDev'] / self.deltaT,
                                    label=self.title if segment_idx == 0 else "", lw=lw,
                                    color=seg['color'], ls=':', alpha=alpha)
                else:
                    axs[0].plot(seg['times'], seg['timeDev'] * yScaleFactor,
                                label=self.title if segment_idx == 0 else "", lw=lw,
                                color=seg['color'], alpha=alpha)
                    if plotInSamples:
                        axs[1].plot(seg['times'], seg['timeDev'] / self.deltaT,
                                    label=self.title if segment_idx == 0 else "", lw=lw,
                                    color=seg['color'], ls=':', alpha=alpha)

            axs[0].legend(loc='upper left', ncol=2)

            if plotInSamplesAxis and not plotInSamples:
                timeUnitsPerDT = self.deltaT * yScaleFactor
                axs[1].set_ylim(axs[0].get_ylim()[0] / timeUnitsPerDT, axs[0].get_ylim()[1] / timeUnitsPerDT)

            if show:
                fig.show()

            if save:
                axs[0].grid()
                fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' + str(
                    lengthInS) + 's_' + 'TimeDev_Corr_ ' + trueFalseAnAus[correctLinFreqDrift] + '_TimeDevitions.png'),
                            dpi=300, bbox_inches='tight')
                fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' + str(
                    lengthInS) + 's_' + 'TimeDev_Corr_' + trueFalseAnAus[correctLinFreqDrift] + '_uncerComps.pdf'),
                            dpi=300, bbox_inches='tight')
                globals()['figSaveCounter'] += 1
                fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' + str(
                    lengthInS) + 's_' + 'TimeDev_Corr_' + trueFalseAnAus[correctLinFreqDrift] + '_uncerComps.svg'),
                            dpi=300, bbox_inches='tight')
                globals()['figSaveCounter'] += 1

            return fig, axs

    def getrandomDeviations(self,length,reytryes=1000):
        isContinousDataSliceRetryCount=0
        while isContinousDataSliceRetryCount<reytryes:
            idx=np.random.randint(self.relSampleNumber.size-(length+1))
            if self.relSampleNumber[idx+length]-self.relSampleNumber[idx]==length:
                break
            else:
                isContinousDataSliceRetryCount+=1
                RuntimeWarning(str(self.title)+" Hit hole in Data")

        return self.deviationFromNominal[idx:idx+length]

    def plotAkf(self,sampleFreq=1000,length=1048576*16):
        fig,ax=plt.subplots()
        tmp=np.zeros(length)
        tmp=self.interpolatedDeviationFromNominal[0:length]
        akf = correlate(tmp, tmp, mode='full')
        akf =akf/np.max(akf)

        gausnoise=np.random.normal(scale=self.std,size=tmp.size)
        gausnoise=gausnoise/np.max(gausnoise)
        akf_gausNoise = correlate (gausnoise,gausnoise, mode='full')
        akf_gausNoise=akf_gausNoise/np.max(akf_gausNoise)
        deltaT=1/sampleFreq
        lag=correlation_lags(tmp.size,tmp.size)*deltaT
        ax.plot(lag,akf,label=r'\textbf{\textbf{Aufgeizeichneter \textit{Jitter}}}')
        ax.plot(lag, akf_gausNoise, label=r'\textbf{Nomalverteilter Jitter} $\sigma = '+str(self.std)+'~\text{ns}$')
        ax.set_xlabel(r'\textbf{Zeitverschiebungen} $\tau$ \textbf{in } s')
        ax.set_ylabel(r'\textbf{Autokorrelations Funktion} $AKF$ \textbf{in } R.U s')
        ax.grid()
        ax.legend(ncol=2)
        fig.show()

    def plotFFT(self,sampleFreq=1000,plotPhase=True,fftlength=1048576*16):
        def abs2(x):
            return x.real ** 2 + x.imag ** 2
        nnumOFSlices=int(np.floor(self.interpolatedDeviationFromNominal.shape[0]/fftlength))
        if plotPhase:
            fig,axs=plt.subplots(2,sharex=True)
        else:
            fig, ax = plt.subplots()
            axs=np.array([ax])
        axs[0].set_yscale('log')
        freqs = np.fft.rfftfreq(fftlength, d=1/sampleFreq)
        scale = 2.0 / (fftlength*fftlength)
        quantNoise=np.random.uniform(size=fftlength)*1/(108e6)*1e9
        quantNoise=quantNoise-(1/54e6)*1e9
        gausnoise=np.random.uniform(size=fftlength)*(self.std)
        gausnoise=gausnoise-(self.std)/2
        fft_gausNoise = np.fft.rfft(gausnoise)
        fft_QuantNoise = np.fft.rfft(quantNoise)
        sliceFFTResultsAbsSqared=np.zeros([nnumOFSlices,freqs.size])
        for i in range(nnumOFSlices):
            print("FFT "+str(i/nnumOFSlices*100)+"% done")
            tmp=copy.deepcopy(self.interpolatedDeviationFromNominal[(fftlength*i):(fftlength+fftlength*i)])*1e9
            fftresult = np.fft.rfft(tmp)
            sliceFFTResultsAbsSqared[i]=abs2(fftresult)*scale
            if i==0:
                axs[0].plot(freqs,abs2(fftresult)* scale,alpha=1/nnumOFSlices,label=r'\textbf{Aufgeizeichneter \textit{Jitter}}',color='tab:blue')
                if plotPhase:
                    axs[1].plot(freqs,np.unwrap(np.angle(fftresult)) / np.pi,label=r'\textbf{\textbf{Aufgeizeichneter \textit{Jitter}}}',color='tab:blue')
            else:
                axs[0].plot(freqs,abs2(fftresult)*scale,alpha=1/nnumOFSlices,color='tab:blue')
                if plotPhase:
                    axs[1].plot(freqs,np.unwrap(np.angle(fftresult)) / np.pi,color='tab:blue')
        axs[0].plot(freqs, np.mean(sliceFFTResultsAbsSqared,axis=0),
                    label=r'\textbf{Mittelwert aufgeizeichneter \textit{Jitter}}', color='tab:blue')
        axs[0].plot(freqs, np.ones(freqs.size) * np.mean(abs2(fft_QuantNoise)*scale),label=r'\textbf{Gleich verteiltes Quantisierungs Rauschen Interval '+"%.2f" % ((1/108e6)*1e9)+' ns }',color='tab:orange')
        axs[0].plot(freqs, np.ones(freqs.size) * np.mean(abs2(fft_gausNoise)*scale),label=r'\textbf{Nomalverteilter Jitter} $\sigma = 43~\text{ns}$',color='tab:red')

        axs[0].set_ylabel(r'\textbf{Jitter~PSD in $\frac{{\text{\textbf{ns}}}^2}{\text{\textbf{Hz}}}$')
        axs[0].grid()
        axs[0].legend(ncol=2)
        if plotPhase:

            axs[1].set_ylabel(r'\textbf{\textit{unwrapped} Phase}\\ $\varphi$ \textbf{in} rad')
            axs[1].plot(freqs,np.unwrap(np.angle(fft_gausNoise)),label=r'\textbf{Gleich verteiltes Quantisierungs Rauschen Interval '+"%.2f" % ((1/108e6)*1e9)+' ns }',color='tab:orange')
            axs[1].plot( freqs,np.unwrap(np.angle(fft_QuantNoise)),
                    label=r'\textbf{Nomalverteiltes Rauschen} $\sigma = 43~\text{ns}$',color='tab:red')
            axs[1].legend(ncol=2)
            axs[1].set_xlabel(r'\textbf{Frequenz $f$ in Hz}')
            axs[1].grid()
        else:
            axs[0].set_xlabel(r'\textbf{Frequenz $f$ in Hz}')
        fig.tight_layout()
        fig.show()

    def plotPhaseNoise(self,sampleFreq=None,samplefreqCorr='local',fftlength=1048576*2,plotRaw=False,fig=None,axs=None,filterWidth=1,show=True,plotTimeDevs=False,lw=PLTSCALFACTOR,signalFreq=None,plotSincSensForLength=None,unit='dBc',save=False,xLims=None,yLims=None):
        fftlength=int(fftlength)
        if sampleFreq==None:
            sampleFreq=self.fs
        if signalFreq==None:
            signalFreq=1.0
        nnumOFSlices=int(np.floor(self.interpolatedDeviationFromNominal.shape[0]/fftlength))
        if fig==None and axs==None:
            fig, axs = plt.subplots()
        if plotTimeDevs:
            figTimeDev,axTimeDev=plt.subplots()

        freqs = np.fft.fftshift(np.fft.fftfreq(fftlength, d=1/sampleFreq))
        sliceFFTResultsAbs=np.zeros([nnumOFSlices,freqs.size])
        correctedFreqs=np.zeros(nnumOFSlices)
        for i in range(nnumOFSlices):
            print("FFT "+str(i/nnumOFSlices*100)+"% done")
            tmp=(copy.deepcopy(self.interpolatedDeviationFromNominal[(fftlength*i):(fftlength+fftlength*i)]))#+np.arange(fftlength)*1/sampleFreq).astype(self.floatType)
            if samplefreqCorr=='local':
                idx = np.arange(tmp.size)*self.deltaT
                slopeDeltaT, offset = np.polyfit(idx, tmp.astype(np.float64), 1)
                DeltaFreq=slopeDeltaT/self.deltaT
                correctedTimes=tmp-(np.arange(tmp.size)*(self.deltaT))*slopeDeltaT-offset
                correcedFreq=sampleFreq + DeltaFreq
                simuSin=np.cos(correctedTimes*(correcedFreq)*2*np.pi)+1j*np.sin(correctedTimes*(correcedFreq)*2*np.pi)
                fftresult = np.fft.fftshift(np.fft.fft(simuSin))
                sliceFFTResultsAbs[i]=abs(fftresult)
                if plotTimeDevs:
                    axTimeDev.plot(correctedTimes,label=r'\textbf{'+self.title+' Slice'+str(i)+'}',lw=lw)
                correctedFreqs[i]=correcedFreq
            else:
                simuSin=np.cos(tmp*sampleFreq*2*np.pi)+1j*np.sin(tmp*sampleFreq*2*np.pi)
                fftresult = np.fft.fftshift(np.fft.fft(simuSin))
                sliceFFTResultsAbs[i]=abs(fftresult)
                if plotTimeDevs:
                    axTimeDev.plot(tmp,lw=lw)
            print('SUM:' + str(np.sum(sliceFFTResultsAbs[i])/fftlength))
        if samplefreqCorr=='local':
            self.fs=np.mean(correctedFreqs)
            self.fs_std=np.std(correctedFreqs)
        sampleFrequFloat=uncertainties.ufloat(self.fs,self.fs_std*2)
        if unit=='dBc':
            psdMean=np.mean((sliceFFTResultsAbs ** 2) * (1 / (sampleFreq * fftlength)),axis=0)
            psdMean=psdMean*2
            p=axs.plot((freqs/(sampleFreq))*signalFreq, 10*np.log10(gaussian_filter(psdMean,filterWidth)),
                    label=r'\textbf{'+self.title,lw=lw)#+ $f_\text{s} = '+' {:.1u}'.format(sampleFrequFloat).replace('+/-',r'\pm')+'$ Hz }'
        else:
            psdMean=np.mean((sliceFFTResultsAbs ** 2) * (1 / (sampleFreq * fftlength)),axis=0)
            psdMean=psdMean*2
            p=axs.plot((freqs/(sampleFreq))*signalFreq, gaussian_filter(np.mean((sliceFFTResultsAbs)/fftlength,axis=0),filterWidth),
                    label=r'\textbf{'+self.title,lw=lw)#+' $f_\text{s} = '+' {:.1u}'.format(sampleFrequFloat).replace('+/-',r'\pm')+'$ Hz }'
        if plotSincSensForLength!=None:
            #create shadow axis to optain second legend
            axSincSens=axs.twinx()
            labelPrefixDict={'EN':'Sine approx. sensitivity ','DE':'Sinus-Approximation, Sensitivität, '}
            if isinstance(plotSincSensForLength, list):
                sincFreqs=np.linspace(-signalFreq*0.5,signalFreq*0.5,num=100000,endpoint=True)
                for length in plotSincSensForLength:
                    WindowAmps = abs(np.sinc(sincFreqs * length))
                    if unit == 'dBc':
                        WindowAmps=10 * np.log10(WindowAmps)
                    line=axSincSens.plot(sincFreqs,WindowAmps,ls='--',label=r'\textbf{'+labelPrefixDict[LANG]+'{:g}'.format(length)+' s}',color=axs._get_lines.get_next_color())
                    line[0].set_zorder(-1)
            if isinstance(plotSincSensForLength, dict):
                if 'poles' in plotSincSensForLength.keys():
                    numPoles=plotSincSensForLength['poles']
                    for length in plotSincSensForLength['length']:
                        maxfreq=numPoles*1/length
                        minFreq=-maxfreq
                        sincFreqs=np.linspace(minFreq,maxfreq,num=100000,endpoint=True)
                        WindowAmps = abs(np.sinc(sincFreqs * length))
                        if unit == 'dBc':
                            WindowAmps=10 * np.log10(WindowAmps)
                        line=axSincSens.plot(sincFreqs,WindowAmps,ls='--',label=r'\textbf{'+labelPrefixDict[LANG]+'{:g}'.format(length)+' s}',lw=lw*0.5,color=axs._get_lines.get_next_color())
                        line[0].set_zorder(-1)
                else:
                    sincFreqs=np.linspace(-plotSincSensForLength['maxFreq'],plotSincSensForLength['maxFreq'],num=100000,endpoint=True)
                    for length in plotSincSensForLength['length']:
                        WindowAmps = abs(np.sinc(sincFreqs * length))
                        if unit == 'dBc':
                            WindowAmps=10 * np.log10(WindowAmps)
                        line=axSincSens.plot(sincFreqs,WindowAmps,ls='--',label=r'\textbf{'+labelPrefixDict[LANG]+'{:g}'.format(length)+' s}',lw=lw*0.5,color=axs._get_lines.get_next_color())
                        line[0].set_zorder(-1)
            axSincSens.set_ylim(axs.get_ylim()) #scale axis like the original
            axSincSens.get_yaxis().set_visible(False)# deactivate gost axis visibility

        if plotRaw:
            for i in range(nnumOFSlices):
                if unit == 'dBc':
                    axs.plot((freqs/sampleFreq)*signalFreq,10*np.log10(sliceFFTResultsAbs[i]),alpha=1/nnumOFSlices,color=p[0].get_color(),lw=lw)#label=r'\textbf{'+self.title+'}'
                if unit == 'A.U':
                    axs.plot((freqs / sampleFreq) * signalFreq, sliceFFTResultsAbs[i], alpha=1 / nnumOFSlices, color=p[0].get_color(), lw=lw)  # label=r'\textbf{'+self.title+'}'

        if show or save:
            if unit !='dBc':
                if LANG=='EN':
                    axs.set_ylabel(r'\textbf{Phase noise amplitude in $\frac{\text{\textbf{A. U.}}^2}{\text{\textbf{Hz}}}$')
                if LANG=='DE':
                    axs.set_ylabel(r'\textbf{Phasenrauschamplitude in $\frac{\text{\textbf{A. U.}}^2}{\text{\textbf{Hz}}}$')
            else:
                if LANG == 'EN':
                    axs.set_ylabel(r'\textbf{Phase noise PSD in} $\frac{\text{\textbf{dBC}}}{\text{\textbf{Hz}}}$')
                if LANG == 'DE':
                    axs.set_ylabel(r'\textbf{Phasenrauschleistungsdichte in} $\frac{\text{\textbf{dBc}}}{\text{\textbf{Hz}}}$')
            if signalFreq!=1.0:
                if LANG == 'EN':
                    axs.set_xlabel(r'\textbf{Offset~frequency to '+str(signalFreq)+' Hz Signal in Hz}')
                if LANG == 'DE':
                    axs.set_xlabel(r'\textbf{Frequenzdifferenz zu einem ' + locale.format_string('%g',signalFreq) + '-Hz-Signal in Hz}')
            else:
                if LANG== 'EN':
                    axs.set_xlabel(r'$\frac{{\text{\textbf{Offset~frequency}}}}{\text{\textbf{Signal~frequency}}}$ \textbf{in} $\frac{\text{\textbf{Hz}}}{\text{\textbf{Hz}}}$')
                if LANG== 'DE':
                    axs.set_xlabel(r'$\frac{{\text{\textbf{Frequenzdifferenz}}}}{\text{\textbf{Signalfrequenz}}}$ \textbf{in} $\frac{\text{\textbf{Hz}}}{\text{\textbf{Hz}}}$')
            axs.grid(True, which="both")
            axs.legend(ncol=1,loc='upper left')
            try:
                axSincSens.legend(ncol=1, loc='upper right')
            except:
                pass
            if yLims is not None:
                axs.set_ylim(yLims)
            #fig.tight_layout()
            fig.show()
        if plotTimeDevs:
            axTimeDev.set_ylabel(r'\textbf{Time Deviation from Nominal in ns}')
            axTimeDev.set_xlabel(r'\textbf{Releative time from slice start in s}')
            axTimeDev.legend(ncol=2)
            figTimeDev.show()
        if xLims is not None:
            axs.set_xlim(xLims)
            try:
                axSincSens.set_xlim(xLims)
            except:
                pass
        if save:

            try:
                paramsStr=str(plotSincSensForLength['maxFreq']).replace(' ','_')+'_Hz_'+'_'.join(str(v) for v in plotSincSensForLength['length'])
            except:
                paramsStr = "None"
            fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' +'sincSens_'+paramsStr+'_PhaseNoise.png'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' +'sincSens_'+paramsStr+'_PhaseNoise.pdf'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' + 'sincSens_' + paramsStr + '_PhaseNoise.svg'), dpi=300, bbox_inches='tight')
            globals()['figSaveCounter']+=1
        return fig,axs

    """
    def plotAllanDev(self,fig=None,ax=None,show=False):
        if fig==None and ax==None:
            fig, ax = plt.subplots()
        phaseDevInrad=self.interpolatedDeviationFromNominal/self.deltaT
        observationLength=self.expectedTime[-1]
        taus=np.logspace(-3, np.log10(observationLength/4), 100)#100 points logspaced until observation length
        (t2, ad, ade, adn) = allantools.oadev(phaseDevInrad, rate=self.fs, data_type="phase", taus=taus)
        ax.loglog(t2, ad,label=self.title)
        if show:
            ax.grid()
            ax.legend()
            ax.set_xlabel(r'\textbf{Averaging time $\tau$ in s}')
            ax.set_ylabel(r'\textbf{Relative Allan deviation $\sigma(\tau)$ in cycles}')
            fig.show()
        return fig,ax
    """

def generateFitWithPhaseNoise(freq,fs=1000,t_jitter=100e-9,lengthInS=jitterSimuLengthInS,A0=1,phi0=0,linearFreqCorrection=localFreqqCorr):
    #TODO change interface
    if t_jitter <= 0:
        fs=jitterGensForSimulations[int(-1*t_jitter)].fs
    originalTimpoints=np.linspace(0,lengthInS,num=int(fs*lengthInS))
    Signal=A0*np.sin(originalTimpoints*np.pi*2*freq+phi0)
    if t_jitter >0:
        jitter=np.random.normal(scale=t_jitter, size=Signal.size)
    else:
        jitter=jitterGensForSimulations[int(-1*t_jitter)].getrandomDeviations(Signal.size)
    if linearFreqCorrection:
        idx=np.arange(originalTimpoints.size)
        m, b = np.polyfit(idx, jitter.astype(np.float64), 1)
        timeWJitter=originalTimpoints+jitter-(idx*m+b)
    else:
        timeWJitter = originalTimpoints + jitter
    fitparams=st.threeparsinefit(Signal,timeWJitter,freq)
    del jitter ,timeWJitter,Signal,originalTimpoints
    return st.phase(fitparams)-phi0,st.amplitude(fitparams)/A0

def getmuAndSTdForFreq(testparams,numOfruns=1000):
    freq=testparams[0]
    t_jitter=testparams[1]
    length = testparams[2]
    FitPhases=np.zeros(numOfruns)
    FitMags=np.zeros(numOfruns)
    for i in range(numOfruns):
        FitPhases[i],FitMags[i]=generateFitWithPhaseNoise(freq,t_jitter=t_jitter,lengthInS=length)
    ampPercentiles=np.percentile(FitMags, np.array([5,32,50,68,95]))
    phasePercentiles=np.percentile(FitPhases, np.array([5,32,50,68,95]))
    stdPhase=np.std(FitPhases)
    meanPhase=np.mean(FitPhases)
    stdMag=np.std(FitMags)
    meanMag=np.mean(FitMags)
    del FitPhases,FitMags
    return stdPhase,\
           meanPhase,\
           stdMag,\
           meanMag,\
           ampPercentiles[0],\
           ampPercentiles[1],\
           ampPercentiles[2],\
           ampPercentiles[3],\
           ampPercentiles[4],\
           phasePercentiles[0],\
           phasePercentiles[1],\
           phasePercentiles[2],\
           phasePercentiles[3],\
           phasePercentiles[4]



def find_nearest_indices(long_vector, short_vector):
    # Compute the absolute differences using broadcasting
    diffs = np.abs(long_vector[:, np.newaxis] - short_vector)
    # Find the indices of the minimum differences
    nearest_indices = np.argmin(diffs, axis=0)
    return nearest_indices


class SineExcitationExperiment:
    def __init__(self, dataFile, idx, sensor='0xbccb0000_MPU_9250', quantity='Acceleration', mainAxis=2,
                 interpolations=[], interpolationFactors=[]):
        self.experimentIDX = idx
        self.dataFile = dataFile
        self.sensorStartIDX = \
        self.dataFile['EXPERIMENTS/Sine excitation']["{:05d}".format(idx) + 'Sine_Excitation'][sensor].attrs[
            'Start_index']
        self.sensorStopIDX = \
        self.dataFile['EXPERIMENTS/Sine excitation']["{:05d}".format(idx) + 'Sine_Excitation'][sensor].attrs[
            'Stop_index']
        self.data = self.dataFile['RAWDATA'][sensor][quantity][:, self.sensorStartIDX:self.sensorStopIDX]
        self.reltime = self.dataFile['RAWDATA'][sensor]['Absolutetime'][0, self.sensorStartIDX:self.sensorStopIDX]
        self.reltime = self.reltime - self.reltime[0]
        self.reltime = self.reltime.astype(float) / 1e9
        self.fs = self.reltime.size / self.reltime[-1]
        self.deltaT = 1 / self.fs
        self.freq = \
        self.dataFile['EXPERIMENTS/Sine excitation']["{:05d}".format(idx) + 'Sine_Excitation'][sensor][quantity][
            'Sin_Fit_freq'][:, 0][2]

        abcw = st2.fourparsinefit(self.data[mainAxis, :], self.reltime, self.freq)
        self.actualFreq = abcw[3]

        # Initialize cache flags and storage
        self._fft_computed = False
        self._interpolated_fft_computed = False
        self._multisine_computed = False

        # Cache storage for expensive computations
        self._multisine_cache = {
            'computed': False,
            'multiSineParamsABC': None,
            'multiSineFitresults': None,
            'multisineFitFreqs': None,
            'numOverTones': None,
            'numLinesAround': None,
            'startStopFreqs': None
        }

        # Disk cache settings
        self.cache_dir = os.path.join(os.getcwd(), 'multisine_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.generateFFT()
        self.interPolationFactors = interpolationFactors
        self.interpolations = interpolations
        self.generateInterpolatedFFT()
        self.generateMultiSineFit()
        self.name = 'MultiSineFFTComp_' + self.dataFile['RAWDATA'][sensor].attrs['Sensor_name'].replace(' ',
                                                                                                        '_') + '_Exp_' + str(
            idx) + '_Axis_' + str(mainAxis) + '_freq_' + f'{self.freq:.2f}' + "_Hz"
        print("INIT DONE")

    def generateFFT(self):
        """Generate FFT data and cache results"""
        if self._fft_computed:
            print("FFT already computed, using cached results")
            return

        lengthmax = self.data.shape[1]
        lengthToTest = int(lengthmax / 2) + np.arange(int(lengthmax / 2))
        fftbinwidth = self.fs / lengthToTest
        nonIntPeriodFraction = self.actualFreq % fftbinwidth
        self.numPointsToUse = lengthToTest[np.argmin(nonIntPeriodFraction)]
        self.fftLowLeak = 2 * np.fft.rfft(self.data[:, :self.numPointsToUse], axis=1) / self.numPointsToUse
        self.fftFreqslowLeak = np.fft.rfftfreq(self.numPointsToUse, d=self.deltaT)
        self.fft = 2 * np.fft.rfft(self.data[:, :], axis=1) / lengthmax
        self.fftFreqs = np.fft.rfftfreq(lengthmax, d=self.deltaT)

        self._fft_computed = True
        print("FFT computed and cached")

    def generateInterpolatedFFT(self):
        """Generate interpolated FFT data and cache results"""
        if self._interpolated_fft_computed:
            print("Interpolated FFT already computed, using cached results")
            return

        self.interpolatedFFTFreqsLowLeak = {}
        self.interpolatedFFTFreqs = {}
        self.interpolatedFFTLowLeak = {}
        self.interpolatedFFT = {}

        for interpolationFactor in self.interPolationFactors:
            numAxis = self.data.shape[0]
            aqTimesLowLeak = np.linspace(self.reltime[0], self.reltime[self.numPointsToUse - 1],
                                         num=self.numPointsToUse * interpolationFactor)
            self.interpolatedFFTFreqsLowLeak[interpolationFactor] = np.fft.rfftfreq(aqTimesLowLeak.size,
                                                                                    d=self.deltaT / interpolationFactor)
            self.interpolatedFFTLowLeak[interpolationFactor] = {}

            for interpolatorKind in self.interpolations:
                print("Interpolating with " + str(interpolationFactor) + " times " + interpolatorKind)
                result = []
                for i in range(numAxis):
                    interpolator = sp.interpolate.interp1d(self.reltime, self.data[i, :], kind=interpolatorKind)
                    interpolatedData = interpolator(aqTimesLowLeak)
                    result.append(2 * np.fft.rfft(interpolatedData) / interpolatedData.size)
                self.interpolatedFFTLowLeak[interpolationFactor][interpolatorKind] = np.array(result)

            aqTimes = np.linspace(self.reltime[0], self.reltime[-1], num=self.reltime.size * interpolationFactor)
            self.interpolatedFFTFreqs[interpolationFactor] = np.fft.rfftfreq(aqTimes.size,
                                                                             d=self.deltaT / interpolationFactor)
            self.interpolatedFFT[interpolationFactor] = {}

            for interpolatorKind in self.interpolations:
                result = []
                for i in range(numAxis):
                    interpolator = sp.interpolate.interp1d(self.reltime, self.data[i, :], kind=interpolatorKind)
                    interpolatedData = interpolator(aqTimes)
                    result.append(2 * np.fft.rfft(interpolatedData) / interpolatedData.size)
                self.interpolatedFFT[interpolationFactor][interpolatorKind] = np.array(result)

        self._interpolated_fft_computed = True
        print("Interpolated FFT computed and cached")

    def _generate_multisine_cache_key(self, numLinesAround, numOverTones):
        """
        Generate a unique cache key for multi-sine fit parameters

        Parameters:
        -----------
        numLinesAround : int
            Number of frequency lines around each harmonic
        numOverTones : int
            Number of harmonic overtones to analyze

        Returns:
        --------
        str : Unique cache key for this configuration
        """
        # Create a hash based on critical parameters that affect the calculation
        # Convert numpy types to native Python types for JSON serialization
        cache_data = {
            'experimentIDX': int(self.experimentIDX),
            'actualFreq': float(self.actualFreq),
            'fs': float(self.fs),
            'data_shape': [int(x) for x in self.data.shape],
            'numLinesAround': int(numLinesAround),
            'numOverTones': int(numOverTones),
            'reltime_hash': hashlib.md5(self.reltime.tobytes()).hexdigest()[:8],
            'data_hash': hashlib.md5(self.data.tobytes()).hexdigest()[:8]
        }

        # Create unique string and hash it
        cache_string = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()

        return f"multisine_exp{int(self.experimentIDX):05d}_{cache_hash}.npz"

    def _save_multisine_to_disk(self, cache_key, multiSineParamsABC, multiSineFitresults,
                                multisineFitFreqs, numOverTones, numLinesAround, startStopFreqs):
        """
        Save multi-sine fit results to disk

        Parameters:
        -----------
        cache_key : str
            Unique identifier for this cache file
        multiSineParamsABC : list
            ABC parameters from multi_threeparsinefit
        multiSineFitresults : np.array
            Complex results from multi_complex
        multisineFitFreqs : np.array
            Frequency array for multi-sine fit
        numOverTones : int
            Number of overtones used
        numLinesAround : int
            Number of lines around each harmonic
        startStopFreqs : list
            Start and stop frequencies for each band
        """
        try:
            cache_filepath = os.path.join(self.cache_dir, cache_key)

            # Prepare data for saving
            save_data = {
                'multiSineFitresults': multiSineFitresults,
                'multisineFitFreqs': multisineFitFreqs,
                'numOverTones': int(numOverTones),
                'numLinesAround': int(numLinesAround),
                'startStopFreqs': np.array(startStopFreqs, dtype=object),
                'actualFreq': float(self.actualFreq),
                'fs': float(self.fs),
                'experimentIDX': int(self.experimentIDX),
                'cache_version': '1.0'
            }

            # Save ABC parameters separately (they might contain different data types)
            for i, abc in enumerate(multiSineParamsABC):
                save_data[f'multiSineParamsABC_{i}'] = np.array(abc, dtype=object)
            save_data['num_axes'] = len(multiSineParamsABC)

            # Save with compression
            np.savez_compressed(cache_filepath, **save_data)
            print(f"Multi-sine results saved to: {cache_key}")

        except Exception as e:
            print(f"Warning: Could not save multi-sine cache to disk: {e}")

    def _load_multisine_from_disk(self, cache_key):
        """
        Load multi-sine fit results from disk

        Parameters:
        -----------
        cache_key : str
            Unique identifier for this cache file

        Returns:
        --------
        dict or None : Loaded data dictionary or None if loading failed
        """
        try:
            cache_filepath = os.path.join(self.cache_dir, cache_key)

            if not os.path.exists(cache_filepath):
                return None

            # Load the data
            loaded = np.load(cache_filepath, allow_pickle=True)

            # Reconstruct ABC parameters
            num_axes = int(loaded['num_axes'])
            multiSineParamsABC = []
            for i in range(num_axes):
                abc_key = f'multiSineParamsABC_{i}'
                if abc_key in loaded:
                    multiSineParamsABC.append(loaded[abc_key].tolist())

            result = {
                'multiSineParamsABC': multiSineParamsABC,
                'multiSineFitresults': loaded['multiSineFitresults'],
                'multisineFitFreqs': loaded['multisineFitFreqs'],
                'numOverTones': int(loaded['numOverTones']),
                'numLinesAround': int(loaded['numLinesAround']),
                'startStopFreqs': loaded['startStopFreqs'].tolist(),
                'actualFreq': float(loaded['actualFreq']),
                'fs': float(loaded['fs']),
                'experimentIDX': int(loaded['experimentIDX'])
            }

            print(f"Multi-sine results loaded from: {cache_key}")
            return result

        except Exception as e:
            print(f"Warning: Could not load multi-sine cache from disk: {e}")
            return None

    def _check_disk_cache_validity(self, loaded_data):
        """
        Check if loaded cache data is still valid for current experiment

        Parameters:
        -----------
        loaded_data : dict
            Data loaded from disk cache

        Returns:
        --------
        bool : True if cache is valid, False otherwise
        """
        try:
            # Check critical parameters match with detailed logging
            exp_id_match = loaded_data['experimentIDX'] == self.experimentIDX
            freq_match = abs(loaded_data['actualFreq'] - self.actualFreq) < 1e-6  # More lenient tolerance
            fs_match = abs(loaded_data['fs'] - self.fs) < 1e-6  # More lenient tolerance

            # Debug output
            if not exp_id_match:
                print(
                    f"Cache invalid: Experiment ID mismatch. Cached: {loaded_data['experimentIDX']}, Current: {self.experimentIDX}")
            if not freq_match:
                print(
                    f"Cache invalid: Frequency mismatch. Cached: {loaded_data['actualFreq']}, Current: {self.actualFreq}, Diff: {abs(loaded_data['actualFreq'] - self.actualFreq)}")
            if not fs_match:
                print(
                    f"Cache invalid: Sampling rate mismatch. Cached: {loaded_data['fs']}, Current: {self.fs}, Diff: {abs(loaded_data['fs'] - self.fs)}")

            checks = [exp_id_match, freq_match, fs_match]
            is_valid = all(checks)

            if is_valid:
                print("Disk cache validation passed - all parameters match")

            return is_valid

        except (KeyError, TypeError) as e:
            print(f"Cache invalid: Missing key or type error: {e}")
            return False

    def _calculate_vonmises_phase_stats(self, phase_data, window_size=5):
        """
        Calculate Von-Mises distribution parameters for phase data using sliding window

        Parameters:
        -----------
        phase_data : array
            1D array of phase values in radians
        window_size : int
            Size of sliding window (default: 6)

        Returns:
        --------
        dict : Dictionary containing unwrapped mean phases, kappa values, and sigma equivalents
        """
        n_points = len(phase_data)
        mean_phases = np.zeros(n_points)
        kappa_values = np.zeros(n_points)
        sigma_equivalents = np.zeros(n_points)

        # Calculate Von-Mises parameters for the first window (will be used for edge handling)
        first_window_phases = phase_data[:window_size]
        first_kappa, first_loc, first_scale = vonmises.fit(first_window_phases, fscale=1)
        first_mean = first_loc
        first_sigma_eq = np.sqrt(1.0 / first_kappa) if first_kappa > 0 else 1.0

        # Fill first 6 values with the same parameters (filter not settled)
        for i in range(min(window_size, n_points)):
            mean_phases[i] = first_mean
            kappa_values[i] = first_kappa
            sigma_equivalents[i] = first_sigma_eq

        # Sliding window calculation for remaining points
        for i in range(window_size, n_points):
            # Extract window around current point
            start_idx = max(0, i - window_size + 1)
            end_idx = min(n_points, i + 1)
            window_phases = phase_data[start_idx:end_idx]

            try:
                # Fit Von-Mises distribution to window
                kappa, loc, scale = vonmises.fit(window_phases, fscale=1)
                mean_phase = loc

                # Convert kappa to sigma equivalent: σ² = 1/κ
                sigma_eq = np.sqrt(1.0 / kappa) if kappa > 0 else 1.0

                mean_phases[i] = mean_phase
                kappa_values[i] = kappa
                sigma_equivalents[i] = sigma_eq

            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                # Fallback to previous values if fitting fails
                if i > 0:
                    mean_phases[i] = mean_phases[i - 1]
                    kappa_values[i] = kappa_values[i - 1]
                    sigma_equivalents[i] = sigma_equivalents[i - 1]
                else:
                    mean_phases[i] = np.mean(window_phases)
                    kappa_values[i] = 1.0
                    sigma_equivalents[i] = 1.0

        # Unwrap phases and apply modulo to keep in [-π, π]
        unwrapped_mean = np.unwrap(mean_phases)
        unwrapped_mean = np.mod(unwrapped_mean + np.pi, 2 * np.pi) - np.pi

        return {
            'mean_phases': unwrapped_mean,
            'kappa_values': kappa_values,
            'sigma_equivalents': sigma_equivalents,
            'upper_bound': unwrapped_mean + sigma_equivalents,
            'lower_bound': unwrapped_mean - sigma_equivalents
        }

    def generateMultiSineFit(self, numLinesAround=100, numOverTones=5, force_recompute=False, use_disk_cache=True):
        """
        Generate multi-sine fit with memory and disk caching support

        Parameters:
        -----------
        numLinesAround : int
            Number of frequency lines around each harmonic
        numOverTones : int
            Number of harmonic overtones to analyze
        force_recompute : bool
            If True, force recomputation even if cached results exist
        use_disk_cache : bool
            If True, try to load/save results from/to disk
        """
        # Generate cache key for disk storage
        cache_key = self._generate_multisine_cache_key(numLinesAround, numOverTones)

        # Check memory cache first
        if (self._multisine_cache['computed'] and
                not force_recompute and
                self._multisine_cache['numLinesAround'] == numLinesAround and
                self._multisine_cache['numOverTones'] == numOverTones):
            print("Multi-sine fit already computed with same parameters, using memory cache")
            # Restore cached results to instance variables
            self.multiSineFitresults = self._multisine_cache['multiSineFitresults']
            self.multisineFitFreqs = self._multisine_cache['multisineFitFreqs']
            self.numOverTones = self._multisine_cache['numOverTones']
            self.numLinesAround = self._multisine_cache['numLinesAround']
            self.startStopFreqs = self._multisine_cache['startStopFreqs']
            return

        # Try to load from disk cache
        if use_disk_cache and not force_recompute:
            loaded_data = self._load_multisine_from_disk(cache_key)
            if loaded_data is not None and self._check_disk_cache_validity(loaded_data):
                print("Multi-sine fit loaded from disk cache")

                # Restore from disk cache to instance variables
                self.multiSineFitresults = loaded_data['multiSineFitresults']
                self.multisineFitFreqs = loaded_data['multisineFitFreqs']
                self.numOverTones = loaded_data['numOverTones']
                self.numLinesAround = loaded_data['numLinesAround']
                self.startStopFreqs = loaded_data['startStopFreqs']

                # Also update memory cache
                self._multisine_cache.update({
                    'computed': True,
                    'multiSineParamsABC': loaded_data['multiSineParamsABC'],
                    'multiSineFitresults': self.multiSineFitresults.copy(),
                    'multisineFitFreqs': self.multisineFitFreqs.copy(),
                    'numOverTones': self.numOverTones,
                    'numLinesAround': self.numLinesAround,
                    'startStopFreqs': self.startStopFreqs.copy()
                })
                return
            elif loaded_data is not None:
                print("Disk cache found but invalid, recomputing...")

        print("Computing multi-sine fit (expensive matrix inversion)...")

        self.numLinesAround = numLinesAround
        self.numOverTones = numOverTones
        self.binwidth = self.fftFreqs[1] - self.fftFreqs[0]
        fs = self.fs

        rawfreqs = []
        rawfreqs.append((np.arange(2 * numLinesAround + 1) + 1) * self.binwidth)
        self.startStopFreqs = [(rawfreqs[0][0], rawfreqs[0][-1])]
        self.startStopIDXs = []

        for k in range(numOverTones):
            rawfreqs.append(
                (np.arange(numLinesAround * 2 + 1) - numLinesAround) * self.binwidth + self.actualFreq * (k + 1))
            self.startStopFreqs.append([self.actualFreq * (k + 1) - numLinesAround * self.binwidth,
                                        self.actualFreq * (k + 1) + numLinesAround * self.binwidth])

        basebandFreqregions = []

        def ConverToBasebandFreqs(region):
            nyqistbandStart = np.round(region[0] / ((self.fs) / 2) - 0.5)
            nyqistbandStop = np.round(region[1] / ((self.fs) / 2) - 0.5)
            if nyqistbandStart != nyqistbandStop:
                print("WAAAA region is in two nyquist bands using upper band; baseband will be negative")
            nyqistband = np.max([nyqistbandStart, nyqistbandStop])
            basebandFreqs = np.array(region) - (nyqistband * (self.fs / 2))
            return basebandFreqs

        def checkRegionOverlap(reg1, reg2):
            baseBandReg1 = ConverToBasebandFreqs(reg1)
            baseBandReg2 = ConverToBasebandFreqs(reg2)
            width1 = (reg1[1] - reg1[0]) / 2
            witdh2 = (reg2[1] - reg2[0]) / 2
            mindistance = (width1 + witdh2)
            dist = np.mean(baseBandReg1) - np.mean(baseBandReg2)
            if abs(dist) < mindistance:
                return True
            else:
                return False

        freqs = []
        numskippedBands = 0

        for i, band in enumerate(self.startStopFreqs):
            if i == 0:
                basebandFreqregions.append([band[0], band[1]])
                freqs.append(rawfreqs[i])
            else:
                overLapDetected = False
                for region in basebandFreqregions:
                    if checkRegionOverlap(region, band):
                        overLapDetected = True
                        print(
                            "Region overlap detected skipped frequency band" + str(band) + "Due to overlap with" + str(
                                region))
                if not overLapDetected:
                    basebandFreqregions.append([band[0], band[1]])
                    freqs.append(rawfreqs[i])
                else:
                    numskippedBands += 1

        self.multisineFitFreqs = np.array(freqs).flatten()
        self.numOverTones -= numskippedBands

        # Perform the expensive computation
        multiSineParams = []
        multiSineParamsABC = []

        for i in range(self.data.shape[0]):
            abc = st2.multi_threeparsinefit(self.data[i, :], self.reltime, self.multisineFitFreqs)
            multiSineParamsABC.append(abc)
            fitResult = st2.multi_complex(abc)
            multiSineParams.append(fitResult)

        self.multiSineFitresults = np.array(multiSineParams)

        # Save to disk cache
        if use_disk_cache:
            self._save_multisine_to_disk(cache_key, multiSineParamsABC, self.multiSineFitresults,
                                         self.multisineFitFreqs, self.numOverTones,
                                         self.numLinesAround, self.startStopFreqs)

        # Update memory cache
        self._multisine_cache.update({
            'computed': True,
            'multiSineParamsABC': [abc.copy() if hasattr(abc, 'copy') else abc for abc in multiSineParamsABC],
            'multiSineFitresults': self.multiSineFitresults.copy(),
            'multisineFitFreqs': self.multisineFitFreqs.copy(),
            'numOverTones': self.numOverTones,
            'numLinesAround': self.numLinesAround,
            'startStopFreqs': self.startStopFreqs.copy()
        })

        print("Multi-sine fit computed and cached successfully (memory + disk)")

    def getMultiSineFitResults(self, numLinesAround=None, numOverTones=None):
        """
        Get multi-sine fit results, using cached version if parameters match

        Parameters:
        -----------
        numLinesAround : int, optional
            Number of frequency lines around each harmonic
        numOverTones : int, optional
            Number of harmonic overtones to analyze

        Returns:
        --------
        dict : Dictionary containing fit results and metadata
        """
        # Use current parameters if not specified
        if numLinesAround is None:
            numLinesAround = getattr(self, 'numLinesAround', 100)
        if numOverTones is None:
            numOverTones = getattr(self, 'numOverTones', 5)

        # Generate fit if needed with specified parameters
        self.generateMultiSineFit(numLinesAround=numLinesAround, numOverTones=numOverTones)

        return {
            'multiSineFitresults': self.multiSineFitresults,
            'multisineFitFreqs': self.multisineFitFreqs,
            'numOverTones': self.numOverTones,
            'numLinesAround': self.numLinesAround,
            'startStopFreqs': self.startStopFreqs,
            'actualFreq': self.actualFreq,
            'fs': self.fs
        }

    def clearMultiSineCache(self, clear_disk_cache=False):
        """
        Clear the multi-sine fit cache to force recomputation

        Parameters:
        -----------
        clear_disk_cache : bool
            If True, also remove disk cache files for this experiment
        """
        self._multisine_cache = {
            'computed': False,
            'multiSineParamsABC': None,
            'multiSineFitresults': None,
            'multisineFitFreqs': None,
            'numOverTones': None,
            'numLinesAround': None,
            'startStopFreqs': None
        }

        if clear_disk_cache:
            try:
                # Remove all cache files for this experiment
                cache_pattern = f"multisine_exp{self.experimentIDX}_"
                for filename in os.listdir(self.cache_dir):
                    if filename.startswith(cache_pattern):
                        cache_filepath = os.path.join(self.cache_dir, filename)
                        os.remove(cache_filepath)
                        print(f"Removed disk cache: {filename}")
            except Exception as e:
                print(f"Warning: Could not clear disk cache: {e}")

        print("Multi-sine cache cleared")

    def getCacheStatus(self):
        """Get status of all cached computations"""
        return {
            'fft_computed': self._fft_computed,
            'interpolated_fft_computed': self._interpolated_fft_computed,
            'multisine_computed': self._multisine_cache['computed'],
            'multisine_params': {
                'numLinesAround': self._multisine_cache['numLinesAround'],
                'numOverTones': self._multisine_cache['numOverTones']
            } if self._multisine_cache['computed'] else None
        }

    def get_cache_info(self):
        """
        Get information about cache usage and disk storage

        Returns:
        --------
        dict : Cache information including disk usage
        """
        memory_status = self.getCacheStatus()

        # Check disk cache
        disk_cache_files = []
        disk_cache_size = 0
        try:
            cache_pattern = f"multisine_exp{self.experimentIDX}_"
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(cache_pattern):
                    filepath = os.path.join(self.cache_dir, filename)
                    size = os.path.getsize(filepath)
                    disk_cache_files.append({
                        'filename': filename,
                        'size_mb': size / (1024 * 1024),
                        'modified': os.path.getmtime(filepath)
                    })
                    disk_cache_size += size
        except Exception as e:
            print(f"Warning: Could not read disk cache info: {e}")

        return {
            'memory_cache': memory_status,
            'disk_cache': {
                'files': disk_cache_files,
                'total_size_mb': disk_cache_size / (1024 * 1024),
                'cache_directory': self.cache_dir
            }
        }

    def getSNR(self, axis=2):
        """Calculate SNR using cached multi-sine results"""
        # Ensure multi-sine fit is computed
        if not self._multisine_cache['computed']:
            self.generateMultiSineFit()

        refIDX = self.numLinesAround * 3 + 1
        referenceAMP = abs(self.multiSineFitresults[axis, refIDX])
        referenceN = (np.sum(abs(self.multiSineFitresults[axis,
                                 refIDX - self.numLinesAround:refIDX + self.numLinesAround + 1])) - referenceAMP) / (
                                 2 * self.numLinesAround)
        self.referenceFFTFreqsForSNR = self.multisineFitFreqs[
                                       refIDX - self.numLinesAround:refIDX + self.numLinesAround + 1]
        self.fitSNR = referenceAMP / referenceN
        print("SineFit SNR is " + str(self.fitSNR))

        self.FFTSNR = {}
        for interpolationFactor in self.interPolationFactors:
            self.FFTSNR[interpolationFactor] = {}
            for interpolatorKind in self.interpolations:
                idxs = find_nearest_indices(self.interpolatedFFTFreqsLowLeak[interpolationFactor],
                                            self.referenceFFTFreqsForSNR)
                centerIDX = idxs[self.numLinesAround]
                spectralData = abs(self.interpolatedFFTLowLeak[interpolationFactor][interpolatorKind][axis,
                                   centerIDX - self.numLinesAround:centerIDX + self.numLinesAround + 1])
                fftAMP = abs(spectralData[self.numLinesAround])
                fftN = (np.sum(abs(self.interpolatedFFTLowLeak[interpolationFactor][interpolatorKind][axis,
                                   centerIDX - self.numLinesAround:centerIDX + self.numLinesAround + 1])) - fftAMP) / (
                                   2 * self.numLinesAround)
                SNR = fftAMP / fftN
                self.FFTSNR[interpolationFactor][interpolatorKind] = SNR
                print("SNR for " + str(interpolationFactor) + " times " + interpolatorKind + " is " + str(SNR))

        result = {'sineSNR': self.fitSNR, 'FFTSNR': self.FFTSNR, 'freq': self.freq, 'actualFreq': self.actualFreq}
        json.dump(result, open('SNRParams/' + self.name + 'SNR_params.json', 'w'))
        return result

    def plotFFTandSineFit(self, axisToPlot=[2], plotQoutient=False, markerSize=1, plotHighLeak=False, filterWidth=0.5,
                          phase=False):
        """
        Plot FFT and sine fit results with light gray backgrounds for zoom regions

        Parameters:
        -----------
        axisToPlot : list
            List of axis indices to plot
        plotQoutient : bool
            Whether to plot the quotient between FFT and fit
        markerSize : float
            Size of markers in the plot
        plotHighLeak : bool
            Whether to plot high leakage FFT results
        filterWidth : float
            Width for Gaussian filter smoothing
        phase : bool
            If True, plot phase; if False, plot amplitude
        """
        # Ensure multi-sine fit is computed
        import numpy as np

        if not self._multisine_cache['computed']:
            self.generateMultiSineFit()

        fig = plt.figure()
        if plotQoutient:
            gs = gridspec.GridSpec(len(axisToPlot) * 3, 1)
            ax = []
            bax = []
            baxQuatient = []
            idxOffset = 1
        else:
            gs = gridspec.GridSpec(len(axisToPlot) * 2, 1)
            ax = []
            bax = []
            idxOffset = 0

        numPlotsPerQuant = 2 + idxOffset
        baxXlims = []

        for i in range(self.numOverTones + 1):
            start = i * (2 * self.numLinesAround + 1)
            stop = (i + 1) * (2 * self.numLinesAround + 1)
            baxXlims.append([self.multisineFitFreqs[start], self.multisineFitFreqs[stop - 1]])

        for i in range(len(axisToPlot)):
            ax.append(plt.subplot(gs[i * numPlotsPerQuant, 0]))
            bax.append(brokenaxes(xlims=baxXlims, subplot_spec=gs[i * numPlotsPerQuant + 1, 0], fig=fig, d=.005))
            if plotQoutient:
                baxQuatient.append(
                    brokenaxes(xlims=baxXlims, subplot_spec=gs[i * numPlotsPerQuant + 2, 0], fig=fig, d=.005))

        # Add light gray shading only for the frequency regions where zooms are plotted
        for i, idx in enumerate(axisToPlot):
            for freq_region in self.startStopFreqs:
                # Shade the frequency regions that correspond to zoom windows
                ax[i].axvspan(freq_region[0], freq_region[1], alpha=0.15, color='gray', zorder=0)

        # Set light gray background for broken axes (zoom regions only)
        for i, idx in enumerate(axisToPlot):
            for j, axis in enumerate(bax[i].axs):
                axis.set_facecolor('#f8f8f8')  # Light gray background for zoom regions
            if plotQoutient:
                for j, axis in enumerate(baxQuatient[i].axs):
                    axis.set_facecolor('#f8f8f8')  # Light gray background for quotient zoom regions

        for i, idx in enumerate(axisToPlot):
            if not phase:
                ax[i].semilogy(self.fftFreqs[1:], sp.ndimage.gaussian_filter1d(np.abs(self.fft[idx, 1:]), filterWidth),
                               label=r'\textbf{DFT }', lw=1)
                bax[i].plot(self.fftFreqs[1:], np.abs(self.fft[idx, 1:]),
                            label=r'\textbf{DFT wening Leckeffekt}', lw=1, marker='o', markersize=markerSize)
                ax[i].semilogy(self.fftFreqslowLeak[1:],
                               sp.ndimage.gaussian_filter1d(np.abs(self.fftLowLeak[idx, 1:]), filterWidth),
                               label=r'\textbf{DFT wening Leckeffekt}', lw=1)
                bax[i].plot(self.fftFreqslowLeak[1:], np.abs(self.fftLowLeak[idx, 1:]),
                            label=r'\textbf{DFT wening Leckeffekt}', lw=1, marker='o', markersize=markerSize)
            else:  # phase plot with Von-Mises enhancement
                # Calculate Von-Mises statistics for FFT data
                fft_stats = self._calculate_vonmises_phase_stats(np.angle(self.fft[idx, 1:]))
                fft_lowleak_stats = self._calculate_vonmises_phase_stats(np.angle(self.fftLowLeak[idx, 1:]))

                # Plot FFT phase with uncertainty bands (color will be auto-assigned)
                fft_line = ax[i].plot(self.fftFreqs[1:], fft_stats['mean_phases'],
                                      label=r'\textbf{DFT}', lw=1, zorder=2)
                fft_color = fft_line[0].get_color()
                ax[i].fill_between(self.fftFreqs[1:], fft_stats['lower_bound'], fft_stats['upper_bound'],
                                   alpha=0.2, color=fft_color, zorder=1)

                # Plot mean as line and raw phases as scatter for broken axes (zoom regions)
                bax[i].plot(self.fftFreqs[1:], fft_stats['mean_phases'],
                            color=fft_color, lw=1, zorder=2)
                # Scatter raw phase values
                bax[i].scatter(self.fftFreqs[1:], np.angle(self.fft[idx, 1:]),
                               color=fft_color, s=markerSize * 10, alpha=0.7, zorder=3)
                # Add uncertainty bands to broken axes
                bax[i].fill_between(self.fftFreqs[1:], fft_stats['lower_bound'], fft_stats['upper_bound'],
                                    alpha=0.2, color=fft_color, zorder=1)

                # Plot low leak FFT phase with uncertainty bands
                fft_lowleak_line = ax[i].plot(self.fftFreqslowLeak[1:], fft_lowleak_stats['mean_phases'],
                                              label=r'\textbf{DFT wening Leckeffekt}', lw=1, zorder=2)
                fft_lowleak_color = fft_lowleak_line[0].get_color()
                ax[i].fill_between(self.fftFreqslowLeak[1:], fft_lowleak_stats['lower_bound'],
                                   fft_lowleak_stats['upper_bound'],
                                   alpha=0.2, color=fft_lowleak_color, zorder=1)

                # Plot mean as line and raw phases as scatter for broken axes (zoom regions)
                bax[i].plot(self.fftFreqslowLeak[1:], fft_lowleak_stats['mean_phases'],
                            color=fft_lowleak_color, lw=1, zorder=2)
                # Scatter raw phase values
                bax[i].scatter(self.fftFreqslowLeak[1:], np.angle(self.fftLowLeak[idx, 1:]),
                               color=fft_lowleak_color, s=markerSize * 10, alpha=0.7, zorder=3)
                # Add uncertainty bands to broken axes
                bax[i].fill_between(self.fftFreqslowLeak[1:], fft_lowleak_stats['lower_bound'],
                                    fft_lowleak_stats['upper_bound'],
                                    alpha=0.2, color=fft_lowleak_color, zorder=1)

        minFFT = np.power(10, np.floor(np.log10(np.min(np.abs(self.fft[axisToPlot, 1:])))))
        maxFFT = np.power(10, np.ceil(np.log10(np.max(np.abs(self.fft[axisToPlot, 1:])))))
        minSine = np.power(10, np.floor(np.log10(np.min(np.abs(self.multiSineFitresults[axisToPlot, :])))))
        maxSine = np.power(10, np.ceil(np.log10(np.max(np.abs(self.multiSineFitresults[axisToPlot, :])))))
        min_val = np.min([minFFT, minSine])
        max_val = np.max([maxFFT, maxSine])

        for j, jdx in enumerate(axisToPlot):
            for i in range(self.numOverTones + 1):
                start = i * (2 * self.numLinesAround + 1)
                stop = (i + 1) * (2 * self.numLinesAround + 1)
                if i == 0:
                    if not phase:
                        firstPlot = ax[j].semilogy(self.multisineFitFreqs[start:stop],
                                                   abs(self.multiSineFitresults[jdx][start:stop]),
                                                   lw=1, label=r'\textbf{Multi-Sinus-Approximation}')
                    else:
                        # Phase plot with Von-Mises enhancement for multi-sine fit
                        multisine_stats = self._calculate_vonmises_phase_stats(
                            np.angle(self.multiSineFitresults[jdx][start:stop]))

                        # Plot mean line and get color
                        firstPlot = ax[j].plot(self.multisineFitFreqs[start:stop],
                                               multisine_stats['mean_phases'],
                                               lw=1, label=r'\textbf{Multi-Sinus-Approximation}', zorder=2)
                        multisine_color = firstPlot[0].get_color()
                        # Plot uncertainty bands with matching color
                        ax[j].fill_between(self.multisineFitFreqs[start:stop],
                                           multisine_stats['lower_bound'], multisine_stats['upper_bound'],
                                           alpha=0.2, color=multisine_color, zorder=1)
                else:
                    if not phase:
                        ax[j].semilogy(self.multisineFitFreqs[start:stop],
                                       abs(self.multiSineFitresults[jdx][start:stop]),
                                       lw=1, color=firstPlot[0].get_color())
                    else:
                        # Phase plot with Von-Mises enhancement for multi-sine fit (continuation)
                        multisine_stats = self._calculate_vonmises_phase_stats(
                            np.angle(self.multiSineFitresults[jdx][start:stop]))

                        # Continue with same color from first plot
                        multisine_color = firstPlot[0].get_color()
                        ax[j].plot(self.multisineFitFreqs[start:stop],
                                   multisine_stats['mean_phases'],
                                   lw=1, color=multisine_color, zorder=2)
                        ax[j].fill_between(self.multisineFitFreqs[start:stop],
                                           multisine_stats['lower_bound'], multisine_stats['upper_bound'],
                                           alpha=0.2, color=multisine_color, zorder=1)

            if not phase:
                bax[j].plot(self.multisineFitFreqs, abs(self.multiSineFitresults[jdx]),
                            lw=1, marker='o', markersize=markerSize)
            else:
                # Enhanced phase plot for broken axes with Von-Mises statistics - line for mean, scatter for raw
                all_multisine_stats = self._calculate_vonmises_phase_stats(np.angle(self.multiSineFitresults[jdx]))
                # Try to get the color from the top plot
                try:
                    # Get the color from the first multi-sine plot line
                    multisine_color = firstPlot[0].get_color()
                except (NameError, IndexError):
                    multisine_color = 'C2'  # Default color if not available

                # Plot mean as line
                bax[j].plot(self.multisineFitFreqs, all_multisine_stats['mean_phases'],
                            color=multisine_color, lw=1, zorder=2)
                # Scatter raw phase values
                bax[j].scatter(self.multisineFitFreqs, np.angle(self.multiSineFitresults[jdx]),
                               color=multisine_color, s=markerSize * 10, alpha=0.7, zorder=3)
                # Add uncertainty bands to broken axes
                bax[j].fill_between(self.multisineFitFreqs, all_multisine_stats['lower_bound'],
                                    all_multisine_stats['upper_bound'],
                                    alpha=0.2, color=multisine_color, zorder=1)

            if plotQoutient and not phase:
                nearestIDX = find_nearest_indices(self.fftFreqs, self.multisineFitFreqs)
                quotients = self.fft[jdx, nearestIDX] / self.multiSineFitresults[jdx, :]
                baxQuatient[j].plot(self.multisineFitFreqs, abs(quotients),
                                    lw=1, marker='o', markersize=markerSize)

        # Plot interpolated results if available
        for i, interPolFactor in enumerate(self.interpolatedFFT.keys()):
            for interpolMethod in self.interpolatedFFT[interPolFactor].keys():
                for j, idx in enumerate(axisToPlot):
                    pointsToPlotLowLeak = int(
                        self.interpolatedFFTFreqsLowLeak[interPolFactor].size / interPolFactor) - 1

                    if not phase:
                        lastNormalPlot = ax[j].semilogy(
                            self.interpolatedFFTFreqsLowLeak[interPolFactor][1:pointsToPlotLowLeak],
                            sp.ndimage.gaussian_filter1d(abs(
                                self.interpolatedFFTLowLeak[interPolFactor][interpolMethod][idx,
                                1:pointsToPlotLowLeak]), filterWidth),
                            label=r'\textbf{DFT ' + str(
                                interPolFactor) + '*' + interpolMethod + ' Interpolation }',
                            alpha=0.5, lw=1, ls=lineSyles[1 + (i % len(lineSyles))][1])

                        lastBrokenPlot = bax[j].semilogy(
                            self.interpolatedFFTFreqsLowLeak[interPolFactor][1:pointsToPlotLowLeak],
                            np.abs(self.interpolatedFFTLowLeak[interPolFactor][interpolMethod][idx,
                                   1:pointsToPlotLowLeak]),
                            label=r'\textbf{DFT ' + str(
                                interPolFactor) + '*' + interpolMethod + ' Interpolation }',
                            alpha=0.5, lw=1, ls=lineSyles[1 + (i % len(lineSyles))][1])

                        if plotHighLeak:
                            pointsToPlot = int(self.interpolatedFFTFreqs[interPolFactor].size / interPolFactor) - 1
                            ax[j].semilogy(
                                self.interpolatedFFTFreqs[interPolFactor][1:pointsToPlot],
                                sp.ndimage.gaussian_filter1d(
                                    np.abs(self.interpolatedFFT[interPolFactor][interpolMethod][idx, 1:pointsToPlot]),
                                    filterWidth),
                                label=r'\textbf{DFT ' + str(
                                    interPolFactor) + '*' + interpolMethod + ' Interpolation}',
                                alpha=0.5, lw=1, ls=lineSyles[1 + (i % len(lineSyles))][1],
                                color=lastNormalPlot[-1].get_color())
                            bax[j].semilogy(
                                self.interpolatedFFTFreqs[interPolFactor][1:pointsToPlot],
                                np.abs(self.interpolatedFFT[interPolFactor][interpolMethod][idx, 1:pointsToPlot]),
                                label=r'\textbf{DFT ' + str(
                                    interPolFactor) + '*' + interpolMethod + ' Interpolation}',
                                alpha=0.5, lw=1, ls=lineSyles[1 + (i % len(lineSyles))][1],
                                color=lastBrokenPlot[0][-1].get_color())
                    else:  # phase plot with Von-Mises enhancement for interpolated data
                        # Calculate Von-Mises statistics for interpolated data
                        interp_stats = self._calculate_vonmises_phase_stats(
                            np.angle(self.interpolatedFFTLowLeak[interPolFactor][interpolMethod][idx,
                                     1:pointsToPlotLowLeak]))

                        # Plot line and get color
                        lastNormalPlot_line = ax[j].plot(
                            self.interpolatedFFTFreqsLowLeak[interPolFactor][1:pointsToPlotLowLeak],
                            interp_stats['mean_phases'],
                            label=r'\textbf{DFT ' + str(interPolFactor) + '*' + interpolMethod + '}',
                            alpha=0.7, lw=1, ls=lineSyles[1 + (i % len(lineSyles))][1], zorder=2)

                        interp_color = lastNormalPlot_line[0].get_color()

                        # Add uncertainty bands with matching color
                        ax[j].fill_between(
                            self.interpolatedFFTFreqsLowLeak[interPolFactor][1:pointsToPlotLowLeak],
                            interp_stats['lower_bound'], interp_stats['upper_bound'],
                            alpha=0.1, color=interp_color, zorder=1)

                        # Line for mean and scatter for raw phases in broken axes (zoom regions)
                        bax[j].plot(
                            self.interpolatedFFTFreqsLowLeak[interPolFactor][1:pointsToPlotLowLeak],
                            interp_stats['mean_phases'],
                            color=interp_color, lw=1, ls=lineSyles[1 + (i % len(lineSyles))][1], zorder=2)
                        # Scatter raw phase values
                        bax[j].scatter(
                            self.interpolatedFFTFreqsLowLeak[interPolFactor][1:pointsToPlotLowLeak],
                            np.angle(self.interpolatedFFTLowLeak[interPolFactor][interpolMethod][idx,
                                     1:pointsToPlotLowLeak]),
                            color=interp_color, s=markerSize * 8, alpha=0.7, zorder=3)
                        # Add uncertainty bands to broken axes
                        bax[j].fill_between(
                            self.interpolatedFFTFreqsLowLeak[interPolFactor][1:pointsToPlotLowLeak],
                            interp_stats['lower_bound'], interp_stats['upper_bound'],
                            alpha=0.1, color=interp_color, zorder=1)

                        if plotHighLeak:
                            pointsToPlot = int(self.interpolatedFFTFreqs[interPolFactor].size / interPolFactor) - 1
                            interp_highleak_stats = self._calculate_vonmises_phase_stats(
                                np.angle(self.interpolatedFFT[interPolFactor][interpolMethod][idx, 1:pointsToPlot]))

                            # Plot high leak line
                            ax[j].plot(
                                self.interpolatedFFTFreqs[interPolFactor][1:pointsToPlot],
                                interp_highleak_stats['mean_phases'],
                                alpha=0.5, lw=1, ls=lineSyles[1 + (i % len(lineSyles))][1],
                                color=interp_color, zorder=2)
                            # Add uncertainty bands
                            ax[j].fill_between(
                                self.interpolatedFFTFreqs[interPolFactor][1:pointsToPlot],
                                interp_highleak_stats['lower_bound'], interp_highleak_stats['upper_bound'],
                                alpha=0.1, color=interp_color, zorder=1)

                            # Line for mean and scatter for raw phases in broken axes
                            bax[j].plot(
                                self.interpolatedFFTFreqs[interPolFactor][1:pointsToPlot],
                                interp_highleak_stats['mean_phases'],
                                alpha=0.5, lw=1, ls=lineSyles[1 + (i % len(lineSyles))][1],
                                color=interp_color, zorder=2)
                            # Scatter raw phase values
                            bax[j].scatter(
                                self.interpolatedFFTFreqs[interPolFactor][1:pointsToPlot],
                                np.angle(self.interpolatedFFT[interPolFactor][interpolMethod][idx, 1:pointsToPlot]),
                                color=interp_color, s=markerSize * 6, alpha=0.5, zorder=3)
                            # Add uncertainty bands to broken axes
                            bax[j].fill_between(
                                self.interpolatedFFTFreqs[interPolFactor][1:pointsToPlot],
                                interp_highleak_stats['lower_bound'], interp_highleak_stats['upper_bound'],
                                alpha=0.1, color=interp_color, zorder=1)

        # Set plot limits and formatting
        for i, idx in enumerate(axisToPlot):
            ax[i].set_xlim([self.fftFreqs[1], self.fftFreqs[-1]])
            ax[i].grid(True, which="major", axis="both", ls="-", lw=PLTSCALFACTOR)
            ax[i].grid(True, which="minor", axis="both", ls="--", lw=0.25 * PLTSCALFACTOR, c='grey')

            if not phase:
                ax[i].set_ylim([min_val, max_val])
                bax[i].set_ylim([min_val, max_val])

                # Add light solid grid lines at all 10^n amplitude levels (decades)
                decade_powers = np.arange(np.floor(np.log10(min_val)), np.ceil(np.log10(max_val)) + 1)
                decade_values = np.power(10, decade_powers)

                for decade in decade_values:
                    if min_val <= decade <= max_val:
                        ax[i].axhline(y=decade, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.7, zorder=1)
            else:
                # Set phase plot limits and ticks
                import numpy as np
                ax[i].set_ylim([-1.05 * np.pi, 1.05 * np.pi])
                ax[i].set_yticks([-np.pi, 0, np.pi])
                ax[i].set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
                bax[i].set_ylim([-1.05 * np.pi, 1.05 * np.pi])

        # Format broken axes
        for i, idx in enumerate(axisToPlot):
            for j, axis in enumerate(bax[i].axs):
                if not phase:
                    axis.set_yscale('log')
                    axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(0.25, 0.5, 0.75, 1.0), numticks=20))

                    # Add light solid grid lines at decade levels for broken axes too
                    decade_powers = np.arange(np.floor(np.log10(min_val)), np.ceil(np.log10(max_val)) + 1)
                    decade_values = np.power(10, decade_powers)

                    for decade in decade_values:
                        if min_val <= decade <= max_val:
                            axis.axhline(y=decade, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.7, zorder=1)
                else:
                    # Set phase plot ticks for broken axes
                    import numpy as np
                    axis.set_yticks([-np.pi, 0, np.pi])
                    if j == 0:  # Only label the first broken axis
                        axis.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
                    else:
                        axis.set_yticklabels([])

                axis.grid(True, which="minor", axis="x", ls=":", lw=0.125 * PLTSCALFACTOR, c='grey')
                axis.grid(True, which="major", axis="x", ls=":", lw=0.25 * PLTSCALFACTOR)
                axis.grid(True, which="minor", axis="y", ls="--", lw=0.25 * PLTSCALFACTOR, c='grey')
                axis.grid(True, which="major", axis="y", ls="-", lw=PLTSCALFACTOR)

                if not phase:
                    axis.set_yticklabels([], minor=True)  # disable minor ticks for all
                    if j != 0:
                        axis.set_yticklabels([])  # disable major ticks for all but the first

                if j != 0:
                    for tick in axis.xaxis.get_minor_ticks():
                        tick.tick1line.set_visible(False)
                        tick.tick2line.set_visible(False)
                        tick.label1.set_visible(False)
                        tick.label2.set_visible(False)

        # Set labels and legends
        for axis in ax:
            axis.legend()
            if not phase:
                axis.set_ylabel(r"\textbf{Amplitude in} $\frac{\text{m}}{\text{s}^2}$")
            else:
                axis.set_ylabel(r"\textbf{Phase in rad}")

        for axis in bax:
            if not phase:
                axis.set_ylabel(r"\textbf{Amplitude in} $\frac{\text{m}}{\text{s}^2}$")
            else:
                axis.set_ylabel(r"\textbf{Phase in rad}")

        if plotQoutient:
            for j, axis in enumerate(baxQuatient):
                axis.grid(True, which="both", axis="both", ls="-")
                axis.set_ylabel(r"\textbf{Amplitude FFT/Fit in A. U.}")
                if j != 0:
                    axis.set_yticklabels([])
                    axis.set_yticklabels([], minor=True)
            baxQuatient[-1].set_xlabel(r"\textbf{Frequenz in Hz }")
        else:
            bax[-1].set_xlabel(r"\textbf{Frequenz in Hz}")

        prefix = ''
        if phase:
            prefix = 'Phase'

        fig.savefig(os.path.join(SAVEFOLDER, prefix + '_' + self.name + ".svg"))
        fig.savefig(os.path.join(SAVEFOLDER, prefix + '_' + self.name + ".pdf"))
        fig.savefig(os.path.join(SAVEFOLDER, prefix + '_' + self.name + ".png"))
        fig.show()

if __name__ == "__main__":
    if LANG=='DE':
        locale.setlocale(locale.LC_ALL, "de_DE.utf8")
    manager = mp.Manager()

    #measurmentFIle=h5py.File(r"/run/media/seeger01/fe4ba5c2-817c-48d5-a013-5db4b37930aa/data/MPU9250PTB_v5(2)(copy).hdf5",'r')
    #measurmentFIle = h5py.File('/run/media/seeger01/fe4ba5c2-817c-48d5-a013-5db4b37930aa/data/MPU9250CEM(2)(copy).hdf5','r')
    leadSensorname='0x1fe40000_MPU_9250'
    pathPrefix = r'/home/seeger01/data/phaseNoise/'
    """
    dataFileEXTREF = h5py.File(os.path.join(pathPrefix, 'extRev_single_GPS_1KHz_Edges.hfd5'), 'r')
    dataFileLSM6DSRX = h5py.File(os.path.join(pathPrefix, 'ST_sensor_test_1667Hz_noTimeGlittCorr.hfd5'), 'r')
    dataFileINTREF = h5py.File(os.path.join(pathPrefix, 'intRev_multi_GPS_1KHz_Edges.hfd5'), 'r')
    """
    dataFileMPU9250 = h5py.File(os.path.join(pathPrefix, 'MPU9250PTB_v5.hdf5'), 'r')
    """
    dataFileBMA280 = h5py.File(os.path.join(pathPrefix, 'BMA280PTB.hdf5'), 'r')
    dataFileLSM6DSRX6667Hz = h5py.File(os.path.join(pathPrefix,'ST_sensor_test_6667Hz_2.hfd5'), 'r')
    dataFileADXL355 = h5py.File(os.path.join(pathPrefix, 'ADXL355_4kHz.hfd5'), 'r')
    """
    """
    jitterGen1 = realWordJitterGen(dataFileINTREF, '0x39f50100_STM32_GPIO_Input',r"\textbf{DAU interner Oszillator}")  # nominalfreq=1000)
    jitterGensForSimulations.append(jitterGen1)
    """
    jitterGenMPU9250 = realWordJitterGen(dataFileMPU9250, '0x1fe40000_MPU_9250',r"\textbf{MPU-9250, IMU, $f_\text{sNom}$ = 1~kHz}")  # $f_s=$ \textbf{1001.0388019191 Hz}")
    jitterGensForSimulations.append(jitterGenMPU9250)
    """
    jitterGenBMA280 = realWordJitterGen(dataFileBMA280, '0x1fe40000_BMA_280',r"\textbf{BMA280, ACC, $f_\text{sNom}$ = 2~kHz}", offset=[int(1.7e6),2048])  # $f_s=$ \textbf{2064.9499858147 Hz} ",)#offset=[100000,1560000+13440562+20])
    jitterGensForSimulations.append(jitterGenBMA280)

    
    jitterGen2 = realWordJitterGen(dataFileINTREF, '0x60ad0100_STM32_GPIO_Input',r"\textbf{Board 2 int. clock}",offset=[0,5000000])#nominalfreq=1000)
    jitterGensForSimulations.append(jitterGen2)

    jitterGen3 = realWordJitterGen(dataFileEXTREF, '0x39f50100_STM32_GPIO_Input',r"\textbf{Board 1 ext. clock")# 1000 Hz}")#nominalfreq=1000)
    jitterGensForSimulations.append(jitterGen3)

    jitterGen4 = realWordJitterGen(dataFileEXTREF, '0x60ad0100_STM32_GPIO_Input',r"\textbf{Board 2 ext. clock}")#nominalfreq=1000)
    jitterGensForSimulations.append(jitterGen4)
    
    jitterGenLSM6DSRX = realWordJitterGen(dataFileLSM6DSRX, '0x60ad0000_LSM6DSRX', r"\textbf{LSM6DSRX $f_s$=1.667~kHz}")
    jitterGensForSimulations.append(jitterGenLSM6DSRX)

    #jitterGenLSM6DSRXPolled2KHz = realWordJitterGen(dataFileLSM6DSRX, '0x60ad0000_LSM6DSRX', r"\textbf{LSM6DSRX polled}", pollingFreq=2000.0)
    #jitterGensForSimulations.append(jitterGenLSM6DSRXPolled2KHz)

    #jitterGenLSM6DSRXLongTerm = realWordJitterGen(dataFileLSM6DSRXlongTerm, '0x60ad0000_LSM6DSRX',r"\textbf{LSM6DSRX long observation time}")
    #jitterGensForSimulations.append(jitterGenLSM6DSRXLongTerm)
    
    jitterGenLSM6DSRX6667Hz = realWordJitterGen(dataFileLSM6DSRX6667Hz, '0x60ad0000_LSM6DSRX',r"\textbf{LSM6DSRX, IMU, $f_\text{sNom}$ = 6,667~kHz}")
    jitterGensForSimulations.append(jitterGenLSM6DSRX6667Hz)

    jitterGenADXL355 = realWordJitterGen(dataFileADXL355, '0x0_ADXL_355',r"\textbf{ADXL355, ACC, $f_\text{sNom}$ = 4~kHz}")  # $f_s=$ \textbf{2064.9499858147 Hz} ",)#offset=[100000,1560000+13440562+20])
    jitterGensForSimulations.append(jitterGenADXL355)

    
    jitterGenMPU9250.plotDeviation(lengthInS=30.0, show=False, correctLinFreqDrift=True, save=True, unit=r'\textmu s',plotInSamplesAxis=True,alpha=1,maxSegments=1,yLims=[-40,40],animate=10)

    jitterGenBMA280.plotDeviation( lengthInS=30.0, show=False, correctLinFreqDrift=True, save=True,unit=r'\textmu s', plotInSamplesAxis=True, alpha=1,maxSegments=1,yLims=[-2500,2500],animate=10)
    
    
    jitterGenMPU9250.plotDeviation(lengthInS=30.0, show=False, correctLinFreqDrift=True, save=True, unit=r'\textmu s',plotInSamplesAxis=True,alpha=1,maxSegments=10,yLims=[-40,40])
    jitterGenBMA280.plotDeviation( lengthInS=30.0, show=False, correctLinFreqDrift=True, save=True,unit=r'\textmu s', plotInSamplesAxis=True, alpha=1,maxSegments=10,yLims=[-2500,2500])
    jitterGenMPU9250.plotDeviation(lengthInS=30.0, show=False, correctLinFreqDrift=True, save=True, unit=r'\textmu s',plotInSamplesAxis=True,alpha=0.33,yLims=[-40,40])
    jitterGenBMA280.plotDeviation( lengthInS=30.0, show=False, correctLinFreqDrift=True, save=True,unit=r'\textmu s', plotInSamplesAxis=True, alpha=0.2,yLims=[-2500,2500])

    # Define a list of argument dictionaries for the different parameter sets
    """
    plot_params = [
        {'lengthInS': 1000, 'lw': 2, 'maxSegments': 1, 'color': tubscolors[0], 'unit': 'ms'},
        {'lengthInS': 30, 'lw': 2, 'maxSegments': 1, 'color': tubscolors[0], 'unit': r'\textmu s'},
        {'lengthInS': 30, 'lw': 2, 'maxSegments': 1, 'color': tubscolors[0], 'unit': r'\textmu s','yLims' : [-15,15]}
    ]
    """
    # Iterate over the list of plot parameters
    for params in plot_params:
        # Create a new figure and axes for each parameter set
        figDviation, axDeviation = jitterGensForSimulations[0].plotDeviation(**params)

        # Iterate over the remaining jitter generators
        for i, jitterGen in enumerate(jitterGensForSimulations[1:], start=1):
            # Adjust color for each plot
            params['color'] = tubscolors[i % len(tubscolors)]
            # Special handling for the last jitter generator (save and show the plot)
            if jitterGen == jitterGensForSimulations[-1]:
                jitterGen.plotDeviation(fig=figDviation, axs=axDeviation, save=True, show=True, **params)
            else:
                # Plot the deviation with the current parameter set
                jitterGen.plotDeviation(fig=figDviation, axs=axDeviation, **params)
    """

    measurmentFIle=dataFileMPU9250
    leadSensorname = '0x1fe40000_MPU_9250'
    def processfitCOmparison(idx):
        sinEX=SineExcitationExperiment(measurmentFIle, idx, sensor=leadSensorname)
        sinEX.plotFFTandSineFit()
        sinEX.plotFFTandSineFit(phase=True)
        snrParams=sinEX.getSNR()
        return snrParams
    snrParams=process_map(processfitCOmparison, np.array([11]), max_workers=1)
    print("DONE")
    """
    WORKER_NUMBER = 12

    show=True
    def plot_graphs(jitterGensForSimulations, plots_params):
        for params in plots_params:
            fig, axs = None, None
            for idx, jitter_gen in enumerate(jitterGensForSimulations):
                # Exclude 'type' parameter when calling the actual plotting function
                specific_params = {k: v for k, v in params.items() if k != 'type' and k != 'plotSincSensForLength' }

                # Handling the fig and axs objects
                if fig is not None and axs is not None:
                    specific_params['fig'] = fig
                    specific_params['axs'] = axs

                # Set show to True in the last iteration
                if idx == len(jitterGensForSimulations) - 1:
                    specific_params['show'] = show
                    specific_params['save'] = True
                    if 'plotSincSensForLength' in params.keys():
                        specific_params['plotSincSensForLength'] = params['plotSincSensForLength']

                if params['type'] == 'deviation':
                    fig, axs = jitter_gen.plotDeviation(**specific_params)
                elif params['type'] == 'phaseNoise':
                    fig, axs = jitter_gen.plotPhaseNoise(**specific_params)
                # Add more conditions here for other types of plots
            plt.close(fig)
            del(fig)
            del(axs)

    #
    for actualjiterGen in jitterGensForSimulations:
        tmp=copy.copy(actualjiterGen.title)
        splitted=tmp.split(r', $f')[0].strip()
        if not tmp==splitted:
            if 'LSM6DSRX' not in splitted:
                actualjiterGen.title=splitted+'}'

    phaseNoiseLW = 2.0
    plots_params = [
        {
            'type': 'deviation',
            'unit': r'\textmu s',
            'lengthInS': 10.0,
            'plotInSamples': False,
            'maxSegments':1
        },
        #{
        #    'type': 'deviation',
        #    'unit': r'\textmu s',
        #    'lengthInS': 10.0,
        #    'maxSegments': 1e4
        #},
        {
            'type': 'deviation',
            'unit': 'ms',
            'lengthInS': 1000.0,
            'plotInSamples': False,
            'maxSegments': 1
        },
        {
            'type': 'deviation',
            'unit': 'ms',
            'lengthInS': 100.0,
            'correctLinFreqDrift': False,
            'plotInSamples': False,
            'maxSegments': 1
        },
        {
            'type': 'deviation',
            'unit': 'ms',
            'lengthInS': 10.0,
            'correctLinFreqDrift': False,
            'plotInSamples': False,
            'maxSegments': 1
        },
        {
            'type': 'phaseNoise',
            'lw':1.0
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'plotSincSensForLength': {'length': [1.0, 10, 100], 'maxFreq': 0.2},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'plotSincSensForLength': {'length': [1.0,10, 30], 'poles':5},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'plotSincSensForLength': {'length': [1.0, 10, 30], 'poles': 5},
            'lw': phaseNoiseLW
        },
        {
        'type': 'phaseNoise',
        'signalFreq': 500,
            'lw':1.0
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [-1.5, 1.5],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [-1.5, 1.5],
            'plotSincSensForLength': {'length':[1.0, 10, 100],'maxFreq':1.5},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [-0.2, 0.2],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'xLims': [-0.2/100, 0.2/100],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [-0.2, 0.2],
            'plotSincSensForLength': {'length':[1.0, 10, 100],'maxFreq':0.2},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [-0.2, 0.2],
            'unit': 'A.U',
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'unit': 'A.U',
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [-0.2, 0.2],
            'unit': 'A.U',
            'plotSincSensForLength': {'length':[1.0, 10, 100],'maxFreq':0.2},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-0.2, 0.2],
            'plotSincSensForLength': {'length': [5.0, 10, 30],'maxFreq': 0.2},
            'lw': phaseNoiseLW
        },
    ]

    plots_params_diss = [
        {
            'type': 'deviation',
            'unit': 'ms',
            'lengthInS': 1000.0,
            'plotInSamples': True,
            'maxSegments': 1
        },
        {
            'type': 'deviation',
            'unit': r'\textmu s',
            'lengthInS': 10.0,
            'plotInSamples': True,
            'maxSegments':1
        },

        {
            'type': 'deviation',
            'unit': 'ms',
            'lengthInS': 10.0,
            'plotInSamples': False,
            'maxSegments': 1,
            'yLims':[-1.0,2.0]
        },
        {
            'type': 'phaseNoise',
            'lw':1.0
        },
        {
            'type': 'phaseNoise',
            'xLims': [-0.002, 0.002],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'plotSincSensForLength': {'length': [1.0, 10, 30], 'poles': 3},
            'yLims':[-140,40],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1, 1],
            'yLims': [-140, 40],
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [0, 0.2],
            'yLims': [0, 1],
            'unit': 'A.U',
            'plotSincSensForLength': {'length': [1.0, 10, 30], 'poles': 3},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 500,
            'xLims': [0, 0.2],
            'yLims': [0, 1],
            'unit': 'A.U',
            'plotSincSensForLength': {'length': [1.0, 10, 30], 'poles': 3},
            'lw': phaseNoiseLW
        },
        {
            'type': 'phaseNoise',
            'signalFreq': 80,
            'xLims': [-1,1],
            'yLims': [0, 1],
            'unit': 'A.U',
            'plotSincSensForLength': {'length': [1.0, 10, 30], 'poles': 3},
            'lw': phaseNoiseLW
        }
    ]
    #plot_graphs(jitterGensForSimulations, plots_params_diss)
    #print("Debug")
    #    def plotDeviation(self, fig=None, axs=None, lengthInS=None, show=False, lw=PLTSCALFACTOR, correctLinFreqDrift=True,plotInSamples=False, save=False, unit='ns', yLims=None, plotInSamplesAxis=False, alpha=1,color=None, maxSegments=1e4):
    """
    figSaveCounter = 70
    """
    freqPoints=1000
    ampPoints=0
    SimuPoints =     ampPoints+len(jitterGensForSimulations)
    nsPreAmpStep=20
    lengthInS=10
    freqs=np.zeros(freqPoints * SimuPoints)
    noiseLevel=np.zeros(freqPoints * SimuPoints)
    runNoiselevel=np.append(np.flip(np.arange(len(jitterGensForSimulations)) - (len(jitterGensForSimulations)-1)), np.array(np.arange(SimuPoints - 2) + 1) * nsPreAmpStep * 10e-9)
    for i in range(SimuPoints):
        tmpFreqs=np.linspace(0.1,1000,freqPoints)
        freqToNear=(tmpFreqs % 1000) < 5
        freqToNear+=(tmpFreqs % 500) < 5
        freqToAdd=10*freqToNear
        tmpFreqs+=freqToAdd
        tmpNoiseLevel=np.ones(freqPoints)*runNoiselevel[i]
        freqs[i*freqPoints:(i+1)*freqPoints]=tmpFreqs
        noiseLevel[i * freqPoints:(i + 1) * freqPoints] = tmpNoiseLevel
    length=np.ones(freqs.size)*lengthInS
    testparams=np.array([freqs,noiseLevel,length]).transpose()
    results=process_map(getmuAndSTdForFreq, testparams, max_workers=WORKER_NUMBER,chunksize=1)
    with closing(Pool()) as p:
        results=p.map(getmuAndSTdForFreq, tqdm.tqdm(testparams))
    results=np.array(results)
    bw=np.ones(SimuPoints)
    """
    """
    def plotMagDeviations(idxs=np.arange(len(jitterGensForSimulations))):
        fig1, ax = plt.subplots(figsize =(24, 8))
        #fig1.set_figwidth(12)
        #fig1.set_figheight(4)
        #if LANG=='EN':
            #fig1.suptitle(r"\textbf{Simulated time = " + str(lengthInS) + ' s, local frequency correction '+str(localFreqqCorr)+'}')
        #if LANG=='DE':
            #fig1.suptitle(r"\textbf{Simulationsdauer = " + str(lengthInS) + ' s, Lokalefrequenzkorrektur '+trueFalseAnAus[localFreqqCorr]+'}')
        doFit=False
        plotErrors=True
        for i in idxs:
            tmpFreqs=freqs[i * freqPoints: (i + 1) * freqPoints]
            if i<=(len(jitterGensForSimulations)-1):
                label = jitterGensForSimulations[i].title
            else:
                label=r"\textbf{\textit{simu.} $2\sigma= " + str(2*((i-1) * nsPreAmpStep)) + "$ ns}"
            AMPS=results[i * freqPoints: (i + 1) * freqPoints,6]
            AMPSErrorBottom = results[i * freqPoints: (i + 1) * freqPoints,4]
            AMPSErrorTop = results[i * freqPoints: (i + 1) * freqPoints, 8]
            AMPSError25Bottom = results[i * freqPoints: (i + 1) * freqPoints,5]
            AMPSError75Top = results[i * freqPoints: (i + 1) * freqPoints, 7]
            coveragenameDict={'EN':'coverage','DE':'Konfidenzinterval'}
            if plotErrors:
                dataPlot=ax.plot(tmpFreqs,
                       AMPS,
                       label=r"\textbf{Median, }"+label,lw=PLTSCALFACTOR*2,color=tubscolors[i])

                errorPlot2 = ax.fill_between(tmpFreqs,
                                   AMPSErrorBottom,
                                    AMPSErrorTop,
                                    #label=r"\textbf{32\% - 68\% "+coveragenameDict[LANG]+" }" + str(label),
                                    color=dataPlot[0].get_color(),
                                    alpha=0.1,
                                    hatch = 'O')
                errorPlot1 = ax.fill_between(tmpFreqs,
                                   AMPSError25Bottom,
                                    AMPSError75Top,
                                    #label=r"\textbf{5\% - 95\% "+coveragenameDict[LANG]+" }"+str(label),
                                    color=dataPlot[0].get_color(),
                                    alpha=0.1,
                                    ls="--",
                                           hatch='o')

            else:
                dataPlot=ax.plot(tmpFreqs,
                       AMPS,
                       label=label)
            if doFit:
                popt, pcov = curve_fit(gaus, tmpFreqs, AMPS, p0=[1, 5e5])
                ax.plot(tmpFreqs,gaus(tmpFreqs,popt[0],popt[1]),label=r"\textbf{Fited bandwidth = "+"{:.2f}".format(abs(popt[1])/1e6)+" MHz }",color=dataPlot[-1].get_color(),ls='--')
                bw[i]=popt[1]
                print('______'+str(i * nsPreAmpStep)+' ns ___________')
                print(popt)
                print(popt[1]/(i * nsPreAmpStep*10e-9)*(i * nsPreAmpStep*10e-9))
                print('_____________________________________________')
        #ax[0].legend()
        #ax[0].legend(ncol=4)
        ax.legend(ncol=2,loc='lower left')
        if LANG=='EN':
            ax.set_xlabel(r"\textbf{Simulated signal frequency in Hz}")
            #ax[0].set_ylabel(r"$2\sigma(\hat{A})$ \textbf{in \%}")
            ax.set_ylabel(r"$\frac{\mathbf{\hat{A}}}{\mathbf{A_{nom}}}$")
        if LANG=='DE':
            ax.set_xlabel(r"\textbf{Signalfrequenz in Hz}")
            #ax[0].set_ylabel(r"$2\sigma(\hat{A})$ \textbf{in \%}")
            ax.set_ylabel(r"\textbf{Magnitude} $\frac{\mathbf{\hat{A}}}{\mathbf{A_{nom}}}$")
        #ax[0].grid(True)
        ax.grid(True)
        #fig1.tight_layout()
        fig1.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' +"Magnitude_degradation_simulation"+ str(lengthInS) + "lockalFreqKoor" +trueFalseAnAus[localFreqqCorr]+'.png'), dpi=300, bbox_inches='tight')
        fig1.savefig(os.path.join(SAVEFOLDER, str(int(globals()['figSaveCounter'])).zfill(2) + '_' +"Magnitude_degradation_simulation"+ str(lengthInS) + "lockalFreqKoor" +trueFalseAnAus[localFreqqCorr]+ '.pdf') , dpi=300, bbox_inches='tight')
        globals()['figSaveCounter'] += 1
        fig1.show()

    plotMagDeviations()
    plotMagDeviations([0])
    plotMagDeviations([0, 1, 3, 4])
    plotMagDeviations([0, 1, 3, 4, 5])

    figSaveCounter = 80
    freqPoints=1000
    ampPoints=0
    SimuPoints =     ampPoints+len(jitterGensForSimulations)
    nsPreAmpStep=20
    lengthInS=100
    freqs=np.zeros(freqPoints * SimuPoints)
    noiseLevel=np.zeros(freqPoints * SimuPoints)
    runNoiselevel=np.append(np.flip(np.arange(len(jitterGensForSimulations)) - (len(jitterGensForSimulations)-1)), np.array(np.arange(SimuPoints - 2) + 1) * nsPreAmpStep * 10e-9)
    for i in range(SimuPoints):
        tmpFreqs=np.linspace(0.1,1000,freqPoints)
        freqToNear=(tmpFreqs % 1000) < 5
        freqToNear+=(tmpFreqs % 500) < 5
        freqToAdd=10*freqToNear
        tmpFreqs+=freqToAdd
        tmpNoiseLevel=np.ones(freqPoints)*runNoiselevel[i]
        freqs[i*freqPoints:(i+1)*freqPoints]=tmpFreqs
        noiseLevel[i * freqPoints:(i + 1) * freqPoints] = tmpNoiseLevel
    length=np.ones(freqs.size)*lengthInS
    testparams=np.array([freqs,noiseLevel,length]).transpose()
    results=process_map(getmuAndSTdForFreq, testparams, max_workers=WORKER_NUMBER,chunksize=1)
    with closing(Pool()) as p:
        results=p.map(getmuAndSTdForFreq, tqdm.tqdm(testparams))
    results=np.array(results)
    bw=np.ones(SimuPoints)
    
    plotMagDeviations()
    plotMagDeviations([0])
    plotMagDeviations([0, 1, 3, 4])
    plotMagDeviations([0, 1, 3, 4, 5])
    
    
    
    #fig, ax = plt.subplots(2,sharex=True)
    fig4, ax4 = plt.subplots(1)
    fig4.set_figwidth(12)
    fig4.set_figheight(4)
    #if LANG=='EN':
    #    fig4.suptitle(r"\textbf{Simulated time = " + str(lengthInS) + ' s local frequency correction '+str(localFreqqCorr)+'}')
    #if LANG=='DE':
    #    fig4.suptitle(r"\textbf{Simulationsdauer = " + str(lengthInS) + ' s, Lokalefrequenzkorrektur '+trueFalseAnAus[localFreqqCorr]+'}')
    for i in range(SimuPoints):
        if i<=(len(jitterGensForSimulations)-1):
            label = jitterGensForSimulations[i].title
        else:
            label=r"\textbf{\textit{simu.} $2\sigma= " + str(2*((i-1) * nsPreAmpStep)) + "$ ns}"
        tmpFreqs = freqs[i * freqPoints: (i + 1) * freqPoints]
        sigmaPhase = results[i * freqPoints: (i + 1) * freqPoints,0]
        dataPlot = ax4.plot(tmpFreqs,
                              2 * sigmaPhase / np.pi * 180,
                              label=label)
        #ax[1].plot(tmpFreqs,
        #           results[i * freqPoints: (i + 1) * freqPoints, 1] / np.pi * 180,
        #           label=label)
        
        #if i != 0:
        #    popt, pcov = curve_fit(logGisticPICaped, tmpFreqs, sigmaPhase, p0=[0.001, bw[i],np.pi])
        #    ax[0].plot(tmpFreqs, logGisticPICaped(tmpFreqs, popt[0], popt[1], popt[2]) / (np.pi * 180),
        #               label=r"\textbf{ Fit Bandbreite = }" + "{:.2f}".format(abs(popt[1]) / 1e6) + " MHz",
        #               color=dataPlot[-1].get_color(), ls='--')
        #    bw[i] = popt[1]
        #    print('______' + str(i * nsPreAmpStep) + ' ns ___________')
        #    print(popt)
        #    print(popt[1] / (i * nsPreAmpStep * 10e-9) * (i * nsPreAmpStep * 10e-9))
        #    print('_____________________________________________')
        
    ax4.legend(ncol=4)
    #ax[1].legend(ncol=4)
    if LANG=='EN':
        ax4.set_xlabel(r"\textbf{Simulated signal frequency in Hz}")
        ax4.set_ylabel(r"\textbf{Max. phase deviation $2\sigma \varphi$ in} $^\circ$")
    if LANG=='DE':
        ax4.set_xlabel(r"\textbf{Simulierte Signalfrequenz in Hz}")
        ax4.set_ylabel(r"\textbf{Phasenabweichung $2\sigma \varphi$ in} $^\circ$")
    #ax[1].set_ylabel(r"$\overline{\varphi}-\varphi_{soll}$ \textbf{in} $^\circ$")
    ax4.grid(True)
    #ax[1].grid(True)
    fig4.tight_layout()
    fig4.show()
    if askforFigPickelSave:
        saveImagePickle("Monte Carlo Amp from PhaseNoise", ax4, fig4)
        saveImagePickle("Monte Carlo Phase from PhaseNoise", ax4, fig4)

    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    surf=ax2.plot_trisurf(freqs, noiseLevel, results[:, 0],cmap=cm.coolwarm)
    fig2.colorbar(surf, shrink=0.5, aspect=5)
    fig2.show()

    
    lengthPoints=15
    StartLength=64
    noiseLevelToUse=100*10e-9
    freqs=np.zeros(freqPoints*lengthPoints)
    noiseLevel=np.zeros(freqPoints*lengthPoints)
    length = np.ones(freqs.size)

    for i in range(lengthPoints):
        tmpFreqs=np.logspace(1.00,7.0,freqPoints)
        freqToNear=(tmpFreqs % 1000) < 5
        freqToNear = (tmpFreqs % 1000) > 995
        freqToNear+=(tmpFreqs % 500) < 5
        freqToNear += (tmpFreqs % 500) > 495
        freqToAdd=10*freqToNear
        tmpFreqs+=freqToAdd
        tmpNoiseLevel=np.ones(freqPoints)*noiseLevelToUse
        freqs[i*freqPoints:(i+1)*freqPoints]=tmpFreqs
        noiseLevel[i * freqPoints:(i + 1) * freqPoints] = tmpNoiseLevel
        length[i * freqPoints:(i + 1) * freqPoints]=StartLength/((i+1)*(i+1))
    testparams=np.array([freqs,noiseLevel,length]).transpose()
    results=process_map(getmuAndSTdForFreq, testparams, max_workers=WORKER_NUMBER,chunksize=10)
    results=np.array(results)


    fig2,ax=plt.subplots(2,sharex=True)
    #fig2.set_figwidth(10)
    #fig2.set_figheight(5)
    #fig2.suptitle(r"\textbf{SampleRate = 1 kHz | 100 ns  \textit{Jitter}}")
    for i in range(lengthPoints):
        ax[0].plot(freqs[i * freqPoints: (i + 1) * freqPoints],
                   2*results[i * freqPoints: (i + 1) * freqPoints,0]/np.pi*180,
                   label=r"\textbf{Dauer= "+"{:.2f}".format(StartLength/((i+1)*(i+1)))+" s}")
        ax[1].plot(freqs[i * freqPoints: (i + 1) * freqPoints],
                   results[i * freqPoints: (i + 1) * freqPoints, 1] / np.pi * 180,
                   label=r"\textbf{Dauer = " + "{:.2f}".format(StartLength/((i+1)*(i+1))) + " s}")
    ax[0].legend(ncol=4)
    ax[1].legend(ncol=4)
    ax[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax[0].set_ylabel(r"$2\sigma(\varphi)$ \textbf{in} $^\circ$")
    ax[1].set_ylabel(r"$\overline{\varphi}-\varphi_{soll}$ \textbf{in} $^\circ$")
    ax[0].grid(True)
    ax[1].grid(True)
    fig2.tight_layout()
    fig2.show()
    
    fig3,ax=plt.subplots(2,sharex=True)
    #fig3.set_figwidth(10)
    #fig3.set_figheight(5)
    #fig3.suptitle(r"\textbf{SampleRate = 1 kHz | 100 ns \textit{Jitter}}")
    for i in range(lengthPoints):
        ax[0].plot(freqs[i * freqPoints: (i + 1) * freqPoints],
                   2*results[i * freqPoints: (i + 1) * freqPoints,2]*100,
                   label=r"\textbf{Dauer= "+"{:.2f}".format(StartLength/((i+1)*(i+1)))+" s}")
        ax[1].plot(freqs[i * freqPoints: (i + 1) * freqPoints],
                   results[i * freqPoints: (i + 1) * freqPoints, 3],
                   label=r"\textbf{Dauer = " + "{:.2f}".format(StartLength/((i+1)*(i+1))) + " s}")
    ax[0].legend(ncol=2)
    ax[1].legend(ncol=2)
    ax[1].set_xlabel(r"\textbf{Frequenz in Hz}")
    ax[0].set_ylabel(r"$2\sigma(\hat{A})$ \textbf{in \%}")
    ax[1].set_ylabel(r"$\frac{\overline{\hat{A}}}{A_{soll}}$ \textbf{in A. U.}")
    ax[0].grid(True)
    ax[1].grid(True)
    fig3.tight_layout()
    fig3.show()

    
    sineESs=[]
    SNRS=[]
    for i in [11]:
        sineESs.append(SineExcitationExperiment(measurmentFIle, i, sensor=leadSensorname))
        sineESs[-1].plotFFTandSineFit()
        SNRS.append(sineESs[-1].getSNR())
    print("Debug")
    """
    print("Hello")

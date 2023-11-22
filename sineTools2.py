# -*- coding: utf-8 -*-
"""

BAHHHH NATSY HACK TO COPY THIS FILE DIRECTLY BUT TIME IS RUNNING OUT
WE WILL CHANGE THIS LATER
I PROMISE
REALLY
TODO DO Sinetoools modularisation
Created on Fri Jun 14 20:36:01 2013
SineTools.py
auxiliary functions related to sine-approximation

@author: Th. Bruns, B.Seeger
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr as sp_lsqr
from scipy.signal import firwin, lfilter
from scipy.stats import linregress
import matplotlib.pyplot as mp
import warnings


def sampletimes(Fs, T):  #
    """
    generate a t_i vector with \n
    sample rate Fs \n
    from 0 to T
    """
    num = int(np.ceil(T * Fs))
    return np.linspace(0, T, num, dtype=np.float64)


# a = displacement, velocity or acceleration amplitude
def sinewave(f, a, phi, ti, offset=0, noise=0, absnoise=0, drift=0, ampdrift=0):
    """
    generate a sampled sine wave s_i = s(t_i) with \n
    amplitude a \n
    initial phase phi \n
    sample times t_i \n
    bias offset (default 0)\n
    noise as multiple of the amplitude in noise level \n
    absnoise as a additive noise component \n
    drift as multiples of amplitude per duration in drifting zero \n
    ampdrift as a drifting amplitude given as multiple of amplitude \n
    """
    Tau = ti[-1] - ti[0]
    n = 0
    n = 0
    if noise != 0:
        n = a * noise * np.random.randn(len(ti))
    if absnoise != 0:
        n = n + absnoise * np.random.randn(len(ti))

    d = drift * a / Tau

    s = (
            a * (1 + ampdrift / Tau * ti) * np.sin(2 * np.pi * f * ti - phi)
            + n
            + d * ti
            + offset
    )
    return s


def fm_counter_sine(
        fm, f, x, phi, ti, offset=0, noise=0, absnoise=0, drift=0, ampdrift=0, lamb=633.0e-9
):
    """
    calculate counter value of heterodyne signal at \n
    carrier freq. fm\n
    x = displacement amplitude \n
    initial phase phi \n
    sample times t_i \n
    bias or offset (default 0)\n
    noise as multiple of the amplitude in noise level \n
    absnoise as a additive noise component \n
    drift as multiples of amplitude per duration in drifting zero \n
    ampdrift as a drifting amplitude given as multiple of amplitude \n
    lamb as wavelength of Laser
    """
    Tau = ti[-1] - ti[0]
    n = 0
    if noise != 0:
        n = x * noise * np.random.randn(len(ti))
    if absnoise != 0:
        n = n + absnoise * np.random.randn(len(ti))

    d = drift * x / Tau

    s = (
            1.0
            / lamb
            * (
                    x * (1 + ampdrift / Tau * ti) * np.sin(2 * np.pi * f * ti - phi)
                    + n
                    + d * ti
                    + offset
            )
    )
    s = np.floor(s + fm * ti)

    return s


# sine fit at known frequency
def threeparsinefit(y, t, f0):
    """
    sine-fit at a known frequency\n
    y vector of sample values \n
    t vector of sample times\n
    f0 known frequency\n
    \n
    returns a vector of coefficients [a,b,c]\n
    for y = a*sin(2*pi*f0*t) + b*cos(2*pi*f0*t) + c
    """
    w0 = 2 * np.pi * f0

    a = np.array([np.cos(w0 * t), np.sin(w0 * t), np.ones(t.size)])

    abc = la.lstsq(a.transpose(), y)
    return abc[0][0:3]  ## fit vector a*sin+b*cos+c


# sine fit at known frequency and detrending
def threeparsinefit_lin(y, t, f0):
    """
    sine-fit with detrending at a known frequency\n
    y vector of sample values \n
    t vector of sample times\n
    f0 known frequency\n
    \n
    returns a vector of coefficients [a,b,c,d]\n
    for y = a*sin(2*pi*f0*t) + b*cos(2*pi*f0*t) + c*t + d
    """
    w0 = 2 * np.pi * f0

    a = np.array([np.cos(w0 * t), np.sin(w0 * t), np.ones(t.size), t, np.ones(t.size)])

    abc = la.lstsq(a.transpose(), y)
    return abc[0][0:4]  ## fit vector


def calc_threeparsine(abc, t, f0):
    """
    return y = abc[0]*sin(2*pi*f0*t) + abc[1]*cos(2*pi*f0*t) + abc[2]
    """
    w0 = 2 * np.pi * f0
    return abc[0] * np.cos(w0 * t) + abc[1] * np.sin(w0 * t) + abc[2]


def amplitude(abc):
    """
    return the amplitude given the coefficients of\n
    y = a*sin(2*pi*f0*t) + b*cos(2*pi*f0*t) + c
    """
    return np.absolute(abc[1] + 1j * abc[0])


def phase(abc, deg=False):
    """
    return the (sine-)phase given the coefficients of\n
    y = a*sin(2*pi*f0*t) + b*cos(2*pi*f0*t) + c \n
    returns angle in rad by default, in degree if deg=True
    """
    return np.angle(abc[1] + 1j * abc[0], deg=deg)


def magnitude(A1, A2):
    """
    return the magnitude of the complex ratio of sines A2/A1\n
    given two sets of coefficients \n
    A1 = [a1,b1,c1]\n
    A2 = [a2,b2,c2]
    """
    return amplitude(A2) / amplitude(A1)


def phase_delay(A1, A2, deg=False):
    """
    return the phase difference of the complex ratio of sines A2/A1\n
    given two sets of coefficients \n
    A1 = [a1,b1,c1]\n
    A2 = [a2,b2,c2]\n
    returns angle in rad by default, in degree if deg=True
    """
    return phase(A2, deg=deg) - phase(A1, deg=deg)


# periodical sinefit at known frequency
def seq_threeparsinefit(y, t, f0, periods=1):
    """
    period-wise sine-fit at a known frequency\n
    y vector of sample values \n
    t vector of sample times\n
    f0 known frequency\n
    periods number of full periods for every fit (default=1)\n
    \n
    returns a (n,3)-matrix of coefficient-triplets [[a,b,c], ...]\n
    for y = a*sin(2*pi*f0*t) + b*cos(2*pi*f0*t) + c
    """
    if y.size < t.size:
        raise ValueError("Dimension mismatch in input data y<t")
    if t.size < y.size:
        warnings.warn(
            "Dimension mismatch in input data y>t. fiting only for t.size values",
            RuntimeWarning,
        )
    Tau = 1.0 / f0 * periods
    dt = np.mean(np.diff(t))
    N = int(Tau // dt)  ## samples per section
    M = int(t.size // N)  ## number of sections or periods

    abc = np.zeros((M, 3))
    for i in range(int(M)):
        ti = t[i * N: (i + 1) * N]
        yi = y[i * N: (i + 1) * N]
        abc[i, :] = threeparsinefit(yi, ti, f0)
    return abc  ## matrix of all fit vectors per period


# four parameter sine-fit (with frequency approximation)
def fourparsinefit(y, t, f0, tol=1.0e-7, nmax=1000):
    """
    y sampled data values \n
    t sample times of y \n
    f0 estimate of sine frequency \n
    tol rel. frequency correction where we stop \n
    nmax maximum number of iterations taken \n
    \n
    returns the vector [a, b, c, w] of  a*sin(w*t)+b*cos(w*t)+c
    """
    abcd = threeparsinefit(y, t, f0)
    w = 2 * np.pi * f0
    err = 1
    i = 0
    while (err > tol) and (i < nmax):
        D = np.array(
            [
                np.cos(w * t),
                np.sin(w * t),
                np.ones(t.size),
                (-1.0) * abcd[0] * t * np.sin(w * t) + abcd[1] * t * np.cos(w * t),
            ]
        )
        abcd = (la.lstsq(D.transpose(), y))[0]
        dw = abcd[3]
        w = w + 0.9 * dw
        i += 1
        err = np.absolute(dw / w)

    assert i < nmax, "iteration error"

    return np.hstack((abcd[0:3], w / (2 * np.pi)))


def calc_fourparsine(abcf, t):
    """
    return y = abc[0]*sin(2*pi*f0*t) + abc[1]*cos(2*pi*f0*t) + abc[2]
    """
    w0 = 2 * np.pi * abcf[3]
    return abcf[0] * np.cos(w0 * t) + abcf[1] * np.sin(w0 * t) + abcf[2]


"""
from octave ...
function abcw = fourParSinefit(data,w0)
  abc = threeParSinefit(data,w0);
  a=abc(1);
  b=abc(2);
  c=abc(3);
  w = w0;

  do 
  D = [sin(w.*data(:,1)) , cos(w.*data(:,1)) , ones(rows(data),1) , a.*data(:,1).*cos(w.*data(:,1)) - b.*data(:,1).*sin(w.*data(:,1)) ];

  s = D \ data(:,2);
  dw = s(4);
  w = w+0.9*dw;
  err = abs(dw/w);

  until (err < 1.0e-8 );

  abcw = [s(1),s(2),s(3),w];

endfunction
"""


# periodical sinefit at known frequency
def seq_fourparsinefit(y, t, f0, tol=1.0e-7, nmax=1000, periods=1, debug_plot=False):
    """
    sliced or period-wise sine-fit at a known frequency\n
    y vector of sample values \n
    t vector of sample times\n
    f0 estimate of excitation frequency\n
    periods integer number of periods used in each slice for fitting
    nmax maximum of iteration to improve f0 \n
    debug_plot Flag for plotting the sequential fit for dubugging \n
     \n
    returns a (n,3)-matrix of coefficient-triplets [[a,b,c], ...]\n
    for y = a*sin(2*pi*f0*t) + b*cos(2*pi*f0*t) + c
    """
    if y.size < t.size:
        raise ValueError("Dimension mismatch in input data y<t")
    if t.size < y.size:
        warnings.warn(
            "Dimension mismatch in input data y>t. fiting only for t.size values",
            RuntimeWarning,
        )
    Tau = 1.0 / f0 * periods
    dt = np.mean(np.diff(t))
    N = int(Tau // dt)  ## samples per section
    M = int(t.size // N)  ## number of sections or periods

    abcd = np.zeros((M, 4))
    for i in range(M):
        ti = t[i * N: (i + 1) * N]
        yi = y[i * N: (i + 1) * N]
        abcd[i, :] = fourparsinefit(yi, ti, f0, tol=tol, nmax=nmax)

    if debug_plot:
        mp.ioff()
        fig = mp.figure("seq_fourparsinefit")
        fig.clear()
        p1 = fig.add_subplot(211)
        p2 = fig.add_subplot(212, sharex=p1)

        for i in range(M):
            p1.plot(t[i * N: (i + 1) * N], y[i * N: (i + 1) * N], ".")
            s = calc_fourparsine(
                abcd[i, :], t[i * N: (i + 1) * N]
            )  # fitted data to plot
            p1.plot(t[i * N: (i + 1) * N], s, "-")
            r = y[i * N: (i + 1) * N] - s  # residuals to plot
            p2.plot(t[i * N: (i + 1) * N], r, ".")
            yi = y[i * N: (i + 1) * N]
        mp.show()

    return abcd  ## matrix of all fit vectors per period


# fitting a pseudo-random multi-sine signal with 2*Nf+1 parameters
def multi_threeparsinefit(y, t, f0):  # f0 vector of frequencies
    """
    fit a time series of a sum of sine-waveforms with a given set of frequencies\n
    y vector of sample values \n
    t vector of sample times\n
    f0 vector of known frequencies\n
    \n
    returns a vector of coefficient-triplets [a,b,c] for the frequencies\n
    for y = sum_i (a_i*sin(2*pi*f0_i*t) + b_i*cos(2*pi*f0_i*t) + c_i
    """
    w0 = 2 * np.pi * f0
    D = np.ones((len(t), 1))  # for the bias
    # set up design matrix
    for w in w0[::-1]:
        D = np.hstack((np.cos(w * t)[:, np.newaxis], np.sin(w * t)[:, np.newaxis], D))

    abc = np.linalg.lstsq(D, y)
    return abc[0]  ## fit vector a*cos+b*sin+c

def multi_complex(abc):  # abc = [a1,b1 , a2,b2, ...,bias]
    """
    return the amplitudes given the coefficients of a multi-sine\n
    abc = [a1,b1 , a2,b2, ...,bias] \n
    y = sum_i (a_i*sin(2*pi*f0_i*t) + b_i*cos(2*pi*f0_i*t) + c_i
    """
    x = abc[:-1][1::2] + 1j * abc[:-1][0::2]  # make complex without Bias (last value)
    return x


def multi_amplitude(abc):  # abc = [a1,b1 , a2,b2, ...,bias]
    """
    return the amplitudes given the coefficients of a multi-sine\n
    abc = [a1,b1 , a2,b2, ...,bias] \n
    y = sum_i (a_i*sin(2*pi*f0_i*t) + b_i*cos(2*pi*f0_i*t) + c_i
    """
    x = abc[:-1][1::2] + 1j * abc[:-1][0::2]  # make complex without Bias (last value)
    return np.absolute(x)


def multi_phase(abc, deg=False):  # abc = [bias, a1,b1 , a2,b2, ...]
    """
    return the initial phases given the coefficients of a multi-sine\n
    abc = [a1,b1 , a2,b2, ...,bias] \n
    y = sum_i (a_i*sin(2*pi*f0_i*t) + b_i*cos(2*pi*f0_i*t) + c_i
    """
    x = abc[:-1][1::2] + 1j * abc[:-1][0::2]  # make complex without Bias (last value)
    return np.angle(x, deg=deg)


def multi_waveform_abc(f, abc, t):
    """
    generate a sample time series of a multi-sine from coefficients and frequencies\n
    f vector of given frequencies \n
    abc = [a1,ba, a2,b2, ..., bias]\n
    t vector of sample times t_i\n
    \n
    returns the vector \n
    y = sum_i (a_i*sin(2*pi*f0_i*t) + b_i*cos(2*pi*f0_i*t) + bias
    """
    ret = 0.0 * t + abc[-1]  # bias
    for fi, a, b in zip(f, abc[0::2], abc[1::2]):
        ret = ret + a * np.cos(2 * np.pi * fi * t) + b * np.sin(2 * np.pi * fi * t)
    return ret


def multi_waveform_mp(f, m, p, t, bias=0, deg=True):
    """
    generate a sample time series of a multi-sine from magnitude/phase and frequencies\n
    f vector of given frequencies \n
    m vector of magnitudes \n
    p vector of phases \n
    t vector of sample times t_i \n
    bias scalar value for total bias
    deg = boolean whether phase is in degree
    \n
    returns the vector \n
    y = sum_i (a_i*sin(2*pi*f0_i*t) + b_i*cos(2*pi*f0_i*t) + bias
    """
    ret = 0.0 * t + bias  # init and bias
    if deg:
        p = np.deg2rad(p)
    for fi, m_i, p_i in zip(f, m, p):
        ret = ret + m_i * np.sin(2 * np.pi * fi * t + p_i)
    return ret


##################################
# Counter based stuff

# periodical sinefit to the linearly increasing heterodyne counter
# version based on Blume
def seq_threeparcounterfit(y, t, f0, diff=False):
    """
    period-wise (single-)sinefit to the linearly increasing heterodyne counter
    version based on "Blume et al. "\n
    y vector of sampled counter values
    t vector of sample times
    f given frequency\n
    \n
    returns (n,3)-matrix of coefficient-triplets [a,b,c] per period \n

    if diff=True use differentiation to remove carrier (c.f. source)
    """
    Tau = 1.0 / f0
    dt = np.mean(np.diff(t))
    N = int(np.floor(Tau / dt))  ## samples per section
    M = int(np.floor(t.size / N))  ## number of sections or periods

    remove_counter_carrier(y, diff=diff)

    abc = np.zeros((M, 4))

    for i in range(int(M)):
        ti = t[i * N: (i + 1) * N]
        yi = y[i * N: (i + 1) * N]

        abc[i, :] = threeparsinefit_lin(yi, ti, f0)
    return abc  ## matrix of all fit vectors per period


def remove_counter_carrier(y, diff=False):
    """
    remove the linear increase in the counter signal
    generated by the carrier frequency of a heterodyne signal\n
    y vector of samples of the signal
    """
    if diff:
        d = np.diff(y)
        d = d - np.mean(d)
        y = np.hstack((0, np.cumsum(d)))
    else:
        slope = y[-1] - y[0]  # slope of linear increment
        y = y - slope * np.linspace(
            0.0, 1.0, len(y), endpoint=False
        )  # removal of linear increment
    return y


# calculate displacement and acceleration to the same analytical s(t)
# Bsp: fm = 2e7, f=10, s0=0.15, phi0=np.pi/3, ti, drift=0.03, ampdrift=0.03,thd=[0,0.02,0,0.004]
def disp_acc_distorted(fm, f, s0, phi0, ti, drift=0, ampdrift=0, thd=0):
    """
    calculate the respective (displacement-) counter and acceleration
    for a parmeterized distorted sine-wave motion in order to compare accelerometry with interferometry \n
    fm is heterodyne carrier frequency (after mixing)\n
    f is mechanical sine frequency (nominal) \n
    phi_0 accelerometer phase delay \n
    ti vector of sample times \n
    drift is displacement zero drift \n
    ampdrift is displacement amplitude druft\n
    thd is vector of higher harmonic amplitudes (c.f. source)
    """
    om = 2 * np.pi * f
    om2 = om ** 2
    tau = ti[-1] - ti[0]
    disp = np.sin(om * ti + phi0)
    if thd != 0:
        i = 2
        for h in thd:
            disp = disp + h * np.sin(i * om * ti + phi0)
            i = i + 1
    if ampdrift != 0:
        disp = disp * (1 + ampdrift / tau * ti)
    if drift != 0:
        disp = disp + s0 * drift / tau * ti
    disp = disp * s0
    disp = np.floor((disp * 2 / 633e-9) + fm * ti)

    acc = -s0 * om2 * (1 + ampdrift / tau * ti) * np.sin(om * ti + phi0)
    if ampdrift != 0:
        acc = acc + (2 * ampdrift * s0 * om * np.cos(om * ti + phi0)) / tau
    if thd != 0:
        i = 2
        for h in thd:
            acc = acc - s0 * h * om2 * (1 + ampdrift / tau * ti) * i ** 2 * np.sin(
                i * om * ti + phi0
            )
            if ampdrift != 0:
                acc = (
                        acc
                        + (2 * ampdrift * s0 * om * i * h * np.cos(om * ti + phi0)) / tau
                )
            i = i + 1

    return disp, acc


###################################
# Generation and adaptation of Parameters of the Multi-Sine considering hardware constraints
def PR_MultiSine_adapt(
        f1,
        Nperiods,
        Nsamples,
        Nf=8,
        fs_min=0,
        fs_max=1e9,
        frange=10,
        log=True,
        phases=None,
        sample_inkr=1,
):
    """
    Returns an additive normalized Multisine time series. \n
    f1 = start frequency (may be adapted by the algorithm) \n
    Nperiods = number of periods of f1 (may be increased by algorithm) \n
    Nsamples = Minimum Number of samples  \n
    Nf = number of frequencies in multi frequency mix \n
    fs_min = minimum sample rate of used device (default 0) \n
    fs_max = maximum sample rate of used device (default 0) \n
    frange = range of frequency as a factor relative to f1 (default 10 = decade) \n
    log = boolean for logarithmic (True, default) or linear (False) frequency scale \n
    phases = float array of given phases for the frequencies (default=None=random) \n
    deg= boolean for return phases in deg (True) or rad (False) \n
    sample_inkr = minimum block of samples to add to a waveform
    \n
    returns: freq,phase,fs,ti,multi \n
    freq= array of frequencies \n
    phase=used phases in deg or rad \n
    T1 = (adapted) duration
    fs=sample rate \n
    """
    if (
            Nsamples // sample_inkr * sample_inkr != Nsamples
    ):  # check multiplicity of sample_inkr
        Nsamples = (
                           Nsamples // sample_inkr + 1
                   ) * sample_inkr  # round to next higher multiple

    T0 = Nperiods / f1  # given duration
    fs0 = Nsamples / T0  # (implicitly) given sample rate

    if False:
        print("0 Nperiods: " + str(Nperiods))
        print("0 Nsamples: " + str(Nsamples))
        print("0 fs: " + str(fs0))
        print("0 T0: " + str(T0))
        print("0 f1: " + str(f1))

    fs = fs0
    if fs0 < fs_min:  # sample rate too low, then set to minimum
        fs = fs_min
        print("sample rate increased")
    elif fs0 > fs_max:  # sample rate too high, set to max-allowed and
        fs = fs_max
        Nperiods = np.ceil(
            Nperiods * fs0 / fs_max
        )  # increase number of periods to get at least Nsamples samples
        T0 = Nperiods / f1
        print("sample rate reduced, Nperiods=" + str(Nperiods))

    Nsamples = T0 * fs
    if (
            Nsamples // sample_inkr * sample_inkr != Nsamples
    ):  # check multiplicity of sample_inkr
        Nsamples = (
                           Nsamples // sample_inkr + 1
                   ) * sample_inkr  # round to next higher multiple

    T1 = Nsamples / fs  # adapt exact duration
    f1 = Nperiods / T1  # adapt f1 for complete cycles
    if False:
        print("Nperiods: " + str(Nperiods))
        print("Nsamples: " + str(Nsamples))
        print("fs: " + str(fs))
        print("T1: " + str(T1))
        print("f1: " + str(f1))

    f_res = 1 / T1  # frequency resolution
    # determine a series of frequencies (freq[])
    if log:
        fact = np.power(frange, 1.0 / (Nf - 1))  # factor for logarithmic scale
        freq = f1 * np.power(fact, np.arange(Nf))
    else:
        step = (frange - 1) * f1 / (Nf - 1)
        freq = np.arange(f1, frange * f1 + step, step)

    # auxiliary function to find the nearest available frequency
    def find_nearest(
            x, possible
    ):  # match the theoretical freqs to the possible periodic freqs
        idx = (np.absolute(possible - x)).argmin()
        return possible[idx]

    fi_pos = np.arange(f1, frange * f1 + f_res, f_res)  # possible periodic frequencies
    f_real = []
    for f in freq:
        f_real.append(find_nearest(f, fi_pos))
    freq = np.hstack(f_real)
    if True:
        print("freq: " + str(freq))

    if phases is None:  # generate random phases
        phase = np.random.randn(Nf) * 2 * np.pi  # random phase
    else:  # use given phases
        phase = phases

    return freq, phase, T1, fs


###################################
# Pseudo-Random-MultiSine for "quick calibration"
def PR_MultiSine(
        f1,
        Nperiods,
        Nsamples,
        Nf=8,
        fs_min=0,
        fs_max=1e9,
        frange=10,
        log=True,
        phases=None,
        deg=False,
        sample_inkr=1,
):
    """
    Returns an additive normalized Multisine time series. \n
    f1 = start frequency (may be adapted) \n
    Nperiods = number of periods of f1 (may be increased) \n
    Nsamples = Minimum Number of samples  \n
    Nf = number of frequencies in multi frequency mix \n
    fs_min = minimum sample rate of used device (default 0) \n
    fs_max = maximum sample rate of used device (default 0) \n
    frange = range of frequency as a factor relative to f1 (default 10 = decade) \n
    log = boolean for logarithmic (True, default) or linear (False) frequency scale \n
    phases = float array of given phases for the frequencies (default=None=random) \n
    deg= boolean for return phases in deg (True) or rad (False) \n
    sample_inkr = minimum block of samples to add to a waveform
    \n
    returns: freq,phase,fs,ti,multi \n
    freq= array of frequencies \n
    phase=used phases in deg or rad \n
    fs=sample rate \n
    ti=timestamps \n
    multi=array of time series values \n
    """

    freq, phase, T1, fs = PR_MultiSine_adapt(
        f1,
        Nperiods,
        Nsamples,
        Nf=Nf,
        fs_min=fs_min,
        fs_max=fs_max,
        frange=frange,
        log=log,
        phases=phases,
        sample_inkr=sample_inkr,
    )

    if deg:  # rad -> deg
        phase = phase * np.pi / 180.0

    ti = np.arange(T1 * fs, dtype=np.float32) / fs

    multi = np.zeros(len(ti), dtype=np.float64)
    for f, p in zip(freq, phase):
        multi = multi + np.sin(2 * np.pi * f * ti + p)

    multi = multi / np.amax(np.absolute(multi))  # normalize

    if False:
        import matplotlib.pyplot as mp

        fig = mp.figure(1)
        fig.clear()
        pl1 = fig.add_subplot(211)
        pl2 = fig.add_subplot(212)
        pl1.plot(ti, multi, "-o")
        pl2.plot(np.hstack((ti, ti + ti[-1] + ti[1])), np.hstack((multi, multi)), "-o")
        mp.show()

    return (
        freq,
        phase,
        fs,
        multi,
    )  # frequency series, sample rate, sample timestamps, waveform


def seq_multi_threeparam_Dmatrix_old(f, t, periods=1, progressive=True):
    """
    Fit a multi-sinus-signal in slices in one go.

    Parameters
    ----------
    f : numpy.array of floats
        list of frequencies in the signal
    t : numpy.array of floats
        timestamps of y (typically seconds)
    periods : float, optional
        the number of periods of each frequncy used for each fit. The default is 1.

    Returns
    -------

    fr,  : 1-d numpy.array of floats
          frequencies related to slices
    D    : 2-d numpy.array of floats
          Design matrix for the fit

    """
    T = t[-1] - t[0]
    col = 0
    data = np.array([])
    ci = []
    ri = []
    fr = []
    # Designmatrix for cos/sin
    for fi, omi in zip(f, 2 * np.pi * f):
        Nri = 0  # counter for the current row index
        if progressive:
            tau = np.ceil(f[0] / fi * periods) / fi  # approximately same abs. slice length for all fi
        else:
            tau = 1 / fi * periods  # slice length in seconds for periods of fi
        t_sl = np.array_split(t, np.ceil(T / tau))  # array of slices of sample times
        fr = fr + [fi] * len(t_sl)  # len(t_sl) times frequency fi in Design matrix
        for ti in t_sl:  # loop over slices for one frequency
            # cosine part
            cos = np.cos(omi * ti)
            data = np.hstack((data, cos))  # data vector
            ci = ci + [col] * len(ti)  # column index vector
            col = col + 1
            ri = ri + [Nri + i for i in range(len(ti))]  # row index vector

            # sine part
            sin = np.sin(omi * ti)
            data = np.hstack((data, sin))  # data vector
            ci = ci + [col] * len(ti)  # column index vector
            col = col + 1
            ri = ri + [Nri + i for i in range(len(ti))]  # row index vector

            Nri = Nri + len(ti)

    # Bias part
    data = np.hstack((data, np.ones((len(t)))))
    ci = ci = ci + [col] * len(t)
    ri = ri + [i for i in range(len(t))]
    col = col + 1

    # build sparse matrix, init as coo, map to csr
    D = coo_matrix((data, (ri, ci)), shape=(len(t), col)).tocsr()

    return np.array(fr), D


def seq_multi_threeparam_Dmatrix(f, t, periods=1, progressive=True):
    """
    Fit a multi-sinus-signal in slices in one go.

    Parameters
    ----------
    f : numpy.array of floats
        list of frequencies in the signal
    t : numpy.array of floats
        timestamps of y (typically seconds)
    periods : float, optional
        the number of periods of each frequncy used for each fit. The default is 1.

    Returns
    -------

    fr,  : 1-d numpy.array of floats
          frequencies related to slices
    D    : 2-d numpy.array of floats
          Design matrix for the fit

    """
    print("## Warning: parameter 'progressive' is ignored in this version of seq_multi_fourparam_Dmatrix")

    Nt = len(t)
    Nf = len(f)

    fr = []
    tau = periods / f[0]
    N_samp = int(tau // np.mean(np.diff(t)))  # samples(rows) per slice
    N_slices = len(t) // N_samp  # number of slices
    slic_inds = np.array_split(np.arange(len(t)), N_slices)  # vectors of r-indices within the slices

    data = np.zeros((Nt * (2 * Nf + 1)), dtype=np.float)  # initialise matrix values
    ri = np.zeros((Nt * (2 * Nf + 1)), dtype=np.uint)  # initialise row indices
    ci = np.zeros((Nt * (2 * Nf + 1)), dtype=np.uint)  # initialise colum indices

    r0 = 0  # row in rectangular matrix
    c0 = 0  # column in rectangular matrix
    ind0 = 0  # index in sparse matrix
    for slice in slic_inds:
        Ns = len(slice)
        data[ind0:ind0 + Ns] = 1.0  # bias part for slice i
        ri[ind0:ind0 + Ns] = r0 + np.arange(Ns, dtype=np.uint)
        ci[ind0:ind0 + Ns] = c0
        ind0 += Ns  # proceed Ns entries in sparse matrix
        fr.extend([0])
        c0 += 1  # proceed to next column
        for j, fi in enumerate(f):
            om_t = 2 * np.pi * fi * t[slice]  # angular frequency x timestamp
            # sine matrix entries for freq fi in slice i
            data[ind0 + 2 * j * Ns:ind0 + (2 * j + 1) * Ns] = np.sin(om_t)
            ri[ind0 + 2 * j * Ns:ind0 + (2 * j + 1) * Ns] = r0 + np.arange(Ns, dtype=np.uint)
            ci[ind0 + 2 * j * Ns:ind0 + (2 * j + 1) * Ns] = c0 + 2 * j
            # cosine matrix entries for freq fi in slice i
            data[ind0 + (2 * j + 1) * Ns:ind0 + (2 * j + 2) * Ns] = np.cos(om_t)
            ri[ind0 + (2 * j + 1) * Ns:ind0 + (2 * j + 2) * Ns] = r0 + np.arange(Ns, dtype=np.uint)
            ci[ind0 + (2 * j + 1) * Ns:ind0 + (2 * j + 2) * Ns] = c0 + 2 * j + 1
            # build the vector of frequencies
            fr.extend([fi, fi])

        ind0 += 2 * Ns * Nf
        c0 += 2 * Nf  # jump 2*Nf columns to next slice-block
        r0 += Ns  # jump Ns rows to next slice-block
    # build sparse matrix, init as coo, map to csr
    print((len(data), len(ri), len(ci)))
    D = coo_matrix((data, (ri, ci)), shape=(Nt, N_slices * (2 * Nf + 1))).tocsr()
    if False:
        print("seq_multi_threeparam_Dmatrix/ D.shape= %s" % str(D.shape))
        mp.spy(D, markersize=1, marker=".", aspect="auto")
        mp.show()
    return np.array(fr), D


def seq_multi_threeparsinefit(f, y, t, periods=1, D_fr=None, abc0=None, progressive=True):
    """
    performs a simultanius, sliced three-parameter fit on a multisine signal y

    Parameters
    ----------
    f : 1-d numpy.array of floats
        frequencies in the signal
    y : 1-d numpy.array
        samples of the signal
    t : 1-d numpy.array
        vector of sample times
    periods : float
        (fractional) number of periods in one slice. The default is 1.

    Returns
    -------
    f_ab_c : 2-d numpy.array of floats
        frequencies, a/b-coefficients and bias related to given frequencies [f1,f1,f1,f2,f2, ...fn, fn, f=0=bias]
    y0 : 1-d numpy.array of floats
        samples of the optimal fit
    resid : 1-d numpy.array of floats
        Residuals = Difference =(y-y0)

    """
    if D_fr is None:
        fr, D = seq_multi_threeparam_Dmatrix(f, t, periods,
                                             progressive=progressive)  # calculate the design matrix (as sparse matrix)
    else:
        D = D_fr[0]  # vector of sparse design matrix
        fr = D_fr[1]  # vector of frequencies

    abc = sp_lsqr(D, y, x0=abc0, atol=1.0e-9, btol=1.0e-9)
    y0 = D.dot(abc[0])

    # print(abc[0])
    N_col = (2 * len(f) + 1)  # length of a row c,a,b,a,b,..,a,b
    N_slices = int(len(abc[0]) / N_col)  # number of slices = number of rows
    f_c_ab = [0] + list(np.array([[fi, fi] for fi in f]).flatten()) + list(abc[0])  # concatenation of arrays
    f_c_ab = np.array(f_c_ab).reshape((N_slices + 1, N_col))

    # print(f_ab_c)
    return f_c_ab, y0, y - y0  # coefficients, fit signal, residuals


def seq_multi_fourparam_Dmatrix(f, t, delta, abc=None, periods=1, progressive=True):
    """
    Fit a multi-sinus-signal in slices in one go.

    Parameters
    ----------
    f : numpy.array of floats
        list of (adapted) frequencies in the signal
    t : numpy.array of floats
        timestamps of y (typically seconds)
    delta: float (scalar)
         common stretch factor for the time-scale or shrink for the frequency
    abc :numpy.array of floats
         parameters from last fit-iteration
    periods : float, optional
        the number of periods of each frequncy used for each fit. The default is 1.

    Returns
    -------

    fr,  : 1-d numpy.array of floats
          frequencies related to slices
    D    : 2-d numpy.array of floats
          Design matrix for the fit

    """
    Nt = len(t)
    Nf = len(f)

    ci = []  # initialise column indices
    ri = [a for a in range(Nt)] * (2 * Nf + 2)  # row indices
    data = [0.0] * len(ri)  # initialise data
    fr = []
    col_a = []  # extra part -A*t*sin(om t)
    col_b = []  # extra part  B*t*cos(om t)
    last_col = np.zeros((Nt), dtype=np.float64)

    # Designmatrix for cos/sin
    c_count = 0  # counter for columns
    f_count = 0  # counter for frequencies * 2
    for fi, omi in zip(f, 2 * np.pi * f):
        col_a = []  # extra part -A*t*sin(om t)
        col_b = []  # extra part  B*t*cos(om t)
        om_t = omi * t * delta  # omega*t
        co = np.cos(om_t)
        si = np.sin(om_t)
        data[f_count * Nt:(f_count + 1) * Nt] = co.tolist()  # sparse matrix entries
        data[(f_count + 1) * Nt:(f_count + 2) * Nt] = si.tolist()  # sparse matrix entries

        # now split the vectors over the sparse matrix
        if progressive:
            tau = 1 / fi * (fi // f[0]) * periods  # approximately same abs. slice length for all fi
        else:
            tau = 1 / fi * periods  # slice length in seconds for periods of fi
        N_samp = int(tau // np.mean(np.diff(t)))  # samples/rows per slice
        N_slices = len(t) // N_samp  # slices for

        slic_inds = np.array_split(np.arange(len(t)), N_slices)  # r-indices in the slices
        if len(slic_inds[-1] < N_samp):  # if last slice too short
            slic_inds[-2] = np.concatenate([slic_inds[-2], slic_inds[-1]])  # merge last two slices
            slic_inds.pop()  # remove old last slice

        fr = fr + [fi] * len(slic_inds)
        # for cosine
        for i, s in enumerate(slic_inds):
            ci = ci + [c_count + 2 * i] * len(s)
        # for sine
        for i, s in enumerate(slic_inds):
            ci = ci + [c_count + (2 * i + 1)] * len(s)

        if abc is not None:
            for i, s in enumerate(slic_inds):
                col_a = col_a + [-abc[2 * i]] * len(s)  # list of A coeffs of cos
                col_b = col_b + [abc[2 * i + 1]] * len(s)  # list of B coeffs of sine
            last_col = last_col + om_t * (np.array(col_b) * co + np.array(col_a) * si)  # add up 4 param column
        # else last column is zero as initialised

        c_count += 2 * len(slic_inds)
        f_count += 2

    # Bias part
    data[2 * Nf * Nt:(2 * Nf + 1) * Nt] = [1.0] * Nt
    ci[2 * Nf * Nt:(2 * Nf + 1) * Nt] = [c_count] * Nt
    c_count += 1

    # 4th parameter part
    data[(2 * Nf + 1) * Nt:(2 * Nf + 2) * Nt] = last_col.tolist()
    ci[(2 * Nf + 1) * Nt:(2 * Nf + 2) * Nt] = [c_count] * Nt
    c_count += 1

    # build sparse matrix, init as coo, map to csr
    D = coo_matrix((data, (ri, ci)), shape=(Nt, c_count)).tocsr()
    if False:
        print("seq_multi_threeparam_Dmatrix/ D.shape= %s" % str(D.shape))
        mp.spy(D, markersize=5, marker=".", aspect="auto")
        mp.show()
    return np.array(fr), D


def seq_multi_fourparam_sineFit(f, y, t, periods=1, progressive=True, tol=1.0e-9, n_max=100):
    """
    Fit a multi-sinus-signal in slices in one go. With a frequency correction
    universal for all frequncies (time-stretch)

    Parameters
    ----------
    f : numpy.array of floats
        list of frequencies in the signal
    t : numpy.array of floats
        timestamps of y (typically seconds)
    periods : float, optional
        the number of periods of each frequncy used for each fit. The default is 1.
    tol : float, optional
        frequency correction factor (1-tol) when convergence is assumed default 1.0e-6
    n_max : uint, optional
        maximum number of iterations performed.

    Returns
    -------

    fr,  : 1-d numpy.array of floats
          frequencies related to slices
    D    : 2-d numpy.array of floats
          Design matrix for the fit

    """

    fr, D = seq_multi_threeparam_Dmatrix(f, t, periods=periods, progressive=progressive)
    f_abc, y1, res1 = seq_multi_threeparsinefit(f, y, t, D_fr=[D, fr], periods=periods, progressive=progressive)

    abc = (f_abc[0:-1, 1:3]).flatten()  # a,b,a,b,a,b,
    abc = np.hstack([abc, f_abc[-2, -2], 0.0])  # ababab, c, dw

    delta = 1.0  # frequency/time stretch factor
    i = 0

    while True:  # do-while emulation
        print("%d. call to solver" % (i + 1))
        fr, D = seq_multi_fourparam_Dmatrix(f, t, delta, abc=abc, periods=periods,
                                            progressive=progressive)  # calculate the design matrix (as sparse matrix)
        param = sp_lsqr(D, y, x0=abc, atol=1.0e-10, btol=1.0e-10)
        abc = param[0]
        delta = delta * (1 + abc[-1])  # korrekt the timestamps by the factor 1 over \Delta\omega
        print("\delta\Omega: %g" % abc[-1])
        print("delta= %s" % str(delta))
        print("corrected f: %s" % str(delta * f))

        if (np.abs(abc[-1]) < tol):
            print("fit arrived at tolerance after %d iterations" % (i + 1))
            print("last correction: %g" % abc[-1])
            break

        if (i > n_max):
            print("too many iterations")
            break
        i += 1

    y0 = D.dot(param[0])
    fr = fr * delta

    f_ab_c = []
    k = 0
    for fi in fr:  # compile a list of frequencies and coefficients
        f_ab_c = f_ab_c + [fi, abc[k], abc[k + 1]]  # [f, a, b]...
        k = k + 2
    f_ab_c = f_ab_c + [0.0, abc[-2], 0.0]  # add the bias to the list [0,c,0]
    f_ab_c = np.array(f_ab_c).reshape((len(f_ab_c) // 3, 3))

    return f_ab_c, y0, y - y0  # coefficients, fit signal, residuals


def seq_multi_amplitude(f_ab_c):
    """
    return the amplitude(s) of a sequentially fit multi-sine signal,
    i.e. amplitudes encoded in the return value of seq_multi_threeparsinefit.

    Parameters
    ----------
    f_ab_c : 2-d numpy array of floats (Nx3)
        f,a,b in a row for several rows, as returned by seq_multi_threeparsinefit.

    Returns
    -------
    2d-numpy-array of floats (Nx2)
        frequency and associated amplitude.

    """
    # print("Test")
    f = f_ab_c[0, 1::2]  # vector of frequencies
    f_vec = []
    amp = []
    for i, fi in enumerate(f):
        a = f_ab_c[1:, 1 + 2 * i]  # sine coefficients
        b = f_ab_c[1:, 2 + 2 * i]  # cosine coefficient
        f_vec.extend([fi] * len(a))
        amp.extend(np.abs(1j * a + b))
    return np.column_stack((np.array(f_vec), np.array(amp)))


def seq_multi_phase(f_ab_c, deg=True):
    """
    Calculates the initial phase of a sequentially fit multi-sine signal,
    i.e. initial phases encoded in the return value of seq_multi_threeparsinefit.
    result is either in degrees (deg=True) or rad (deg=False).

    Parameters
    ----------
    f_ab_c : 2-d numpy array of floats (Nx3)
        f,a,b in a row for several rows, as returned by seq_multi_threeparsinefit.
        x=a*cos+b*sin+c
    deg : Boolean, optional
        Flag whether result is in degrees or rad. The default is True (Degrees).

    Returns
    -------
    2d-numpy-array of floats (Nx2)
        frequency and associated initial phase.

    """
    f = f_ab_c[0, 1::2]  # vector of frequencies
    f_vec = []
    amp = []
    for i, fi in enumerate(f):
        a = f_ab_c[1:, 1 + 2 * i]  # cosine coefficients
        b = f_ab_c[1:, 2 + 2 * i]  # sine coefficient
        f_vec.extend([fi] * len(a))
        amp.extend(np.angle(1j * b + a, deg=deg))
    return np.column_stack((np.array(f_vec), np.array(amp)))


def seq_multi_bias(f_ab_c):
    """
    Returns the single bias of a sequentially fit multi-sine signal,
    i.e. bias encoded in the return value of seq_multi_threeparsinefit.

    Parameters
    ----------
    f_ab_c : 2-d numpy array of floats (Nx3)
        f,a,b in a row for several rows, as returned by seq_multi_threeparsinefit.

    Returns
    -------
    float
        Bias of the signal.

    """

    return f_ab_c[1:, 0]


def seq_multi_fscale(f_ab_c, periods=0):
    """
    calculate a frequency correction (scaling) based on linear phase drift
    in sequential multisine fitting results for phase delay.

    Parameters
    ----------
    f_ab_c : 2-d numpy array of floats (Nx3)
        f,a,b in a row for several rows, as returned by seq_multi_threeparsinefit.

    Returns
    -------
    None.

    """
    assert periods > 0, "SineTools.seq_multi_fscale: assertion failed, periods > 0 strictly required"

    f_phi = seq_multi_phase(f_ab_c, deg=True)  # frequencies and phases

    f = f_phi[np.where(f_phi[:, 0] == np.amin(f_phi[:, 0])), 0][0]  # take vector of lowest frequencies
    phi = f_phi[np.where(f_phi[:, 0] == np.amin(f_phi[:, 0])), 1][0]  # take the phases for those

    t = np.linspace(0.0, len(f), num=len(f), endpoint=False) / f[0] * periods  # time incremented per n-periods
    dphi = np.diff(phi)
    dphi[np.where(np.abs(dphi) >= 180)] = dphi[np.where(np.abs(dphi) >= 180)] - 360 * np.sign(
        dphi[np.where(np.abs(dphi) >= 180)])  # unwrap
    phi = np.hstack(([0.0], np.cumsum(dphi)))  # accumulated phase shift per n-periods over time

    print("t , phi")
    print(np.vstack((t, phi)).T)
    slope = np.polyfit(t, phi, 1)[0] / f[0]  # slope of the normalized phase over time straight line per period of f_0

    return (1 + slope / 360.0)  # phase was in deg so normalize to period


def atan_demodulate(samples, fc=0.0, fs=0.0, fc_correct=True, lamb=1.0):
    """
    Arc-tangent demodulation of a frequency modulated signal (heterodyne Interferometer raw data)
    with auto-estimate of carrier frequency and carrier frequency adjustment if needed

    Parameters
    ----------
    samples : numpyp.array of floats
        raw data samples of the FM-signal
    fc : float, optional
        if fc>0 given it is the assumed carrier frequency for the (first) demodulation.
        The default is 0.0 which means auto-estimation.
    fs : TYPE, (not really) optional
        sample rate of the measured samples. The default is 0.0.
    fc_correct : Boolean, optional
        if True an extra adjustment of the carrier frequency is performed.
        The default is True.
    lamb : float, optional
        The wavelength of the interferometer which is used to scale phase to
        displacement. The default is 1.0, which returns the raw phase.

    Returns
    -------
    ti   : numpyp.array of floats
        timestamps of results
    phase : numpyp.array of floats
        phase or displacement i.e. demodulation results
    fc    : float
        the carrier frequency ultimately used for the result..

    """
    assert fs > 0, "sample rate fs must be given with fs>0"
    assert fc >= 0, "carrier frequency fc must be zero (auto-mode) or >0"

    def filter_fir(data, fc, fs, order):
        # auxilliary filter function
        nyq = 0.5 * fs
        normal_cutoff = fc / nyq
        b = firwin(order, normal_cutoff, width=None, window='hamming', pass_zero=True, scale=True, nyq=1.0)
        y = lfilter(b, [1], data)
        return y

    def filter_fir2(data, fc, fs, order):
        # auxilliarry filter function (bi-directional, forward-backward)
        l = 4 * order
        data = np.hstack((np.flipud(data[:l]), data))
        y = filter_fir(data, fc, fs, order)
        y = np.flipud(y[l:])
        y = np.hstack((np.flipud(y[:l]), y))
        y = filter_fir(y, fc, fs, order)
        y = np.flipud(y[l:])
        return y

    if fc == 0:  # estimate the carrier frequency from the signal
        pm = np.sign(samples)  # +/- 1
        zer = len(np.where(np.diff(pm) < 0)[0])  # number of falling edge zero crossings
        fc_est = zer / len(samples) * fs  # estimate of carrier frequency
    else:
        fc_est = fc  # take the given carrier frequency

    om_fc = 2 * np.pi * fc_est  # angular frequency for sine/cosine
    ti = np.arange(samples.shape[0]) / fs  # calculate a time_scale
    si = filter_fir2(samples * np.sin(ti * om_fc), 0.8 * fc_est, fs, 11)  # sine-quadrature
    co = filter_fir2(samples * np.cos(ti * om_fc), 0.8 * fc_est, fs, 11)  # cosine-quadrature
    dis = np.arctan2(si, co)  # retreive wrapped phase

    phase = np.unwrap(dis)  # unwrap phase

    if fc_correct:
        delta_f = linregress(ti, phase).slope / (2 * np.pi)
        fc_est -= delta_f
        om_fc = 2 * np.pi * fc_est
        ti = np.arange(samples.shape[0]) / fs  # calculate a time_scale
        si = filter_fir2(samples * np.sin(ti * om_fc), 0.8 * fc_est, fs, 11)  # sine-quadrature
        co = filter_fir2(samples * np.cos(ti * om_fc), 0.8 * fc_est, fs, 11)  # cosine-quadrature
        dis = np.arctan2(si, co)  # wrapped phase
        phase = np.unwrap(dis)

    ti = 1 / fs * np.arange(dis.shape[0])  #

    if lamb != 1.0:  # calculate displacement if wavelength given
        phase = lamb * phase / (4 * np.pi)

    return ti, phase, fc_est


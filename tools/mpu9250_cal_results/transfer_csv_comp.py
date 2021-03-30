import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

scalefactor = 5
MARKER_SIZE = 4 * scalefactor
SMALL_SIZE = 8 * scalefactor
MEDIUM_SIZE = 10 * scalefactor
BIGGER_SIZE = 12 * scalefactor

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)
plt.rc("lines", linewidth=scalefactor)
plt.rc("text", usetex=True)

PTB = pd.read_csv("MPU9250_PTB.csv")
CEM = pd.read_csv("MPU9250_CEM.csv")
CEM = CEM.drop(
    [
        0,
        1,
        19,
        20,
        38,
        39,
        57,
        58,
        76,
        77,
        95,
        96,
        114,
        115,
        133,
        134,
        152,
        153,
    ]
)

# mittelwerte berechnen
FREQPOINTS = 17  # number of frequency test points
magmeans = np.zeros(FREQPOINTS)
phasemeans = np.zeros(FREQPOINTS)
for i in range(FREQPOINTS):
    mags = np.array(CEM["mag"][i::FREQPOINTS], PTB["mag"][i::FREQPOINTS]).ravel()
    phases = np.array(CEM["phase"][i::FREQPOINTS], PTB["phase"][i::FREQPOINTS]).ravel()
    magmeans[i] = np.mean(mags)
    phasemeans[i] = np.mean(phases)
PTBmagmean = np.array([magmeans] * 10).ravel()
PTBPhasemean = np.array([phasemeans] * 10).ravel()
CEMmagmean = np.array([magmeans] * 9).ravel()
CEMPhasemean = np.array([phasemeans] * 9).ravel()


fig, ax = plt.subplots()
ax.errorbar(PTB["freqs"], PTB["mag"], yerr=PTB["maguncer"], label="PTB", fmt="o")
ax.errorbar(CEM["freqs"], CEM["mag"], yerr=CEM["maguncer"], label="CEM", fmt="o")
ax.set(xlabel=r"Frequency $f$ in Hz", ylabel=r" $|S(f)|$")
ax.legend()
fig, ax = plt.subplots(2, 1)
ax[0].errorbar(
    PTB["freqs"],
    PTB["mag"],
    yerr=PTB["maguncer"],
    label="PTB",
    fmt="o",
    markersize=MARKER_SIZE,
)
ax[0].errorbar(
    CEM["freqs"],
    CEM["mag"],
    yerr=CEM["maguncer"],
    label="CEM",
    fmt="o",
    markersize=MARKER_SIZE,
)
ax[0].set(xlabel=r"Frequency $f$ in Hz", ylabel=r" $|S(f)|$")
ax[0].legend()
ax[0].grid()
ax[1].errorbar(
    PTB["freqs"],
    PTB["mag"] - PTBmagmean,
    yerr=PTB["maguncer"],
    label="PTB",
    fmt="o",
    markersize=MARKER_SIZE,
)
ax[1].errorbar(
    CEM["freqs"],
    CEM["mag"] - CEMmagmean,
    yerr=CEM["maguncer"],
    label="CEM",
    fmt="o",
    markersize=MARKER_SIZE,
)
ax[1].set(xlabel=r"Frequency $f$ in Hz", ylabel=r" $|S(f)|-\overline{|S(f)|}$")
ax[1].legend()
ax[1].grid()
fig.suptitle("Magnitude response")
fig.show()


fig, ax = plt.subplots()
ax.errorbar(
    PTB["freqs"], PTB["mag"] - PTBmagmean, yerr=PTB["maguncer"], label="PTB", fmt="o"
)
ax.errorbar(
    CEM["freqs"], CEM["mag"] - CEMmagmean, yerr=CEM["maguncer"], label="CEM", fmt="o"
)
ax.set(xlabel=r"Frequency $f$ in Hz", ylabel=r" $|S(f)|-\overline{|S(f)|}$")
ax.legend()
fig, ax = plt.subplots(2, 1)
ax[0].errorbar(
    PTB["freqs"],
    PTB["phase"] / np.pi * 180,
    yerr=PTB["phaseuncer"] / np.pi * 180,
    label="PTB",
    fmt="o",
    markersize=MARKER_SIZE,
)
ax[0].errorbar(
    CEM["freqs"],
    (CEM["phase"] - np.pi) / np.pi * 180,
    yerr=CEM["phaseuncer"] / np.pi * 180,
    label="CEM",
    fmt="o",
    markersize=MARKER_SIZE,
)
ax[0].set(xlabel=r"Frequency $f$ in Hz", ylabel=r"$\Delta\varphi(f)$ in $^\circ$")
ax[0].legend()
ax[0].grid()
ax[1].errorbar(
    PTB["freqs"],
    ((PTB["phase"] - PTBPhasemean) + np.pi) / np.pi * 180,
    yerr=PTB["phaseuncer"] / np.pi * 180,
    label="PTB",
    fmt="o",
    markersize=MARKER_SIZE,
)
ax[1].errorbar(
    CEM["freqs"],
    (CEM["phase"] - CEMPhasemean) / np.pi * 180,
    yerr=CEM["phaseuncer"] / np.pi * 180,
    label="CEM",
    fmt="o",
    markersize=MARKER_SIZE,
)
ax[1].set(
    xlabel=r"Frequency $f$ in Hz",
    ylabel=r"$\Delta\varphi(f)-\overline{\Delta\varphi(f)}$ in $^\circ$",
)
ax[1].legend()
ax[1].grid()
fig.suptitle("Phase response")
fig.show()

fig, ax = plt.subplots()
ax.errorbar(
    PTB["freqs"],
    (PTB["phase"] - PTBPhasemean) + np.pi,
    yerr=PTB["phaseuncer"],
    label="PTB",
    fmt="o",
)
ax.errorbar(
    CEM["freqs"],
    (CEM["phase"] - CEMPhasemean),
    yerr=CEM["phaseuncer"],
    label="CEM",
    fmt="o",
)
ax.set(
    xlabel=r"Frequency $f$ in Hz",
    ylabel=r"$\Delta\varphi(f)-\overline{\Delta\varphi(f)}$ in deg",
)
ax.legend()
fig.show()

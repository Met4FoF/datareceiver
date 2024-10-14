import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
import tkinter as tk
from tkinter import filedialog
import bz2

figPickleVersion=0.1
filetypes=[("Comressed pickled Image Dict", ".cpid")]


plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
PLTSCALFACTOR = 1.5
SMALL_SIZE = 12 * PLTSCALFACTOR
MEDIUM_SIZE = 16 * PLTSCALFACTOR
BIGGER_SIZE = 18 * PLTSCALFACTOR

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

def saveImagePickle(name,fig,axs,axs2=None):
    saveDict={'Name':name,
              'fig':fig,
              'axs':axs,
              'matplotlib_version':matplotlib.__version__,
              'figPickleVersion':figPickleVersion}
    if not axs2 is None:
        saveDict['axs2']=axs2
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.asksaveasfilename(filetypes=filetypes)
    saveDict['file_name']=os.path.basename(file_path)
    with open(file_path, 'wb') as handle:
        pickled = pickle.dumps(saveDict)
        compressed=bz2.compress(pickled)
        print('Compression Ratio = '+str((len(compressed)/len(pickled))*100)+' %')
        handle.write(compressed)
        handle.flush()


if __name__=="__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=filetypes)
    with bz2.open(file_path, 'rb') as handle:

        figDict = pickle.load(handle)

    fig=figDict['fig']
    axs=figDict['axs']
    try:
        axs2=figDict['axs2']
    except KeyError:
        pass
    if     figDict['matplotlib_version']!=matplotlib.__version__:
        raise RuntimeWarning("File was Save mit Matplotlib Version "+str(figDict['matplotlib_version'])+'this Veiwer uses Version '+str(matplotlib.__version__))
    figDict['fig'].show()

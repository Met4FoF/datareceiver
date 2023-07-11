import matplotlib.pyplot as plt
import os
import pickle
import tkinter as tk
from tkinter import filedialog
import bz2

tubscolors=[(0/255,112/255,155/255),(250/255,110/255,0/255), (109/255,131/255,0/255), (81/255,18/255,70/255),(102/255,180/255,211/255),(255/255,200/255,41/255),(172/255,193/255,58/255),(138/255,48/255,127/255)]
tubsred=(176/255,0/255,70/255)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=tubscolors) #TUBS Blue,Orange,Green,Violet,Light Blue,Light Orange,Lieght green,Light Violet
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\boldmath'
LANG='DE'
if LANG=='DE':
    import locale
    locale.setlocale(locale.LC_NUMERIC,"de_DE.utf8")
    locale.setlocale(locale.LC_ALL,"de_DE.utf8")
    plt.rcParams['text.latex.preamble'] = r'\usepackage{icomma}\usepackage{amsmath}\boldmath' # remove nasty Space behind comma in de_DE.utf8 locale https://stackoverflow.com/questions/50657326/matplotlib-locale-de-de-latex-space-btw-decimal-separator-and-number
    plt.rcParams['axes.formatter.use_locale'] = True
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'NexusProSans'
plt.rcParams['mathtext.it'] = 'NexusProSans:italic'
plt.rcParams['mathtext.bf'] = 'NexusProSans:bold'
plt.rcParams['mathtext.tt'] = 'NexusProSans:monospace'
plt.rc('text', usetex=True)
plt.rc("figure", figsize=[16,9])  # fontsize of the figure title
plt.rc("figure", dpi=300)
PLTSCALFACTOR = 1
SMALL_SIZE = 9 * PLTSCALFACTOR
MEDIUM_SIZE = 12 * PLTSCALFACTOR
BIGGER_SIZE = 15 * PLTSCALFACTOR
plt.rc("font", weight='bold') # controls default text sizes
plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
figPickleVersion="0.2.0"
filetypes=[("Comressed pickled Image Dict", ".cpid")]
def saveImagePickle(name,fig,axs):
    saveDict={'Name':name.replace(' ','_'),
              'fig':fig,
              'axs':axs,
              'matplotlib_version':matplotlib.__version__,
              'figPickleVersion':figPickleVersion}
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.asksaveasfilename(title=saveDict['Name'],initialfile=saveDict['Name'],filetypes=filetypes)
    if len(file_path)== 0:
        print("no path given skipping")
        return
    saveDict['file_name']=os.path.basename(file_path)
    with open(file_path, 'wb') as handle:
        pickled = pickle.dumps(saveDict)
        compressed=bz2.compress(pickled)
        print('Compression Ratio = '+str((len(compressed)/len(pickled))*100)+' %')
        handle.write(compressed)
        handle.flush()
    return


if __name__=="__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=filetypes)
    with bz2.open(file_path, 'rb') as handle:
        figDict = pickle.load(handle)
    fig=figDict['fig']
    axs=figDict['axs']
    if     figDict['matplotlib_version']!=matplotlib.__version__:
        raise RuntimeWarning("File was Save mit Matplotlib Version "+str(figDict['matplotlib_version'])+'this Veiwer uses Version '+str(matplotlib.__version__))
    figDict['fig'].show()




import os
import sys
import pandas as pd
import csv
import json
import numpy as np
import matplotlib.pyplot as plt


class StaticMet4FoFData:
    def __init__(self, filename, ReltimeStart=0):
        self.reader = csv.reader(open(filename), delimiter=";")
        fristrow = next(self.reader)
        self.param = json.loads(fristrow[0])
        self.Data = pd.read_csv(filename, sep=";", skiprows=[0])
        self.Data["Time"] = (
            self.Data["unix_time"].astype(np.float64)
            + self.Data["unix_time_nsecs"].astype(np.float64) * 1e-9
        )
        if ReltimeStart == 0:
            self.Data["RelTime"] = self.Data["Time"] - self.Data["unix_time"][0]
        else:
            self.Data["RelTime"] = self.Data["Time"] - ReltimeStart


location = r"D:\datareceiver\data"

MPUFileNames = []
SD = []
for file in os.listdir(location):
    if file.rfind("MPU_9250_") != -1:
        MPUFileNames.append(os.path.join(location, file))
StartTimeSec = 0
i = 0
for filname in MPUFileNames:
    SD.append(StaticMet4FoFData(filname, StartTimeSec))
    if i == 0:
        StartTimeSec = SD[0].Data["unix_time"][0]

fig, ax = plt.subplots()
for D in SD:
    ax.plot(D.Data["RelTime"])
fig.show()

fig, ax = plt.subplots()

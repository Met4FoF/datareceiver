# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:12:32 2020

@author: benes
"""

import numpy as np
import pandas as pd
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm
from scipy.spatial.transform import Rotation as R
from scipy import optimize


def getAccelrationMeansFromDataSet(filename):
    PD = pd.read_csv(filename, sep=";", header=1)
    X = np.mean(PD["Data_01"])
    Y = np.mean(PD["Data_02"])
    Z = np.mean(PD["Data_03"])
    return np.array([X, Y, Z])


def cart2sph(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    XsqPlusYsq = x ** 2 + y ** 2
    r = m.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = m.atan2(z, m.sqrt(XsqPlusYsq))  # theta
    az = m.atan2(y, x)  # phi
    return r, elev, az


BMA280FileNames = [
    r"C:/Users/benes/datareceiver/zema_staic_cal/1/20200708123245_BMA_280_0xe0040000.dump",
    r"C:/Users/benes/datareceiver/zema_staic_cal/2/20200708123522_BMA_280_0xe0040000.dump",
    r"C:/Users/benes/datareceiver/zema_staic_cal/3/20200708123916_BMA_280_0xe0040000.dump",
    r"C:/Users/benes/datareceiver/zema_staic_cal/4/20200708124159_BMA_280_0xe0040000.dump",
    r"C:/Users/benes/datareceiver/zema_staic_cal/5/20200708125418_BMA_280_0xe0040000.dump",
    r"C:/Users/benes/datareceiver/zema_staic_cal/6/20200708125933_BMA_280_0xe0040000.dump",
]


MPU9250FileNames = [
    r"C:/Users/benes/datareceiver/zema_staic_cal/1/20200708123245_MPU_9250_0xe0040100.dump",
    r"C:/Users/benes/datareceiver/zema_staic_cal/2/20200708123523_MPU_9250_0xe0040100.dump",
    r"C:/Users/benes/datareceiver/zema_staic_cal/3/20200708123916_MPU_9250_0xe0040100.dump",
    r"C:/Users/benes/datareceiver/zema_staic_cal/4/20200708124159_MPU_9250_0xe0040100.dump",
    r"C:/Users/benes/datareceiver/zema_staic_cal/5/20200708125418_MPU_9250_0xe0040100.dump",
    r"C:/Users/benes/datareceiver/zema_staic_cal/6/20200708125933_MPU_9250_0xe0040100.dump",
]
NumOfVectors = len(MPU9250FileNames)
ExperimentNumber = [1, 2, 3, 4, 5, 6]
BMA280MeanVectors = []
BMA280MeanVectorsPol = []
BMA280MeanVectorsPolDeg = []
for File in BMA280FileNames:
    tmp = getAccelrationMeansFromDataSet(File)
    BMA280MeanVectors.append(tmp)
    BMA280MeanVectorsPol.append(cart2sph(tmp))
    tmpPol = cart2sph(tmp)
    tmpdeg = tmpPol * np.array([1, 180 / np.pi, 180 / np.pi])
    BMA280MeanVectorsPolDeg.append(tmpdeg)

MPU9250MeanVectors = []
MPU9250MeanVectorsPol = []
MPU9250MeanVectorsPolDeg = []
for File in MPU9250FileNames:
    tmp = getAccelrationMeansFromDataSet(File)
    MPU9250MeanVectors.append(tmp)
    MPU9250MeanVectorsPol.append(cart2sph(tmp))
    tmpPol = cart2sph(tmp)
    tmpdeg = tmpPol * np.array([1, 180 / np.pi, 180 / np.pi])
    MPU9250MeanVectorsPolDeg.append(tmpdeg)


def getDistance(angles):  # [alpha,beta,gamma]
    rotation = R.from_rotvec(np.array(angles))  # generate rotation matrix
    tmpDistance = 0
    for v in range(len(MPU9250MeanVectors)):  # loop over all test orientations
        rotvec = rotation.apply(BMA280MeanVectors[v])  # rotate one vector
        distvect = MPU9250MeanVectors[v] - rotvec  # calulate difference vector
        tmpDistance += np.linalg.norm(distvect)  # sum up difference vector length
    print(angles, tmpDistance)
    return tmpDistance


res = optimize.minimize(
    getDistance,
    np.array([0, np.pi, 0]),
    bounds=((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
    method="L-BFGS-B",
)
r_matrix = R.from_rotvec(np.array(res.x)).as_matrix()


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
color = iter(cm.rainbow(np.linspace(0, 1, NumOfVectors)))
for i in range(NumOfVectors):
    c = next(color)
    ax.quiver(
        0,
        0,
        0,
        MPU9250MeanVectors[i][0],
        MPU9250MeanVectors[i][1],
        MPU9250MeanVectors[i][2],
        color=c,
        linestyle="-",
        label="MPU " + str(ExperimentNumber[i]),
    )
    ax.quiver(
        0,
        0,
        0,
        BMA280MeanVectors[i][0],
        BMA280MeanVectors[i][1],
        BMA280MeanVectors[i][2],
        color=c,
        linestyle="--",
        label="BMA " + str(ExperimentNumber[i]),
    )
    BMARotated = r_matrix * BMA280MeanVectors[i]
    ax.quiver(
        0,
        0,
        0,
        BMARotated[0],
        BMARotated[1],
        BMARotated[2],
        color=c,
        linestyle="-.",
        label="BMA Rotated" + str(ExperimentNumber[i]),
    )
Lim = [-11, 11]
ax.set_xlim(Lim)
ax.set_ylim(Lim)
ax.set_zlim(Lim)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()

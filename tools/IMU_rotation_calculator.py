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
import scipy.linalg
from scipy.spatial.transform import Rotation as R
from scipy import optimize

def getAccelrationMeansFromDataSet(filename):
    PD=pd.read_csv(filename,sep=';',header=1)
    X=np.mean(PD['Data_01'])
    Y=np.mean(PD['Data_02'])
    Z=np.mean(PD['Data_03'])
    return np.array([X,Y,Z])

def cart2sph(xyz):
    x=xyz[0]
    y=xyz[1]
    z=xyz[2]
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # theta
    az = m.atan2(y,x)                           # phi
    return r, elev, az
    
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

BMA280FileNames=[r"C:/Users/benes/datareceiver/zema_staic_cal/1/20200708123245_BMA_280_0xe0040000.dump",
                 r"C:/Users/benes/datareceiver/zema_staic_cal/2/20200708123522_BMA_280_0xe0040000.dump",
                 r"C:/Users/benes/datareceiver/zema_staic_cal/3/20200708123916_BMA_280_0xe0040000.dump",
                 r"C:/Users/benes/datareceiver/zema_staic_cal/4/20200708124159_BMA_280_0xe0040000.dump",
                 r"C:/Users/benes/datareceiver/zema_staic_cal/5/20200708125418_BMA_280_0xe0040000.dump",
                 r"C:/Users/benes/datareceiver/zema_staic_cal/6/20200708125933_BMA_280_0xe0040000.dump"]
                 
                 
                 
MPU9250FileNames=[r"C:/Users/benes/datareceiver/zema_staic_cal/1/20200708123245_MPU_9250_0xe0040100.dump",
                  r"C:/Users/benes/datareceiver/zema_staic_cal/2/20200708123523_MPU_9250_0xe0040100.dump",
                  r"C:/Users/benes/datareceiver/zema_staic_cal/3/20200708123916_MPU_9250_0xe0040100.dump",
                  r"C:/Users/benes/datareceiver/zema_staic_cal/4/20200708124159_MPU_9250_0xe0040100.dump",
                  r"C:/Users/benes/datareceiver/zema_staic_cal/5/20200708125418_MPU_9250_0xe0040100.dump",
                  r"C:/Users/benes/datareceiver/zema_staic_cal/6/20200708125933_MPU_9250_0xe0040100.dump"]
NumOfVectors=len(MPU9250FileNames)
ExperimentNumber=[1,2,3,4,5,6]
BMA280MeanVectors=[]
BMA280MeanVectorsPol=[]
BMA280MeanVectorsPolDeg=[]
for File in BMA280FileNames:
    tmp=getAccelrationMeansFromDataSet(File)
    BMA280MeanVectors.append(tmp)
    BMA280MeanVectorsPol.append(cart2sph(tmp))
    tmpPol=cart2sph(tmp)
    tmpdeg=tmpPol*np.array([1,180/np.pi,180/np.pi])
    BMA280MeanVectorsPolDeg.append(tmpdeg)
    
MPU9250MeanVectors=[]
MPU9250MeanVectorsPol=[]
MPU9250MeanVectorsPolDeg=[]
for File in MPU9250FileNames:
    tmp=getAccelrationMeansFromDataSet(File)
    MPU9250MeanVectors.append(tmp)
    MPU9250MeanVectorsPol.append(cart2sph(tmp))
    tmpPol=cart2sph(tmp)
    tmpdeg=tmpPol*np.array([1,180/np.pi,180/np.pi])
    MPU9250MeanVectorsPolDeg.append(tmpdeg)
    
RotMat=[]

for i in range(NumOfVectors):
    RotMat.append(rotation_matrix_from_vectors(MPU9250MeanVectors[i],BMA280MeanVectors[i]))

#### Calculate Rotation Matrix
A = np.zeros([NumOfVectors,3])
B = np.zeros([NumOfVectors,3])
for i in range(NumOfVectors):
    #b (je eine spalte) = A * x(zeilenweise wird das die Rotationsmatrix)
    A[i,:] = BMA280MeanVectors[i]
    B[i,:] = MPU9250MeanVectors[i]

# Eigentliche QR-Zerlegung

#Q,R = scipy.linalg.qr(A)
#r_matrix = np.zeros([3,3])
#for i  in range(3):
    #Matlab ---> python
    # a'   ---->  a.conj().transpose()
    #A\B    ---> result, resid, rank, s = np.linalg.lstsq(A, B)
    #r_matrix(i,:)=(R\(Q'*b(:,i)))';
 #   tmp1=np.matmul(Q.conj().transpose(),B[:,i])#(Q'*b(:,i))
  #  tmp2, resid, rank, s = np.linalg.lstsq(R, tmp1)
   # r_matrix[i,:]=tmp2.conj().transpose()

RotTanja=-np.array([[0.9818,0.0438   ,0.0076],
           [0.0734,0.9786,-0.0511],
           [0.0042,-0.1034,-0.9214]])

def OptimizeRotMatrix(SensorAMeans,SensorBMeans,pointsPerAngle=360,):
    Distance=np.zeros([pointsPerAngle,pointsPerAngle,pointsPerAngle])
    smalestDistanceSoFar=1e300
    smalestAngles=[0,0,0]
    for i in range(pointsPerAngle):
        print(i)
        for j in range(pointsPerAngle):
            for k in range(pointsPerAngle):
                alpha=2*np.pi*i/(pointsPerAngle-1)
                beta=2*np.pi*j/(pointsPerAngle-1)
                gamma=2*np.pi*k/(pointsPerAngle-1)
                RotMatrix=R.from_rotvec(np.pi / 2 * np.array([alpha, beta, gamma]))
                tmpDistance=0
                for v in range(len(SensorAMeans)):
                    DistVect=SensorAMeans[v]-RotMatrix.as_matrix()*SensorBMeans[v]
                    tmpDistance+=np.linalg.norm(DistVect)
                Distance[i,j,k]=tmpDistance
                if(tmpDistance<smalestDistanceSoFar):
                    smalestDistanceSoFar=tmpDistance
                    smalestAngles=[alpha,beta,gamma]
                    print(smalestAngles,[i,j,k])
    return smalestAngles

def getDistance(angles):
    Rotation= R.from_rotvec(np.array(angles))
    tmpDistance=0
    for v in range(len(MPU9250MeanVectors)):
        RotVec=Rotation.apply(BMA280MeanVectors[v])
        DistVect = MPU9250MeanVectors[v] - RotVec
        tmpDistance += np.linalg.norm(DistVect)
    print(angles,tmpDistance)
    return tmpDistance
res=optimize.minimize(getDistance, np.array([-0.12337825,-3.14159265 ,0.07821242]), bounds=((-np.pi, np.pi), (-np.pi, np.pi),(-np.pi, np.pi)),method='SLSQP')

#samlestAngles=OptimizeRotMatrix(MPU9250MeanVectors,BMA280MeanVectors,pointsPerAngle=360)
r_matrix=R.from_rotvec(np.array(res.x)).as_matrix()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color=iter(cm.rainbow(np.linspace(0,1,NumOfVectors)))
for i in range(NumOfVectors):
    c = next(color)
    ax.quiver(0,0,0,MPU9250MeanVectors[i][0],MPU9250MeanVectors[i][1],MPU9250MeanVectors[i][2],color=c,linestyle='-',label='MPU '+str(ExperimentNumber[i]))
    #ax.quiver(0, 0, 0, BMA280MeanVectors[i][0], BMA280MeanVectors[i][1], BMA280MeanVectors[i][2], color=c,linestyle='--',label='BMA '+str(ExperimentNumber[i]))
    BMARotated=r_matrix*BMA280MeanVectors[i]
    ax.quiver(0, 0, 0, BMARotated[0], BMARotated[1], BMARotated[2], color=c,
             linestyle='-.', label='BMA Rotated' + str(ExperimentNumber[i]))
Lim=[-11,11]
ax.set_xlim(Lim)
ax.set_ylim(Lim)
ax.set_zlim(Lim)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()
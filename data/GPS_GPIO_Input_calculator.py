import h5py as h5py
import numpy as np
import matplotlib.pyplot as plt

hdf=h5py.File('data/multiBordGPSLog2.hfd5', "r+")
maxpoints=hdf['RAWDATA/0x60ad1e00_STM32_GPIO_Input'].attrs['Data_point_number']
times1=hdf['RAWDATA/0x60ad1e00_STM32_GPIO_Input/Absolutetime'][0,11:maxpoints]
times1uncer=hdf['RAWDATA/0x60ad1e00_STM32_GPIO_Input/Absolutetime_uncertainty'][0,11:maxpoints]
times2=hdf['RAWDATA/0xf1031e00_STM32_GPIO_Input/Absolutetime'][0,:maxpoints-11]
times2uncer=hdf['RAWDATA/0xf1031e00_STM32_GPIO_Input/Absolutetime_uncertainty'][0,:maxpoints-11]
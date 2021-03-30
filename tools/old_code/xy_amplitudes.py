if __name__ == "__main__":
    yappi.start()
    start = time.time()

    hdffilename = r"/media/benedikt/nvme/data/201118_BMA280_amplitude_frequency/20201118153703_BMA_280_0x1fe40000_00000.hdf5"
    # revcsv = r"/media/benedikt/nvme/data/201118_BMA280_amplitude_frequency/20201118153703_BMA_280_0x1fe40000_00000_Ref_TF.csv"

    datafile = h5py.File(hdffilename, "r+", driver="core")
    # add1dsinereferencedatatohdffile(revcsv, datafile)
    test = hdfmet4fofdatafile(datafile)

    # adc_tf_goup=datafile.create_group("REFENCEDATA/0x1fe40a00_STM32_Internal_ADC")
    # addadctransferfunctiontodset(adc_tf_goup,datafile["RAWDATA/0x1fe40a00_STM32_Internal_ADC"], [r"/media/benedikt/nvme/data/201118_BMA280_amplitude_frequency/200318_1FE4_ADC123_19V5_1HZ_1MHZ.json"])
    datafile.flush()

    # nomovementidx,nomovementtimes=test.detectnomovment('0x1fe40000_MPU_9250', 'Acceleration')
    movementidx, movementtimes = test.detectmovment(
        "0x1fe40000_BMA_280",
        "Acceleration",
        treshold=0.08,
        blocksinrow=1000,
        blocksize=50,
        plot=True,
    )
    manager = multiprocessing.Manager()
    mpdata = manager.dict()
    mpdata["hdfinstance"] = test
    mpdata["movementtimes"] = movementtimes
    mpdata["uniquexfreqs"] = np.unique(
        test.hdffile["REFENCEDATA/Acceleration_refference/Frequency"][0, :], axis=0
    )
    i = np.arange(movementtimes.shape[0])
    # i=np.arange(4)
    with multiprocessing.Pool(15) as p:
        results = p.map(processdata, i)
    end = time.time()
    print(end - start)
    i = 0

    DC = np.zeros(movementtimes.shape[0])
    AC = np.zeros(movementtimes.shape[0])
    ACNominal = test.hdffile[
        "REFENCEDATA/Acceleration_refference/Excitation amplitude"
    ][2, :, "value"]
    F = np.zeros(movementtimes.shape[0])
    for ex in results:
        DC[i] = ex.Data["0x1fe40000_BMA_280"]["Acceleration"]["SinPOpt"][2][1]
        AC[i] = ex.Data["0x1fe40000_BMA_280"]["Acceleration"]["SinPOpt"][2][0]
        F[i] = ex.Data["0x1fe40000_BMA_280"]["Acceleration"]["SinPOpt"][2][2]
        i = i + 1
    color = iter(cm.rainbow(np.linspace(0, 1, np.unique(F).size)))
    colordict = {}
    for i in range(np.unique(F).size):
        colordict[np.unique(F)[i]] = next(color)
    freqcolors = []
    for i in range(F.size):
        freqcolors.append(colordict[F[i]])
    fig, ax = plt.subplots()
    labelplotet = []
    for i in range(len(AC)):
        if F[i] not in labelplotet:
            ax.scatter(
                ACNominal[i], DC[i], color=freqcolors[i], Label="{:.1f}".format(F[i])
            )
            labelplotet.append(F[i])
        else:
            ax.scatter(ACNominal[i], DC[i], color=freqcolors[i])
    ax.set_xlabel("Nominal amplitude in m/s^2")
    ax.set_ylabel("DC in m/s^2")
    ax.legend()
    fig.show()

    # results[0].plotXYsine('0x1fe40000_BMA_280', 'Acceleration', 2)
    # fig,ax=plt.subplots()
    # coefs = np.empty([len(results), 3])
    # for ex in results:
    #    coefs[i]=ex.plotXYsine('0x1fe40000_BMA_280', 'Acceleration',2,fig=fig,ax=ax,mode='XY+fit')

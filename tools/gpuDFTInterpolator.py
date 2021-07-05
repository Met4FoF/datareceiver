import numpy as np
from reikna import cluda
from reikna.cluda import dtypes
from reikna.fft import FFT
from reikna.cluda.tempalloc import TrivialManager

tr = Transformation(
    inputs=['in_re'], outputs=['out_c'],
    derive_o_from_is=lambda in_re: dtypes.complex_for(in_re),
    snippet="${out_c.store}(COMPLEX_CTR(${out_c.ctype})(${in_re.load}, 0));")

def main():
    api = cluda.ocl_api()
    thr = api.Thread.create(temp_alloc=dict(cls=TrivialManager))

    N = 256
    M = 10000

    data_in = np.random.rand(N)
    data_in = data_in.astype(np.float32)

    cl_data_in = thr.to_device(data_in)

    cl_data_out = thr.array(data_in.shape, np.complex64)

    fft = FFT(thr)
    fft.connect(tr, 'input', ['input_re'])
    fft.prepare_for(cl_data_out, cl_data_in, -1, axes=(0,))


if __name__ == "__main__":
    main()
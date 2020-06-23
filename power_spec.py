import numpy as np
import matplotlib.mlab as mlab
import readligo as rl
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf


def rolling_windows(x, n, noverlap=None, axis=0):
    # Get all windows of x with length n as a single array, using strides to avoid data duplication.
    # This was taken from the documentation

    if noverlap is None:
        noverlap = 0
    x = np.asarray(x)

    if n == 1 and noverlap == 0:
        if axis == 0:
            # This adds one more dimension
            print(x)
            print(x[np.newaxis])
            return x[np.newaxis]

        else:
            return x[np.newaxis].transpose()

    noverlap = int(noverlap)
    n = int(n)
    step = n - noverlap

    if axis == 0:
        shape = (n, (x.shape[-1]-noverlap)//step)
        strides = (x.strides[0], step * x.strides[0])
    else:
        shape = ((x.shape[-1] - noverlap) // step, n)
        strides = (step * x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def psd_autocorr(x, NFFT, Fs):


    corr = np.correlate(x, x, mode='same')
    lags = np.arange(0, len(x)// 2)

    # Take the Fourier Transform of the data
    Pxx = np.fft.rfft(corr, NFFT)

    # Scale the Pxx values
    if not NFFT % 2:
        slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
    else:
        slc = slice(1, None, None)
    Pxx[slc] = Pxx[slc] * 2.
    Pxx = Pxx / Fs

    # Find the Frequencies
    freqs = np.fft.rfftfreq(NFFT, 1 / Fs)

    return Pxx.real, freqs

x = np.random.normal(0, 0.1, 500)
N = len(x)
corr = np.correlate(x, x, mode='same')
lags = np.arange(-N/2, N/2)
plt.plot(lags, corr)
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.show()

N = len(corr)
lengths = range(N, N//2, -1)

half = corr[N//2:].copy()
half /= lengths
half /= half[0]
plt.plot(half)
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.show()







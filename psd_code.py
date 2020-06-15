import numpy as np
import matplotlib.mlab as mlab
from numbers import Number


def rolling_windows(x, NFFT):
    # Returns array of windowed data segments
    window = np.hanning(NFFT)
    result = []
    i = 0
    while i < len(x):
        if len(x) - i <= int(NFFT/2):
            break
        else:
            seg = x[i:(i + NFFT)]
            result.append(window * seg)
            i += int(NFFT / 2)
    return result


def psd(x, NFFT, Fs):
    # Make sure the data is a np array
    x = np.asarray(x)

    # zero pad x if the data is less than the segment
    if len(x) < NFFT:
        n = len(x)
        x = np.resize(x, NFFT)
        x[n:] = 0

    if NFFT % 2:
        numFreqs = (NFFT + 1) // 2
    else:
        numFreqs = NFFT // 2 + 1


    # Add zeros to both sides to allow for rolling window to be applied
    h = int(NFFT / 2)
    n = np.zeros(h)
    x = np.concatenate([n, x, n])
    # Detrend the data
    x = mlab.detrend(x, key='none')
    # Apply the rolling hanning windows
    x = rolling_windows(x, NFFT)
    # Compute the real fft and only look at the
    x = np.fft.fft(x)[:numFreqs, :]
    print("x: ")
    print(x)
    x1 = np.fft.fft(x)
    print("x1: ")
    print(x1)
    x = x * np.conj(x)



    return


x = np.arange(1, 13, 1)
psd(x, 6, 2)



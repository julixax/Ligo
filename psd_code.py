import numpy as np
import matplotlib.mlab as mlab
from numbers import Number


def rolling_windows(x, NFFT):
    # Returns array of windowed data segments
    window = np.hanning(NFFT)
    print("window: " + str(len(window)))
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

    # Add zeros to both sides to allow for rolling window to be applied
    ##### Check with the even thing and what happens here might be an error
    h = int(NFFT / 2)
    n = np.zeros(h)
    x = np.concatenate([n, x, n])
    x = rolling_windows(x, NFFT)

    

    return


x = np.arange(1, 13, 1)
psd(x, 6, 2)

print(40 // 7)
print(40 / 7)
print(int(40 / 7))

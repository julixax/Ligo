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
    # Compute the real fft and only look at the positive frequencies
    x = np.fft.fft(x)[:numFreqs, :]
    # Calculate the magnitude squared
    Pxx = x * np.conj(x)


    # Scaling of the psd
    if not NFFT % 2:
        slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
    else:
        slc = slice(1, None, None)


    Pxx[slc] *= 2.
    Pxx /= Fs
    window = np.hanning(NFFT)
    # Scale the spectrum by the norm of the window to compensate for window loss
    Pxx /= (np.abs(window) ** 2).sum()

    # Determine the positive frequencies
    freqs = np.fft.fftfreq(NFFT, 1 / Fs)[:numFreqs]

    return Pxx, freqs





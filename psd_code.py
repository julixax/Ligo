import numpy as np
import matplotlib.mlab as mlab
import readligo as rl
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf


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


def psd(x, NFFT, Fs, noverlap=None):

    # If there is noverlap is None, then there is no overlap between the windows
    if noverlap is None:
        noverlap = 0

    # Make the window the size of the NFFT
    window = np.hanning(NFFT)

    # Make sure the data is a np array
    x = np.asarray(x)

    # zero pad x if the data is less than the segment NFFT
    if len(x) < NFFT:
        n = len(x)
        x = np.resize(x, NFFT)
        x[n:] = 0

    # Apply the rolling window to the data
    Pxx = rolling_windows(x, NFFT, noverlap, axis=0)

    # Detrend the data
    Pxx = mlab.detrend(Pxx, key='none')

    # Reshape the data
    Pxx = Pxx * window.reshape((-1, 1))

    # Compute the fft and only look at the positive frequencies
    Pxx = np.fft.rfft(Pxx, n=NFFT, axis=0)

    # Calculate the magnitude squared
    Pxx = Pxx * np.conj(Pxx)

    # Take the mean of the Pxx
    Pxx = Pxx.mean(axis=1)
    Pxx = Pxx.real

    # Scale the Pxx due to power loss from windowing
    # Scaling factors taken from the documentation to ensure that the graphs looked the same
    if not NFFT % 2:
        slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
    else:
        slc = slice(1, None, None)
    Pxx[slc] = Pxx[slc] * 2.
    Pxx = Pxx / Fs
    Pxx = Pxx / (np.abs(window) ** 2).sum()

    # Determine the positive frequencies
    freqs = np.fft.rfftfreq(NFFT, 1 / Fs)

    return Pxx, freqs


def psd_autocorr(x, NFFT, Fs, noverlap=None):

    # Determine the AutoCorrelation
    acorr_x = acf(x)

    # Take the Fourier Transform of the autocorrelation
    Pxx = np.fft.rfft(acorr_x, n=NFFT, axis=0)

    # Determine the frequencies
    freqs = np.fft.rfftfreq(NFFT, 1 / Fs)

    return Pxx, freqs


# -- Read in the file and data
fileName = '/Users/juliabellamy/PycharmProjects/ligo_stuff/LIGO/H-H1_LOSC_C00_4_V1-1186739813-4096.hdf5'
strain, time, channel_dict = rl.loaddata(fileName)
ts = time[1] - time[0]  # Time between samples
fs = int(1.0 / ts)  # Sampling frequency


# -- Choose a few seconds of "good data"
# this splits the time series into the sections of good data that can be used as a default analysis
# can look into the flag qualities to divide the list differently if we needed different things
segList = rl.dq_channel_to_seglist(channel_dict['DEFAULT'], fs)  # section of good data
length = 16  # seconds
# take the section of good data and get the data
strain_seg = strain[segList[0]][0:(length * fs)]
time_seg = time[segList[0]][0:(length * fs)]

'''''
# Plot the time series
plt.figure()
plt.plot(time_seg - time_seg[0], strain_seg)
plt.title("Time series")
plt.xlabel('Time since GPS ' + str(time_seg[0]))
plt.ylabel('Strain')
plt.show()
'''''

# Determine the segment length
NFFT = fs

# Calculate the autocorrelated PSD (above)
#PSD, freq = psd_autocorr(strain_seg, NFFT=NFFT, Fs=fs, noverlap=NFFT//2)




# Plot the autocorrelated psd and method psd on figure (make sure parameters are the same)
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle("PSD Comparisons")

plt.subplot(211)
plt.loglog(freq, PSD)
plt.title("PSD (Autocorrelation)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")

window1 = np.hanning(NFFT)
Pxx1, freqs1 = mlab.psd(strain_seg, NFFT=NFFT, Fs=fs, noverlap=NFFT//2, window=window1)

plt.subplot(212)
plt.loglog(freqs1, Pxx1)
plt.title("PSD (function)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.show()



'''''
diff = Pxx1 - PSD
plt.figure()
plt.plot(freq, diff)
plt.title("Difference between then two plots")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, 100)
plt.ylabel("Difference")
plt.show()
'''''

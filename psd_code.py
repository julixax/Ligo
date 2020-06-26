import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import readligo as rl


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


def acorr(x):

    # Make sure data is a numpy array and make copy
    x = np.asarray(x)
    y = np.copy(x)

    n = len(x)
    lags = np.arange(0, n, 1)
    acorr = []

    # Calculate autocorrelation
    for lag in lags:
        if lag == 0:
            c = (x * y).sum()
            acorr.append(c)
        else:
            x = x[1:]
            y = y[:n-lag]
            c = (x * y).sum()
            acorr.append(c)

    return acorr



def psd_auto(data, Fs):

    # Remove the mean from the signal
    data = data - np.mean(data)

    # Compute the autocorrelation of the data
    rxx = acorr(data)

    # Normalize the autocorrelated data between -1 and 1
    rxx_max = np.max(rxx)
    rxx = rxx / rxx_max

    # Apply a window to the correlated data
    window = np.hanning(len(rxx))
    rxx = rxx * window

    # Take the magnitude of the fft of the autocorrelated data
    pxx = np.fft.rfft(rxx)
    pxx = np.abs(pxx) / Fs

    # Determine the frequencies
    freq = np.fft.rfftfreq(len(rxx), 1 / Fs)

    return pxx, freq


'''''
def corr_coef(x, y, rowvar=True):

    # R (corr coef matrix) ij = C (covariance matrix) ij / sqrt( Cii * Cjj)
    # x and y are data of 1D or 2D array
    # rowvar = True (each row is a variable and observations in the columns)

    # Calculate the covariance
    c = np.cov(x, y, rowvar)

    try:
        # Determine the diagonal of the covariance matrix for calculation
        d = np.diag(c)
    except ValueError:
        # if the covariance is scalar, then the matrix is divided by itself
        return c / c

    d_sqrt = np.sqrt(d.real)
    c = c / d_sqrt[:, None]
    c = c / d_sqrt[None, :]

    # Data is now in 64 bits instead of 32, so clip the values and normalize between -1 and 1
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c


def auto_corr(x):
    n = len(x)
    lags = np.arange(len(x)//2)
    corrs = []
    for lag in lags:
        y1 = x[lag:]
        y2 = x[:n-lag]
        coefs = corr_coef(y1, y2)[0, 1]
        corrs.append(coefs)

    return lags, corrs


def psd_autocorr(x, NFFT, Fs):


    corr = np.correlate(x, x, mode='same')

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


def psd_auto(data, Fs):
    # Remove the mean from the signal
    data = data - np.mean(data)

    # Compute the autocorrelation of the data
    rxx = np.correlate(data, data, mode='same')

    # Normalize the autocorrelated data between -1 and 1
    rxx_max = np.max(rxx)
    rxx = rxx / rxx_max

    # Apply a window to the correlated data
    window = np.hanning(len(rxx))
    rxx = rxx * window

    # Take the magnitude of the fft of the autocorrelated data
    pxx = np.fft.rfft(rxx)
    pxx = np.abs(pxx) / Fs

    # Determine the frequencies
    freq = np.fft.rfftfreq(len(rxx), 1 / Fs)

    return pxx, freq
'''''


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


# Plot the time series
plt.figure()
plt.plot(time_seg - time_seg[0], strain_seg)
plt.title("Time series")
plt.xlabel('Time since GPS ' + str(time_seg[0]))
plt.ylabel('Strain')
plt.show()


# Plot and compare the PSD methods

# Calculate the mlab psd
nx = len(strain_seg)
Pxx1, freqs1 = mlab.psd(strain_seg, NFFT=fs, Fs=fs, noverlap=fs//2)
print(len(Pxx1))

# Calculate psd from averaging (mine)
Pxx2, freqs2 = psd(strain_seg, NFFT=fs, Fs=fs, noverlap=fs//2)
print(len(Pxx2))

# Calculate the autocorrelated PSD (above)
Pxx3, freqs3 = psd_auto(strain_seg, Fs=fs)
print(len(Pxx3))


fig = plt.figure(figsize=(12, 8))
fig.suptitle("PSD Comparisons")
plt.subplot(221)
plt.semilogy(freqs1, Pxx1, 'r', label="mlab psd")
plt.semilogy(freqs2, Pxx2, 'b', label="averaging fft psd")
plt.semilogy(freqs3, Pxx3, 'g', label="autocorrelation psd")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.legend()
plt.subplot(222)
plt.semilogy(freqs1, Pxx1, 'r')
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.subplot(223)
plt.semilogy(freqs2, Pxx2, 'b')
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.subplot(224)
plt.semilogy(freqs3, Pxx3, 'g')
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.show()

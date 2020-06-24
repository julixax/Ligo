import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


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


def psd_autocorr(x, NFFT, Fs):

    # Determine the autocorrelation
    lag, auto_x = auto_corr(x)

    # Take the Fourier Transform
    Pxx = np.fft.rfft(auto_x, NFFT)

    # Find the frequencies
    freqs = np.fft.rfftfreq(NFFT, 1 / Fs)

    '''''
    # Scale the Pxx values
    if not NFFT % 2:
        slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
    else:
        slc = slice(1, None, None)
    Pxx[slc] = Pxx[slc] * 2.
    Pxx = Pxx / Fs
    '''''

    return Pxx.real, freqs


# Try with signal
time = np.arange(0, 1, 0.1)
sin = np.sin(time)

psd_welch, freq_welch = psd(sin, 10, 10)
psd_auto, freq_auto = psd_autocorr(sin, 10, 10)

la, co = auto_corr(sin)

psd_inverse = np.fft.irfft(psd_welch)
print(psd_inverse)
print(co)
print(psd_inverse - co)

difference = psd_auto - psd_welch
avg_diff = np.average(difference)
ratio = psd_welch / psd_auto
avg_ratio = np.average(ratio)

print("difference: " + str(difference))
print("ratio: " + str(ratio))
print("avg difference: " + str(avg_diff))
print("avg ratio: " + str(avg_ratio))

fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle("PSD of Sine Wave (Comparison)")
plt.subplot(211)
plt.plot(freq_welch, psd_welch)
plt.title("PSD (Welch Method)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.subplot(212)
plt.plot(freq_auto, psd_auto)
plt.title("PSD (Autocorrelation Method)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.show()


# Try to reverse engineer what it should be


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


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


def autocorrelation(x):

    # Determine the length of the data
    n = len(x)

    # The maximum number of time lags
    lags = np.arange(len(x)//2)

    # Determine the correlation coefficient for each shifted signal with itself
    corrs = []
    for lag in lags:
        y1 = x[lag:]
        y2 = x[:n-lag]
        coefs = corr_coef(y1, y2)[0, 1]
        corrs.append(coefs)

    return corrs, lags


def psd_from_autocorrolate(x, NFFT, Fs, noverlap=None):

    # If there noverlap is None, then there is no overlap between the windows
    if noverlap is None:
        noverlap = 0

    # Make sure the data is a np array
    x = np.asarray(x)

    # Remove the mean from the data
    x = x - np.mean(x)

    # Break the data into blocks
    b = rolling_windows(x, NFFT, noverlap, axis=0)

    # Detrend the data
    b = mlab.detrend(b, key='none')

    # Calculate the PSD of each block
    m, n = b.shape
    i = 0
    Pxx = []
    while i < n:
        # Take the i-th column of the data blocks
        a = b[:, i]
        # Determine the autocorrelation
        corr, lag = autocorrelation(a)
        # Apply a window to each block
        corr_windowed = corr * np.hanning(len(corr))
        # Take the Fourier Transform of the autocorrelation of each block
        P = np.fft.rfft(corr_windowed, NFFT)
        P = np.abs(P)
        Pxx.append(P)
        i += 1

    # Average the blocks together
    Pxx = np.asarray(Pxx)
    Pxx = np.mean(Pxx, axis=0)

    # Scale the PSD
    if not NFFT % 2:
        slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
    else:
        slc = slice(1, None, None)
    Pxx[slc] = Pxx[slc] * 2.5
    Pxx = Pxx / Fs

    # Determine the frequencies
    freq = np.fft.rfftfreq(NFFT, 1 / Fs)
    return Pxx, freq


def psd_auto(x, NFFT, Fs):
    x = x - np.mean(x)
    Rxx, lags = autocorrelation(x)
    window = np.hanning(len(Rxx))
    Rxx_win = Rxx * window
    Rxx_fft = np.fft.rfft(Rxx_win, NFFT)
    Pxx = (Rxx_fft)
    fr = np.fft.rfftfreq(NFFT, 1 / Fs)

    if not NFFT % 2:
        slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
    else:
        slc = slice(1, None, None)
    Pxx[slc] = Pxx[slc] * 2
    Pxx = Pxx / Fs

    return Pxx, fr


# Number of data points
N = 1024
# time resolution
dt = 0.01
T = N*dt
t = np.linspace(0, N*dt, N)
# signal time series parameters
f1 = 10
h = np.sin(f1 * 2.0*np.pi*t)

plt.plot(t, h)
plt.show()


fs = 100
NFFT = fs
pxx, freq = mlab.psd(h, NFFT=NFFT, Fs=fs, noverlap=NFFT//2)
p, f = psd_from_autocorrolate(h, NFFT=NFFT, Fs=fs, noverlap=NFFT//2)
pa, fa = psd_auto(h, NFFT=NFFT, Fs=fs)

print(len(h))
print(len(pxx))
print(len(pa))



plt.semilogy(freq, np.sqrt(pxx), 'r', label="mlab psd")
#plt.plot(f, p, 'b', label="new auto psd")
plt.semilogy(fa, np.abs(pa.real), 'g', label="(real) auto psd")
plt.semilogy(fa, np.abs(pa.imag), 'b', label="(imag) auto psd")
plt.semilogy(fa, np.abs(pa), 'c', label="(mag) auto psd")
plt.xlabel("freq")
plt.ylabel("PSD")
plt.legend()
plt.show()


ratio = pxx / pa
print(ratio)
diff = pa - pxx
print(diff)


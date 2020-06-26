import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl

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


def autocorrelation(x):

    # Determine the length of the data
    n = len(x)

    # The maximum number of time lags
    lags = np.arange(len(x)//2)

    # Remove the mean from the data
    x = x - np.mean(x)

    # Determine the correlation coefficient for each shifted signal with itself
    corrs = []
    for lag in lags:
        y1 = x[lag:]
        y2 = x[:n-lag]
        coefs = corr_coef(y1, y2)[0, 1]
        corrs.append(coefs)

    return corrs, lags


def psd_from_autocorrolate(x, NFFT, Fs):
    Rxx, lags = autocorrelation(x)
    Rxx_copy = Rxx[:NFFT]
    window = np.hanning(len(Rxx_copy))
    Rxx_win = Rxx_copy * window
    Rxx_fft = np.fft.rfft(Rxx_win, NFFT)
    Pxx = np.abs(Rxx_fft)
    fr = np.fft.rfftfreq(NFFT, 1 / Fs)

    if not NFFT % 2:
        slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
    else:
        slc = slice(1, None, None)
    Pxx[slc] = Pxx[slc] * 1.32
    Pxx = Pxx / Fs


    return Pxx, fr


def rolling_windows(x, n, noverlap=None, axis=0):
    # Get all windows of x with length n as a single array, using strides to avoid data duplication.
    # This was taken from the documentation

    if noverlap is None:
        noverlap = 0
    x = np.asarray(x)

    if n == 1 and noverlap == 0:
        if axis == 0:
            # This adds one more dimension
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


'''''
# Number of data points
N = 512
# time resolution
dt = 0.01
T = N*dt
t = np.linspace(0, N*dt, N)
# signal time series parameters
f1 = 20
h = np.sin(f1 * 2.0*np.pi*t)

fs = 100

NFFT = fs
pxx, freq = mlab.psd(h, NFFT=NFFT, Fs=fs)       # Works!
P, f = psd_from_autocorrolate(h, NFFT=NFFT, Fs=fs)



plt.plot(freq, pxx, 'r', label="mlab psd")
plt.plot(f, P, 'b', label="auto psd")
plt.xlabel("freq")
plt.ylabel("PSD")
plt.legend()
plt.show()

ratio = pxx / P
print(np.max(ratio))
diff = P - pxx
print(np.max(diff))
'''''

t = np.arange(1, 51, 1)
f = 5
sin = np.sin(f * 2 * np.pi * t)

a = rolling_windows(t, 10, 2)
print(a)
print(a.shape)

i = 0
rxx = []
n, m = a.shape
while i < m:
    c = a[:, i]
    cor = acorr(c)
    rxx.append(cor)
    i += 1
rxx = np.asarray(rxx)
print(rxx.shape)
print(rxx)















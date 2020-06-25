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








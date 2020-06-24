import numpy as np
import matplotlib.pyplot as pl
from scipy.fftpack import fft, ifft, ifftshift


def autocorrelation(x):
    # Calculate the autocorrelation by computing the inverse of the psd
    xp = ifftshift((x - np.average(x))/np.std(x))
    n, = xp.shape
    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
    f = fft(xp)
    p = np.absolute(f)**2
    pi = ifft(p)
    print((np.arange(n//2)[::-1]+n//2))
    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)


def autocorrelation2(x):
    # Calculate autocorrelation directly
    maxdelay = len(x)//5
    N = len(x)
    mean = np.average(x)
    var = np.var(x)
    xp = (x - mean)/np.sqrt(var)
    autocorrelation = np.zeros(maxdelay)
    for r in range(maxdelay):
        for k in range(N-r):
            autocorrelation[r] += xp[k]*xp[k+r]
        autocorrelation[r] /= float(N-r)
    return autocorrelation


def autocorrelation3(x):
    # Calculate using numpy method
    xp = (x - np.mean(x))/np.std(x)
    result = np.correlate(xp, xp, mode='full')
    return result[result.size//2:]/len(xp)

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


def autocorrelation4(x):
    n = len(x)
    lags = np.arange(len(x)//2)
    corrs = []
    for lag in lags:
        y1 = x[lag:]
        y2 = x[:n-lag]
        coefs = corr_coef(y1, y2)[0, 1]
        corrs.append(coefs)

    return corrs


t = np.linspace(0,20,1024)
x = np.sin(t)
pl.plot(t[:200], autocorrelation(x)[:200], 'r', label='scipy fft')
pl.plot(t[:200], autocorrelation2(x)[:200],'b', label='direct autocorrelation')
pl.plot(t[:200], autocorrelation3(x)[:200],'g', label='numpy correlate')
pl.plot(t[:200], autocorrelation4(x)[:200],'c', label='numpy correlate')
pl.legend()
pl.show()

diff = autocorrelation2(x)[:200] - autocorrelation4(x)[:200]
print(np.amax(diff))
pl.plot(t[:200], diff[:200])
pl.show()

import numpy as np


def serial_corr(wave, lag=1):
    n = len(wave)
    y1 = x[lag:]
    y2 = x[:n-lag]
    corr = np.corrcoef(y1, y2)[0, 1]
    print(corr)
    return corr


def autocorr(x):
    lags = np.arange(len(x)//2)
    corrs = [serial_corr(x, lag) for lag in lags]
    return lags, corrs


'''''
corr coef:
R (corr coef matrix) ij = C (covariance matrix) ij / sqrt( Cii * Cjj)

take x (1D or 2D array) and y
rowvar = True (each row is a variable and observations in teh columns)

returns R (corr coef matrix of the values)
'''''


def corr_coef(x, y=None, rowvar=True):

    # R (corr coef matrix) ij = C (covariance matrix) ij / sqrt( Cii * Cjj)

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


x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])



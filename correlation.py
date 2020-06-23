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

    # Calculate the covariance (complex valued)
    c = np.cov(x, y, rowvar)
    # Determine the diagonal of the covariance matrix for calculation
    d = np.diag(c)
    # Calculate the denominator
    denominator = d[0] * d[1]
    denominator = np.sqrt(denominator)
    c = c / denominator

    return c


x = np.arange(0, 11, 1)
print(x)
cor = serial_corr(x)
print(cor)


import numpy as np
import matplotlib.pyplot as plt

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


x = np.random.normal(0, 0.1, 100)
lag, corr = auto_corr(x)

# Compare with np.correlate

# np.correlate returns values not normalized between -1 and 1
# returns positive and negative lags so lag=0 is the middle
# does not correct that the number of overlapping elements changes as the lag increases
corr2 = np.correlate(x, x, mode='same')

# Correct np.correlate to better reflect what is wanted
N = len(x)
lengths = range(N, N//2, -1)
half = corr2[N//2:].copy()
half = half / lengths
half = half / half[0]

# Comparison of autocorrelation methods
plt.plot(lag, corr, 'r', label="Coded")
plt.plot(lag, half, 'b', label="Modified Method")
plt.xlabel("Lags")
plt.ylabel("Correlation")
plt.title("Correlation comparison")
plt.legend()
plt.show()


# Check autocorrelation function with the logic of the autocorrelation of a sine wave

time = np.arange(0, 25, 0.1)
sin = np.sin(time)
lags, sin_auto_corr = auto_corr(sin)


fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle("Autocorrelation of Sine Wave")
plt.subplot(211)
plt.plot(time, sin)
plt.title("Signal")
plt.xlabel("Time")
plt.subplot(212)
plt.plot(lags, sin_auto_corr)
plt.title("Autocorrelated Signal")
plt.xlabel("Lags")
plt.show()





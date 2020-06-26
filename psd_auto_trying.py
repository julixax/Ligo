import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


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


t = np.arange(0, 100, .001)
ts = t[1] - t[0]
fs = 1/ts
f_nyq = 1 / (2 * ts)
print(ts)
print(fs)

f = 5
sin = np.sin(f * 2 * np.pi * t)
nse = np.random.normal(0, 0.1, len(sin))
signal = sin + nse

plt.subplot(211)
plt.plot(t, sin)
plt.xlim(0, 2)
plt.subplot(212)
plt.plot(t, signal)
plt.xlim(0, 2)
plt.show()

sin_x = sin - np.mean(sin)
signal_x = signal - np.mean(signal)

corr = np.correlate(sin_x, sin_x, mode='full')
corr = corr[len(corr)//2:]
corr_max = np.max(corr)
print("len(corr): " + str(len(corr)))
corr = corr / corr_max

sig_corr = np.correlate(signal_x, signal_x, mode='full')
sig_corr = sig_corr[len(sig_corr)//2:]
sig_corr_max = np.max(sig_corr)
print("len(sig_corr): " + str(len(sig_corr)))
sig_corr = sig_corr / sig_corr_max


window1 = np.hanning(len(corr))
corr = corr * window1
window2 = np.hanning(len(sig_corr))
sig_corr = sig_corr * window2

x = np.fft.rfft(corr)
x = np.abs(x) / fs
print("len(x): " + str(len(x)))
freq = np.fft.rfftfreq(len(corr), 1/fs)
print("len(freq): " + str(len(freq)))
print(freq[0:20])

sig_x = np.fft.rfft(sig_corr)
sig_x = np.abs(sig_x) / fs
print("len(sig_x): " + str(len(sig_x)))
sig_freq = np.fft.rfftfreq(len(sig_corr), 1/fs)
print("len(freq): " + str(len(sig_freq)))
print(sig_freq[0:20])

pxx, fxx = mlab.psd(sin, NFFT=len(sin), Fs=fs)
print("len(pxx): " + str(len(pxx)))
print("len(fxx): " + str(len(fxx)))

sig_pxx, sig_fxx = mlab.psd(signal, NFFT=len(signal), Fs=fs)
print("len(sig_pxx): " + str(len(sig_pxx)))
print("len(sig_fxx): " + str(len(sig_fxx)))

fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle("PSD comparison of 5Hz sin wave")
plt.subplot(121)
plt.semilogy(freq, x, 'r', label='correlation psd')
plt.semilogy(fxx, pxx, 'b', label='mlab pas')
plt.xlim(0, 10)
plt.title("PSD of pure signal")
plt.legend()
plt.subplot(122)
plt.semilogy(sig_freq, sig_x, 'r', label='correlation psd')
plt.semilogy(sig_fxx, sig_pxx, 'b', label='mlab pas')
plt.xlim(0, 10)
plt.title("PSD of noisy signal")
plt.legend()
plt.show()

pxx_int = np.trapz(pxx, dx=0.01)
x_int = np.trapz(x, dx=0.01)
sig_int = np.trapz(sig_x, dx=0.01)
print(pxx_int)
print(x_int)
print(sig_int)

print(np.max(x))
print(np.max(pxx))
print(np.max(sig_x))
print(np.max(sig_pxx))


y = np.array([1, 2, 3, 4, 5])
a_x = acorr(y)
a = np.correlate(y, y, 'full')
a = a[len(a)//2:]


ls = np.arange(0, len(y), 1)

plt.plot(ls, a_x, 'r', label='mine')
plt.plot(ls, a, 'b', label='numpy')
plt.legend()
plt.show()

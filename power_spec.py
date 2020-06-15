import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl

fileName = '/Users/juliabellamy/PycharmProjects/ligo_stuff/LIGO/H-H1_LOSC_C00_4_V1-1186739813-4096.hdf5'
# -- Read in the file and data
strain, time, channel_dict = rl.loaddata(fileName)
ts = time[1] - time[0]  # Time between samples
fs = int(1.0 / ts)  # Sampling frequency

# Info about raw data points
print("Number of data points in time series: " + str(len(time)))
print("ts: " + str(ts) + " \nfs: " + str(fs))

# -- Choose a few seconds of "good data"
# this splits the time series into the sections of good data that can be used as a default analysis
# can look into the flag qualities to divide the list differently if we needed different things
segList = rl.dq_channel_to_seglist(channel_dict['DEFAULT'], fs)  # section of good data
length = 16  # seconds
# take the section of good data and get the data
strain_seg = strain[segList[0]][0:(length * fs)]
time_seg = time[segList[0]][0:(length * fs)]

# Info about data points in the time series
print("length of segList: " + str(len(segList)))
print("segList[0]: " + str(segList[0]))
print("strain_seg length: " + str(len(strain_seg)))
print("time_seg length: " + str(len(time_seg)))

'''''
# Plot the time series
plt.figure()
plt.plot(time_seg - time_seg[0], strain_seg)
plt.title("Time series")
plt.xlabel('Time since GPS ' + str(time_seg[0]))
plt.ylabel('Strain')
plt.show()
'''''


def start(array, trend, window_type):
    seg_trend = mlab.detrend(array, key=trend)
    win = window_type[int(len(window_type) / 2):len(window_type)]
    seg_win = win * seg_trend
    return seg_win


def end(array, trend, window_type):
    seg_trend = mlab.detrend(array, key=trend)
    win = window_type[0:int(len(window_type) / 2)]
    seg_win = win * seg_trend
    return seg_win


def power(data):
    fft = np.fft.rfft(data)
    mag_fft = np.abs(fft)
    p = np.square(mag_fft)
    psd = np.average(p)
    return psd


len_strain_seg = len(strain_seg)  # length of data
len_seg = 64  # length of segments
i = 0  # index
seg = [] * len_seg
Pxx = []
window = np.hanning(len_seg)
print("window_length: " + str(len(window)))


time_slice = ts * len_seg
min_freq = 1 / time_slice
print("min_freq: " + str(min_freq))
f_nyquist = 1 / (2 * ts)
print("f_nyquist: " + str(f_nyquist))

begin = True


# Create the segments
while i < len_strain_seg:
    if begin:
        seg = strain_seg[0:int(len_seg / 2)]
        seg_windowed = start(seg, "none", window)
        P = power(seg_windowed)
        Pxx.append(P)
        begin = False
        pass
    elif not len_strain_seg - i < len_seg and not begin:
        seg1 = strain_seg[i:(i + len_seg)]
        seg_trend_1 = mlab.detrend(seg1, key='none')
        seg_windowed_1 = window * seg_trend_1
        P1 = power(seg_windowed_1)
        Pxx.append(P1)
        i += int(len_seg/2)
    else:
        seg2 = strain_seg[i:(i + int(len_seg / 2))]
        seg_windowed_2 = start(seg2, "none", window)
        P2 = power(seg_windowed_2)
        Pxx.append(P2)
        break

# Generate the frequency space
f = np.linspace(0, f_nyquist, int(len(Pxx)))
print("len(f): " + str(len(f)))
print("len(Pxx): " + str(len(Pxx)))

# Plot the coded PSD and the PSD python generated
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle("PSD Comparisons")
plt.subplot(211)
plt.loglog(f, Pxx)
plt.title("PSD (coded)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")


plt.subplot(212)
Pxx_1, freqs_1 = mlab.psd(strain_seg, Fs=fs, NFFT=fs)

print("len(freqs_1): " + str(len(freqs_1)))
print("len(Pxx_1): " + str(len(Pxx_1)))

plt.loglog(freqs_1, Pxx_1)
plt.title("PSD (function)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.show()


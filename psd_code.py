import numpy as np
import matplotlib.mlab as mlab

def psd(x, NFFT, Fs):
    # Make sure the data is a np array
    x = np.asarray(x)
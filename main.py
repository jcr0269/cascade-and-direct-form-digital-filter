import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import scipy.signal as signal
from matplotlib.figure import Figure
from matplotlib import rcParams
# from plot_zplane import zplane
from math import pi
import filtercascade


def zplane(b, a, filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b / float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a / float(kd)
    else:
        kd = 1

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn / float(kd)

    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0,
             markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0,
             markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5;
    plt.axis('scaled');
    plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1];
    plt.xticks(ticks);
    plt.yticks(ticks)

    if filename is None:
        plt.title('Z pole plot')
        plt.show()
    else:
        plt.savefig(filename)

    return z, p, k

def magnitude(a, b):
    w1, h1 = signal.freqz(b, a, fs=pi)
    plt.plot(w1 /(2*np.pi), 20*np.log10(abs(h1)))
    plt.title('Filter 1 Frequency Response')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()
def impulse(a, b, begin, end):
    t = np.linspace(begin, end)
    freq, h = signal.freqz(b,a, fs=pi)
    samp_freq = 1000
    z = (freq * pi, 20 * np.log10(abs(h)))
    h = h/h.max()
    x = signal.filtfilt(b, a, h)
    plt.title('Impulse Response')
    plt.plot(x)
    plt.show()
    return x
def convolve(a, b):
    a4 = signal.convolve(a,b)
    return a4

b = np.array([1])
a = np.array([1, -3.502, 5.026, -3.464, 0.979])
a1 = np.array([1, -3.50, 5.03, -3.46, 0.98])
a2 = np.array([1, -1.9, 0.99])
a3 = np.array([1, -1.61, 0.99])
a4 = convolve(a2,a3)

zplane(b,a4)
zplane(b, a)
zplane(b, a1)
impulse(a1, b, begin=0, end=100)
impulse(a, b, begin=0, end=100)
impulse(a4, b, begin=0, end=100)
magnitude(a, b)
magnitude(a1,b)
magnitude(a4,b)
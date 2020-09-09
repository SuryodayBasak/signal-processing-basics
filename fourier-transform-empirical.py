import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

SUBSAMPLE_SIZE = 1024
START_IDX = 1024

NAME = 'power-chord-distorted'
audio = wave.open('audio-samples/'+NAME+'.wav')

signal = audio.readframes(-1)
signal = np.fromstring(signal, "Int16")
fs = audio.getframerate()

fx = signal[START_IDX: START_IDX + SUBSAMPLE_SIZE]
t = np.linspace(0, SUBSAMPLE_SIZE, num = SUBSAMPLE_SIZE) / 10.0

"""
First, we try the discrete Fourier transform.
"""
X = []
N = SUBSAMPLE_SIZE

for k in range(SUBSAMPLE_SIZE):
    Xk = 0   
    for n in range(0, N):
        Xk += fx[n] * (np.e ** ((-1j * 2 * np.pi * k * n )/ N))
    # Xre.append(np.real(Xk))
    X.append(Xk)

# Run the method from scipy.fft to validate my approach.
yf = fft(fx)

fig, axs = plt.subplots(2)
#T = t[1] - t[0]
T = 1/44100.0
xf = np.linspace(0.0, 1.0/(2.0 * T), N//2)
axs[0].plot(xf, 2.0/N * np.abs(X[0:N//2]))
axs[1].plot(xf, 2.0/N * np.abs(yf[0:N//2]))
axs[0].set_ylabel("My method")
axs[1].set_ylabel("scipy")
fig.suptitle("DFT and FFT on " + NAME)
plt.savefig('plots/fourier-transform-empirical/' + NAME + '.jpg')

"""
Next, we try the inverse Fourier transform.
"""

y = [] #Contain the points of the original signal
for n in range(SUBSAMPLE_SIZE):
    yn = 0   
    for k in range(0, N):
        yn += X[k] * (np.e ** ((1j * 2 * np.pi * k * n )/ N))
    y.append(yn)  

y_scipy = ifft(yf)
plt.clf()

fig, axs = plt.subplots(3)
axs[0].plot(t, y)
axs[1].plot(t, y_scipy)
axs[2].plot(t, fx)
axs[0].set_ylabel("My method")
axs[1].set_ylabel("scipy")
axs[2].set_ylabel("Original signal")

fig.suptitle("IFFT on " + NAME)
plt.savefig('plots/fourier-transform-empirical/inverse-' + NAME + '.jpg')

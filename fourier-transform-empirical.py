import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.fft import fft

SUBSAMPLE_SIZE = 1024
START_IDX = 1024

NAME = 'two-notes-distorted'
audio = wave.open('audio-samples/'+NAME+'.wav')

signal = audio.readframes(-1)
signal = np.fromstring(signal, "Int16")
fs = audio.getframerate()

fx = signal[START_IDX: START_IDX + SUBSAMPLE_SIZE]
t = np.linspace(0, SUBSAMPLE_SIZE, num = SUBSAMPLE_SIZE) / 10.0

# DFT

Xre = []
Xim = []
N = SUBSAMPLE_SIZE

for k in range(SUBSAMPLE_SIZE):
    Xk = 0   
    for n in range(0, N):
        Xk += fx[n] * (np.e ** ((-1j * 2 * np.pi * k * n )/ N))
    # Xre.append(np.real(Xk))
    Xre.append(Xk)
    Xim.append(np.imag(Xk))

# Run the method from scipy.fft to validate my approach.
yf = fft(fx)

fig, axs = plt.subplots(2)
#T = t[1] - t[0]
T = 1/44100.0
xf = np.linspace(0.0, 1.0/(2.0 * T), N//2)
axs[0].plot(xf, 2.0/N * np.abs(Xre[0:N//2]))
axs[1].plot(xf, 2.0/N * np.abs(yf[0:N//2]))
axs[0].set_ylabel("My method")
axs[1].set_ylabel("scipy")
fig.suptitle("DFT and FFT on " + NAME)
plt.savefig('plots/fourier-transform-empirical/' + NAME + '.jpg')

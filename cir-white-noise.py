import wavio
import numpy as np
from scipy.fft import fft, ifft
from matplotlib import pyplot as plt

SAMPLE_RATE = 44100

data_orig = wavio.read('audio-samples/random-signal/original.wav')
data_recd = wavio.read('audio-samples/random-signal/recorded.wav')

y_o = data_orig.data[:, 0]
y_r = data_recd.data[:, 0]

n_samples = len(y_o)
x = np.linspace(0, n_samples/SAMPLE_RATE, n_samples)

# Plot original signal.
plt.title("Original Signal -- Sample rate: 44100 Hz")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.plot(x, y_o)
plt.savefig("plots/cir-whitenoise/original-signal.png")
plt.clf()

# Find the DFT of the original signal.
sample_spacing = 1 / SAMPLE_RATE
xt = np.linspace(0.0, 1/(2.0 * sample_spacing), n_samples//2)
yt_original = fft(y_o)
plt.plot(xt, 2.0/n_samples * np.abs(yt_original[0:n_samples//2]))
plt.grid()
plt.savefig("plots/cir-whitenoise/original-signal-dft.png")

# Find the DFT of the recorded signal.
plt.clf()
sample_spacing = 1 / SAMPLE_RATE
yt_recd = fft(y_r)
plt.plot(xt, 2.0/n_samples * np.abs(yt_recd[0:n_samples//2]))
plt.grid()
plt.savefig("plots/cir-whitenoise/recorded-signal-dft.png")

# Find CIR
plt.clf()
#yt_original = 2.0/n_samples * np.abs(yt_original[0:n_samples])
#yt_recd = 2.0/n_samples * np.abs(yt_recd[0:n_samples])
CIR = np.divide(yt_recd, yt_original)
CIR = np.abs(ifft(CIR))
plt.xlim(0.0, 0.03) 
plt.plot(x, CIR)
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.show()
plt.savefig("plots/cir-whitenoise/cir.png")

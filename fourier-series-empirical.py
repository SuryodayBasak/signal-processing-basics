import wave
import numpy as np
import sys
import matplotlib.pyplot as plt

# Return an array of cosines.
def getCosines(t, n):
    cosArray = np.zeros(n)
    for i in range(0, n):
        cosArray[i] = np.cos((i + 1)*t)
    return cosArray

# Return an array of sines.
def getSines(t, n):
    sinArray = np.zeros(n)
    for i in range(0, n):
        sinArray[i] = np.sin((i + 1)*t)
    return sinArray

SUBSAMPLE_SIZE = 1024
START_IDX = 1024

audio = wave.open('audio-samples/single-note-clean.wav')

signal = audio.readframes(-1)
signal = np.fromstring(signal, "Int16")
fs = audio.getframerate()

Time = np.linspace(0, len(signal)/fs, num=len(signal))
plt.figure(1)
plt.title("Signal Wave")

# Plot time by offsetting to starting point.
#plt.plot(Time[START_IDX:START_IDX + SUBSAMPLE_SIZE],
#         signal[START_IDX: START_IDX + SUBSAMPLE_SIZE])
plt.plot(Time[0:SUBSAMPLE_SIZE],
         signal[START_IDX: START_IDX + SUBSAMPLE_SIZE])
plt.show()

t = np.linspace(0, SUBSAMPLE_SIZE, num = SUBSAMPLE_SIZE) / 10.0
fx = np.array([signal[START_IDX: START_IDX + SUBSAMPLE_SIZE]]).T
N = 4000 # Number of terms in fourier expansion.

featMat = []
print(len(t))
for i in range(0, SUBSAMPLE_SIZE):
    cosines = getCosines(t[i], N)
    sines = getSines(t[i], N)
    feature_vec = np.concatenate((np.array([1]), cosines, sines))
    featMat.append(feature_vec)

featMat = np.array(featMat)

# Linear regression.

# Init a coefficients array
w = np.random.random((N * 2 + 1, 1))

print(np.shape(fx))
print(np.shape(featMat))
print(np.shape(w))

N_ITERS = 1000
LR = 0.0005
for i in range(0, N_ITERS):
    l = (fx - np.dot(featMat, w))
    xT = featMat.T
    L = np.dot(xT, l)
    w -= (1/N) * L * LR
    print(i)

# Visualize the graph:
fx_ = []
for ti in t:
    cosines = getCosines(ti, N)
    sines = getSines(ti, N)
    feature_vec = np.concatenate((np.array([1]), cosines, sines))
    feature_ary = np.array([feature_vec])
    fx_.append(np.dot(feature_ary, w)[0, 0])

plt.clf()
plt.plot(Time[0:SUBSAMPLE_SIZE],fx_)
plt.show()


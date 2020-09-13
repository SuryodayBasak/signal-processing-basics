import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy import interpolate

note_map = {'Ds3': 155.56,
            'E3': 164.81,
            'F3': 174.61,
            'Fs3': 185.0,
            'G3': 196.00,
            'Gs3': 207.65,
            'A3': 220.00,
            'As3': 233.08,
            'B3': 246.94,
            'C4': 261.63,
            'Cs4': 277.18,
            'D4': 293.66,
            'Ds4': 311.13,
            'E4': 329.63,
            'F4': 349.23,
            'Fs4': 369.99,
            'G4': 392.00,
            'Gs4': 415.30,
            'A4': 440.00,
            'As4': 466.16,
            'B4': 493.88,
            'C5': 523.25,
            'Cs5': 554.37,
            'D5': 587.33,
            'Ds5': 622.25,
            'E5': 659.25,
            'F5': 698.46,
            'Fs5': 739.99,
            'G5': 783.99,
            'Gs5': 830.61,
            'A5': 880.00,
            'As5': 932.33,
            'B5': 987.77,
            'C6': 1046.50,
            'Cs6': 1108.73,
            'D6': 1174.66}

idx_map = {1: 'Ds3',
           2: 'E3',
           3: 'F3',
           4: 'Fs3',
           5: 'G3',
           6: 'Gs3',
           7: 'A3',
           8: 'As3',
           9: 'B3',
           10: 'C4',
           11: 'Cs4',
           12: 'D4',
           13: 'Ds4',
           14: 'E4',
           15: 'F4',
           16: 'Fs4',
           17: 'G4',
           18: 'Gs4',
           19: 'A4',
           20: 'As4',
           21: 'B4',
           22: 'C5',
           23: 'Cs5',
           24: 'D5',
           25: 'Ds5',
           26: 'E5',
           27: 'F5',
           28: 'Fs5',
           29: 'G5',
           30: 'Gs5',
           31: 'A5',
           32: 'As5',
           33: 'B5',
           34: 'C6',
           35: 'Cs6',
           36: 'D6'}

SUBSAMPLE_SIZE = 1024 * 4
START_IDX = 1024 * 4

DIR = 'audio-samples/cir-samples2/'
FNAME = 'cir-samples-2-'

"""
Find CIR of the first (left) channel.
First, loop over the audio samples and find their frequency response.
"""


amp_list = []
freq_list = []

for i in range(1, 37):
    name = DIR + FNAME + str(i).zfill(3) + '.wav'
    audio = wave.open(name)

    signal = audio.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    fs = audio.getframerate()

    # note = idx_map[i]
    note = idx_map[i]
    freq = note_map[note]

    freq_list.append(freq)
    start = START_IDX
    end = START_IDX + SUBSAMPLE_SIZE
    xt = fft(signal[start:end])
    amp = np.max(np.abs(xt))
    amp_list.append(amp)

intr = interpolate.interp1d(freq_list, amp_list, kind='cubic')
x_intr = np.linspace(156, 1174.66, 200)
y_intr = intr(x_intr)
plt.scatter(freq_list, amp_list)
plt.plot(x_intr, y_intr)
plt.show()
plt.clf()

"""
IFFT Time.
"""

cir = ifft(y_intr)
n = len(cir)
x_cir = [i for i in range(1, int(n/2))]
plt.plot(x_cir, np.abs(cir[1:int(n/2)]))
plt.show()

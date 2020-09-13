import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy import interpolate

note_map = {'A4': 440.00,
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

idx_map = {1: 'A4',
           2: 'As4',
           3: 'B4',
           4: 'C5',
           5: 'Cs5',
           6: 'D5',
           7: 'Ds5',
           8: 'E5',
           9: 'F5',
           10: 'Fs5',
           11: 'G5',
           12: 'Gs5',
           13: 'A5',
           14: 'As5',
           15: 'B5',
           16: 'C6',
           17: 'Cs6',
           18: 'D6'}

SUBSAMPLE_SIZE = 1024 * 4
START_IDX = 1024 * 4

DIR = 'audio-samples/cir-samples/'
FNAME = 'cir-sample-splitting-'

"""
Find CIR of the first (left) channel.
First, loop over the audio samples and find their frequency response.
"""


amp_list = []
freq_list = []

for i in range(1, 19):
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
x_intr = np.linspace(440, 1174, 100)
y_intr = intr(x_intr)
plt.scatter(freq_list, amp_list)
plt.plot(x_intr, y_intr)
plt.show()
plt.clf()

"""
IFFT Time.
"""

cir = ifft(y_intr)
x_cir = [i for i in range(0, len(cir))]
plt.plot(x_cir, np.abs(cir))
plt.show()

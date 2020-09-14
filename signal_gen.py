import wavio
import numpy as np

"""
Create a random sequence.
"""

SAMPLE_RATE = 44100
N_CHANNELS = 1
N_SECONDS = 5
N_WIDTH = 3
MIN_AMP = -8388605
MAX_AMP = 8388605
sig_amps = np.random.random(N_SECONDS * SAMPLE_RATE)
sig_amps = (MAX_AMP - MIN_AMP) * sig_amps + MIN_AMP
sig_amps = sig_amps.astype(np.int32)

"""
Generate wave file.
"""
wavio.write('audio-samples/random-signal/data.wav',
             sig_amps,
             SAMPLE_RATE,
             sampwidth = N_WIDTH)

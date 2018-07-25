import numpy as np
import math
import matplotlib.pyplot as plt

num_freq_bins = 512

frequencies = np.array([2.0 * 180 * i / num_freq_bins * 22050 / 360 for i in range(num_freq_bins)])
frequencies[0] = 2.0 * 180 / num_freq_bins / 2 * 22050 / 360 # 0th frequency threshold is computed at 3/4th of the frequency range

ath_val = 3.64 * np.power(1000 / frequencies, 0.8) - 6.5 * np.exp(-0.6 * np.power(frequencies / 1000 - 3.3, 2)) +\
          np.power(0.1, 3) * np.power(frequencies / 1000, 4)
ath_shifted = (1 - np.amin(ath_val)) + ath_val  # shift all ath vals so that min is 1
weights = 1 / ath_shifted
print(frequencies)
print(weights)

normalized = np.full(weights.shape, np.sqrt(np.sum(np.power(weights, 2)) / num_freq_bins))
print(np.linalg.norm(weights, ord=2))
print(np.linalg.norm(normalized, ord=2))

plt.plot(frequencies, weights)
plt.xlabel('Frequency (Hz)')
plt.ylabel('weights')
plt.show()

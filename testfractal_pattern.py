import numpy as np
import matplotlib.pyplot as plt

def create_fractal_rhythm(size, fractal_power=1.5):
    timeline = np.linspace(0, 1, size)
    signal = np.zeros_like(timeline)
    for i in range(1, int(size / 2)):
        signal += np.sin(2 * np.pi * (2 ** i) * timeline) / (i ** fractal_power)
    return (signal - signal.min()) / (signal.max() - signal.min())

# Parameters
series_length = 1000
fractal_power = 1.5

# Generate sequence
sequence = create_fractal_rhythm(50, fractal_power)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, 1, series_length), sequence, color='blue')
plt.title(f'Fractal Rhythm Sequence (Exponent: {fractal_power})')
plt.xlabel('Time')
plt.ylabel('Normalized Amplitude')
plt.grid(True, linestyle='--', alpha=0.7)

# Reference line
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

# Display
plt.tight_layout()
plt.show()

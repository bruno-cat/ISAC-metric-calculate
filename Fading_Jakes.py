import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class JakesGen:
    def __init__(self, blockSize, fDts, Ns, randomSeed):
        self.blockSize = blockSize
        self.fDts = fDts
        self.Ns = Ns
        self.state = np.random.RandomState(randomSeed)

    def generate(self):
        # Initialize output
        y = np.zeros((self.blockSize,), dtype=complex)
        # Generate independent in-phase and quadrature components
        for i in range(self.Ns):
            # Uniformly distributed angles
            alpha = (2 * np.pi / self.Ns) * i
            # Doppler frequency for this component
            fn = self.fDts * np.cos(alpha)
            # Random initial phase
            phi = 2 * np.pi * self.state.rand()
            # Generate in-phase and quadrature components
            I = np.cos(2 * np.pi * fn * np.arange(self.blockSize) + phi)
            Q = np.sin(2 * np.pi * fn * np.arange(self.blockSize) + phi)
            # Accumulate components
            y += I + 1j * Q

        # Normalize output
        y /= np.sqrt(2 * self.Ns)
        return y


def apply_fading_to_iq_data(iq_data, h):
    return iq_data * h

# Test the function
SampleRate = 30.72e6
N = 10 * np.int16(SampleRate / 1000)  # Number of samples
Ts = 1 / SampleRate  # Sampling period (1 ms). Timelapse between the two consecutive samples

# Generate 4-QAM IQ data
M = 4  # QAM order
m = int(np.sqrt(M))

# Generate QAM symbols
symbols = [(2 * i - m + 1) + 1j * (2 * j - m + 1) for i in range(m) for j in range(m)]
symbols = np.array(symbols, dtype=complex)

# Repeat symbols to create IQ data
iq_data = np.tile(symbols, N // M + 1)

# Ensure iq_data length is N
iq_data = iq_data[:N]

# Create JakesGen object. Try out various different values for the following values. Play first with Ns and speed.
randomSeed = None  # Random seed

Ns = 20  # Number of scatterers
speed = 60  # km/hour
v = speed * 1000 / 3600  # velocity in m/s
fc = 2000 * 10e6  # frequency in Hz
c = 3e8  # light of speed in m/s
fD = v * fc / c
fDts = fD * Ts  # Normalized Doppler frequency
blockSize = N  # Block size
print("fDts = ", fDts)
jakesGen = JakesGen(blockSize, fDts, Ns, randomSeed)

# Generate Jake's model
h = jakesGen.generate()
faded_iq_data = apply_fading_to_iq_data(iq_data, h)

# Create a grid for the subplots
gs = gridspec.GridSpec(3, 2)
plt.rcParams.update({'font.size': 7})
# Plot the original and faded data in separate subplots
fig = plt.figure(figsize=(6, 8))

# Original signal constellation

ax0 = plt.subplot(gs[0, 0])
ax0.scatter(iq_data.real, iq_data.imag)
ax0.set_title('Original Signal Constellation')
ax0.set_xlabel('In-Phase (I)')
ax0.set_ylabel('Quadrature (Q)')
ax0.grid(True)

# Original signal power profile

ax1 = plt.subplot(gs[0, 1])
ax1.plot(10 * np.log10(np.abs(iq_data) ** 2))
ax1.set_title('Original Signal Power Profile')
ax1.set_xlabel('Time')
ax1.set_ylabel('Power (dB)')
ax1.grid(True)

# Faded signal constellation

ax2 = plt.subplot(gs[1, 0])
ax2.scatter(faded_iq_data.real, faded_iq_data.imag, s=1)
ax2.set_title('Faded Signal Constellation')
ax2.set_xlabel('In-Phase (I)')
ax2.set_ylabel('Quadrature (Q)')
ax2.grid(True)

# Faded signal power profile

ax3 = plt.subplot(gs[1, 1])
power_faded_db = 10 * np.log10(np.abs(faded_iq_data) ** 2)
power_faded_db_normalized = power_faded_db - np.max(power_faded_db)
ax3.plot(power_faded_db_normalized)
ax3.set_title('Faded Signal Power Profile')
ax3.set_xlabel('Time')
ax3.set_ylabel('Power (dB)')
ax3.grid(True)

# Fading process

ax4 = plt.subplot(gs[2, 0])
ax4.plot(h.real)
ax4.set_title('h (real)')
ax4.set_xlabel('t')
ax4.set_ylabel('h')
ax4.grid(True)

ax5 = plt.subplot(gs[2, 1])
ax5.plot(h.imag)
ax5.set_title('h (imag)')
ax5.set_xlabel('t')
ax5.set_ylabel('h')
ax5.grid(True)

fig.tight_layout()

plt.show()
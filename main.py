import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal


class AudioSample:
    def __init__(self, Fe, data):
        self.Fe = Fe
        self.data = data
        self.N = len(data)
        self.total_time =  1/Fe * self.N


def load_audio(file):
    Fe, data = wavfile.read(file)
    return AudioSample(Fe, data)


# Down sample beginning
def down_sample(audioSample, samples=None, start_time=0, end_time=None, plot=True):

    # Resolve parameters
    samples = not samples and audioSample.N or samples
    end_time = not end_time and 1/audioSample.Fe * audioSample.N or end_time
    max_samples = int(audioSample.Fe * (end_time - start_time))
    N2 = samples > max_samples and max_samples or samples
    start_ind = int(start_time*audioSample.Fe)
    end_ind = int(end_time*audioSample.Fe)

    down_sample_data = signal.resample(audioSample.data[start_ind:end_ind], N2)
    dn = (end_time - start_time) / N2
    n = np.arange(start_time, end_time, dn)

    newFe = audioSample.Fe*(samples/audioSample.N)

    if plot:
        plt.figure(1)
        plt.plot(n, down_sample_data)
        plt.title('LA# (N: ' + str(N2) + '), time: ' + str(start_time) + 's - ' + str(end_time) + 's')

    return AudioSample(newFe, down_sample_data)

# Return FULL amplitude and phase (not just for the specified intervals)
def fourier_spectra(audioSample, x_normalized=False, x_Freq = False, y_dB = False, showPhase=False, start_m=0, end_m=None):
    end_m = not end_m and audioSample.N or end_m

    # X axis
    w_norm = 2 * np.pi / audioSample.N
    if x_normalized:
        n = np.arange(w_norm*start_m, w_norm*end_m, w_norm)
    elif x_Freq:
        step = audioSample.Fe/audioSample.N
        n = np.arange(start_m*step, end_m*step, step)
    else:
        n = np.arange(start_m, end_m, 1) # m

    # Compute amplitude
    dft = np.fft.fft(audioSample.data)
    amp = np.abs(dft)
    if y_dB:
        amp = 20 * np.log10(amp)

    # Show amplitude (amp)
    plt.figure(2)
    if showPhase:
        plt.subplot(2, 1, 1)
    plt.plot(n, amp[start_m:end_m], 'g')
    plt.title('Spectre amplitude')

    # Name axis
    if y_dB:
        plt.ylabel('Amplitude (dB)')
    else:
        plt.ylabel('Amplitude')

    if x_normalized:
        plt.xlabel('Freq norm (Rad/echantillon)')
    elif x_Freq:
        plt.xlabel('Freq (Hz)')
    else:
        plt.xlabel('m')

    # Show phase
    phase = np.angle(dft)
    if showPhase:
        plt.subplot(2, 1, 2)
        plt.plot(n, phase, 'g')
        plt.title('Spectre phase')

    return amp, phase

def get_harmonic_params(f0, num_harmonics, amp_data, phase_data, sample, printResults=True):
    for i in range(1, num_harmonics+1):
        harmonic_freq = f0 * i
        harmonic_m = round(harmonic_freq * sample.N / sample.Fe)
        harmonic_amp = amp_data[harmonic_m]
        harmonic_phase = phase_data[harmonic_m]
        if printResults:
            print(f'Harmonic #{i}: {harmonic_freq} Hz --> Amp = {harmonic_amp:.3f} | Phase = {harmonic_phase:.4f}')

    return harmonic_amp, harmonic_phase

sample = load_audio('./audio/note_guitare_LAd.wav')
sampleample_down = down_sample(sample, 160000, plot=True)
amp, phase = fourier_spectra(sample, x_normalized=False, x_Freq=True, y_dB=True, showPhase=False)

harm_amp, harm_phase = get_harmonic_params(466, 32, amp, phase, sample, printResults=True)




plt.show()











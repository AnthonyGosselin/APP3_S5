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
    print(newFe)

    if plot:
        plt.figure(1)
        plt.plot(n, down_sample_data)
        plt.title('LA# (N: ' + str(N2) + '), time: ' + str(start_time) + 's - ' + str(end_time) + 's')

    return AudioSample(newFe, down_sample_data)


def fourier_spectra(audioSample, normalized=False, in_dB = False, showPhase=False):
    dft = np.fft.fft(audioSample.data)

    w_norm = 2 * np.pi / audioSample.N
    n = np.arange(0, w_norm*audioSample.N, w_norm) if normalized else np.arange(0, audioSample.N, 1)

    # Compute amplitude
    amp = np.abs(dft)
    if in_dB:
        amp = 20 * np.log10(amp)

    # Show amplitude (amp)
    plt.figure(2)
    if showPhase:
        plt.subplot(2, 1, 1)
    plt.plot(n, amp, 'g')
    plt.title('Spectre amplitude')

    if in_dB:
        plt.ylabel('Amplitude (dB)')

    # Show phase
    if showPhase:
        plt.subplot(2, 1, 2)
        plt.plot(n, np.angle(dft), 'g')
        plt.title('Spectre phase')


def filtreFIR(audioSample, normalized=False):
    wc = np.pi / 1000
    Fe = audioSample.Fe
    N = audioSample.N

    w_norm = 2 * np.pi / N

    fc = wc * Fe / (2 * np.pi)

    m = Fe * N / Fe
    k = 2 * m + 1

    n = np.arange(0, w_norm * N, w_norm) if normalized else np.arange(0, N, 1)

    signalRedressé = np.abs(audioSample.data)

    FIRpb = np.zeros(N)

    index = 0
    for nval in n:
        if index == 0:
            FIRpb[index] = k / N
            #if num == 'c1' or num == 'c2':
            #    hwindow[index] = k / N * window[index]
        else:
            FIRpb[index] = np.sin(np.pi * nval * k / N) / (N * np.sin(np.pi * nval / N))
            #if num == 'c1' or num == 'c2':
            #    hwindow[index] = np.sin(np.pi * nval * k / N) / (N * np.sin(np.pi * nval / N)) * window[index]
        index += 1




    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title("Signal redressé")
    plt.plot(signalRedressé)
    plt.subplot(3,1,2)






guitarSample = load_audio('./audio/note_guitare_LAd.wav')
guitarSample_down = down_sample(guitarSample, 160000, plot=True)
fourier_spectra(guitarSample, normalized=True, in_dB=False, showPhase=False)

filtreFIR(guitarSample, normalized=True)

plt.show()

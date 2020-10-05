import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

guitarFile = './audio/note_guitare_LAd.wav'
bassonFile = './audio/note_basson_plus_sinus_1000_Hz.wav'


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

    print(N2)

    down_sample_data = signal.resample(audioSample.data[start_ind:end_ind], N2)
    dn = (end_time - start_time) / N2
    n = np.arange(start_time, end_time, dn)

    newFe = audioSample.Fe*(samples/audioSample.N)

    if plot:
        plt.figure(1)
        plt.plot(n, down_sample_data*np.hanning(N2))
        plt.title('LA# (N: ' + str(N2) + '), time: ' + str(start_time) + 's - ' + str(end_time) + 's')

    plt.show()

    return AudioSample(newFe, down_sample_data)

# Return FULL amplitude and phase (not just for the specified intervals)
def fourier_spectra(audioSample, x_normalized=False, x_Freq = False, y_dB = False, showPhase=False, start_m=0, end_m=None):
    end_m = not end_m and audioSample.N or end_m

    # X axis
    w_norm = 2 * np.pi / audioSample.N
    if x_normalized:
        m = np.arange(w_norm*start_m, w_norm*end_m, w_norm)
    elif x_Freq:
        step = audioSample.Fe/audioSample.N
        m = np.arange(start_m*step, end_m*step, step)
    else:
        m = np.arange(start_m, end_m, 1) # m

    # Compute amplitude
    dft = np.fft.fft(audioSample.data)
    amp = np.abs(dft)
    if y_dB:
        amp = 20 * np.log10(amp)

    # Show amplitude (amp)
    plt.figure(2)
    if showPhase:
        plt.subplot(2, 1, 1)
    plt.plot(m, amp[start_m:end_m], 'g')#, use_line_collection=True)
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
        plt.plot(m, phase, 'g')
        plt.title('Spectre phase')

    plt.show()

    return amp, phase

def get_harmonic_params(f0, num_harmonics, amp_data, phase_data, sample, printResults=True):
    harmonic_amp = []
    harmonic_phase = []
    for i in range(1, num_harmonics+1):
        harmonic_freq = f0 * i
        harmonic_m = round(harmonic_freq * sample.N / sample.Fe)
        harmonic_amp.append(amp_data[harmonic_m])
        harmonic_phase.append(phase_data[harmonic_m])
        if printResults:
            print(f'Harmonic #{i}: {harmonic_freq} Hz --> Amp = {harmonic_amp[i]:.3f} | Phase = {harmonic_phase[i]:.4f}')

    return harmonic_amp, harmonic_phase


def filtrePasseBas(audioSample, normalized=False):
    wc = np.pi / 1000
    Fe = audioSample.Fe
    N = audioSample.N
    w_norm = 2 * np.pi / N
    fc = wc * Fe / (2 * np.pi)

    m = Fe * N / Fe
    k = 2 * m + 1

    n = np.arange(0, w_norm * N, w_norm) if normalized else np.arange(0, N, 1)

    FIRpb = np.zeros(N)

    # Calculer valeurs du filtres
    index = 0
    for nval in n:
        if index == 0:
            FIRpb[index] = k / N
            #hwindow[index] = k / N * window[index]
        else:
            FIRpb[index] = np.sin(np.pi * nval * k / N) / (N * np.sin(np.pi * nval / N))
            #hwindow[index] = np.sin(np.pi * nval * k / N) / (N * np.sin(np.pi * nval / N)) * window[index]
        index += 1

    return FIRpb

def convFiltre(signal, filtre, y_dB=False, verbose=False):

    if y_dB:
        redressedSignal = 20 * np.log10(np.abs(signal.data))
    else:
        redressedSignal = np.abs(signal.data)

    # Convolution du filtre et du signal
    redressedFFTSignal = np.fft.fft(redressedSignal)
    filterFFT = np.fft.fft(filtre)

    filteredSignalFrequentiel = filterFFT * redressedFFTSignal
    filteredSignalTemporel = np.fft.ifft(filteredSignalFrequentiel)

    if verbose:
        plt.figure()
        # Signal redressé
        plt.subplot(3, 2, 1)
        plt.title("Signal redressé")
        plt.plot(redressedSignal)
        plt.subplot(3,2,2)
        plt.title("Signal redressé (freq)")
        plt.plot(np.abs(np.fft.fft(redressedSignal)))

        # Réponse du filtre
        plt.subplot(3,2,3)
        plt.title("Filtre passe-bas")
        plt.plot(filtre)
        plt.subplot(3, 2, 4)
        plt.title("Filtre passe-bas (freq)")
        plt.plot(np.abs(np.fft.fft(redressedSignal)))

        # Filtre * signal
        #filteredSignal = np.abs(redressedSignal * FIRpb)
        plt.subplot(3,2,5)
        plt.title("Signal redressé convolué avec filtre passe-bas")
        plt.plot(np.abs(filteredSignalTemporel))
        plt.subplot(3, 2, 6)
        plt.title("Signal redressé convolué avec filtre passe-bas (freq)")
        plt.plot(np.abs(filteredSignalFrequentiel))

        plt.show()


def filtreCoupeBande(signal, normalized=False, verbose=False):

    N = signal.N
    filtreCB = np.zeros(N)

    w1 = (1040 - 960) / 2
    w0 = 960 + w1

    n = np.arange(0, N, 1)

    reponseImp = filtrePasseBas(signal, normalized)
    d = np.concatenate([[1], np.zeros(N-1)])

    filtreCB = d - 2 * reponseImp * np.cos(w0 * n)

    if verbose:
        plt.figure()
        plt.subplot(2,1,1)
        plt.title("Filtre coupe-bande temporel")
        plt.plot(filtreCB)
        plt.subplot(2,1,2)
        plt.title("Filtre coupe-bande frequentiel")
        plt.plot(np.abs(np.fft.fft(filtreCB)))

    return filtreCB


def guitFunct():
    sample = load_audio('./audio/note_guitare_LAd.wav')
    sampleample_down = down_sample(sample, 160000, plot=True)
    amp, phase = fourier_spectra(sample, x_normalized=False, x_Freq=True, y_dB=True, showPhase=False)

    harm_amp, harm_phase = get_harmonic_params(466, 32, amp, phase, sample, printResults=True)

    filtrePB = filtrePasseBas(sample, normalized=True)
    convFiltre(sample, filtrePB, False, True)

    plt.show()


def bassonFunct():
    sample = load_audio('./audio/note_basson_plus_sinus_1000_Hz.wav')

    filtreCB = filtreCoupeBande(sample, normalized=False, verbose=True)

    plt.show()

#guitFunct()
bassonFunct()
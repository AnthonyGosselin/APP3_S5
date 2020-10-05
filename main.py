import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
import Filtres

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


def convFiltre(signal, filtre, y_dB=False, verbose=False):

    redressedSignal = np.abs(signal.data)

    # Convolution du filtre et du signal
    redressedFFTSignal = np.fft.fft(redressedSignal)
    filterFFT = np.fft.fft(filtre)

    if y_dB:
        redressedFFTSignal = 20 * np.log10(np.abs(redressedFFTSignal))
        filterFFT = 20 * np.log10(np.abs(filterFFT))


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
        plt.title("Signal conv. filtre passe-bas")
        plt.plot(np.abs(filteredSignalTemporel))
        plt.subplot(3, 2, 6)
        plt.title("Signal conv. filtre passe-bas (freq)")
        plt.plot(np.abs(filteredSignalFrequentiel))

        plt.show()

    def sample_synthesis(f0, harmonic_amp, harmonic_phase, original_audio, sin_count=32):

        dn = original_audio.total_time / original_audio.N
        n = np.arange(0, original_audio.total_time, dn)

        n2 = np.arange(0.1, 0.12, dn)

        plt.figure(4)
        synth_signal = 0
        synth_test = 0
        for i in range(0, sin_count):
            amp = harmonic_amp[i]
            freq = 2 * np.pi * (f0 * (i + 1))
            phase = harmonic_phase[i]

            new_sin = amp * np.sin(freq * n + phase)
            synth_signal = synth_signal + new_sin

            # if i < 2:
            #     test_sin = amp*np.sin(freq*n2 - phase)
            #     plt.plot(n2, test_sin, 'b')
            #     synth_test = synth_test + test_sin

        # plt.plot(n2, synth_test, 'r')
        # plt.show()
        # plt.figure(5)
        # plt.plot(n, synth_signal)

        print(np.round(synth_signal))
        wavfile.write('./audio/out_audio.wav', original_audio.Fe, np.round(synth_signal))

    ###########################################################################################


def guitFunct():
    sample = load_audio(guitarFile)
    sample_down = down_sample(sample, plot=True)  # , start_time=0.17, end_time=0.18)
    amp, phase = fourier_spectra(sample, x_normalized=False, x_Freq=True, y_dB=False,
                                 showPhase=False)  # , start_m=0, end_m=1000)

    harm_amp, harm_phase = get_harmonic_params(466, 32, amp, phase, sample, printResults=False)
    # sample_synthesis(466, harm_amp, harm_phase, sample)

    filtrePB = Filtres.filtrePasseBas(sample, y_dB=True, normalized=True, verbose=True)
    #convFiltre(sample, filtrePB, y_dB=True, verbose=True)

    plt.show()


def bassonFunct():
    sample = load_audio(bassonFile)

    filtreCB = Filtres.filtreCoupeBande(sample, normalized=False, verbose=True)

    plt.show()

guitFunct()
#bassonFunct()
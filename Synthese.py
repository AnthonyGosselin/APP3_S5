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

def write_audio(file_name, Fe, data):
    data_in16 = data.astype("int16")
    wavfile.write('./audio/'+file_name+'.wav', Fe, data_in16)

# Down sample beginning
def down_sample(audioSample, samples=None, start_time=0, end_time=None, plot=True, window=False):

    # Resolve parameters
    samples = not samples and audioSample.N or samples
    end_time = not end_time and 1/audioSample.Fe * audioSample.N or end_time
    max_samples = int(audioSample.Fe * (end_time - start_time))
    N2 = samples > max_samples and max_samples or samples
    start_ind = int(start_time*audioSample.Fe)
    end_ind = int(end_time*audioSample.Fe)

    print(N2)

    down_sample_data = signal.resample(audioSample.data[start_ind:end_ind], N2)
    if window:
        down_sample_data = down_sample_data * np.hanning(N2)
    dn = (end_time - start_time) / N2
    n = np.arange(start_time, end_time, dn)

    newFe = int(audioSample.Fe*(samples/audioSample.N))

    if plot:
        plt.figure(1)
        plt.plot(n, down_sample_data)
        plt.title('LA# (N: ' + str(N2) + '), time: ' + str(start_time) + 's - ' + str(end_time) + 's')

    return AudioSample(newFe, down_sample_data)


def get_local_max(data, center_val):
    r = 20
    max = 0
    max_ind = 0
    for i in range(center_val-r, center_val+r+1):
        val = data[i]
        if val > max:
            max = val
            max_ind = i

    # if max_ind == center_val - r or max_ind == center_val + r:
    #     print("LIMIT CASE")

    return max_ind, max


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
    amp = np.abs(dft)/audioSample.N
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


    # Test: retain 32 sinus-------------------
    newData = [0]*audioSample.N
    sin_extract = []
    sinus_count = 0

    for h in range(1, 32+1):
        freq = 466*h
        ind_m = int(freq*audioSample.N/audioSample.Fe)
        max_ind, max_amp = get_local_max(amp, ind_m)
        newData[max_ind] = max_amp
        sin_extract.append(amp[ind_m])
        # print(max_ind, max_amp)


    # print(sin_extract)
    plt.figure(3)
    plt.plot(m, newData)
    plt.title('32 sin extrait')

    dn = (audioSample.total_time) / audioSample.N
    t = np.arange(0, audioSample.total_time, dn)

    plt.figure(4)
    inv_signal = np.fft.ifft(newData)
    plt.plot(t, inv_signal)
    plt.title("Inverse fft for synth signal")
    print(inv_signal)

    write_audio('inv_spectra', audioSample.Fe, inv_signal)

    return amp, phase, inv_signal

def get_harmonic_params(f0, num_harmonics, amp_data, phase_data, sample, printResults=True):
    harmonic_amp = []
    harmonic_phase = []
    for i in range(1, num_harmonics+1):
        harmonic_freq = f0 * i
        harmonic_m = round(harmonic_freq * sample.N / sample.Fe)
        max_ind, max_amp = get_local_max(amp_data, harmonic_m)
        harmonic_amp.append(amp_data[max_ind])
        harmonic_phase.append(phase_data[max_ind])
        if printResults:
            print(f'Harmonic #{i}: {harmonic_freq} Hz --> Amp = {amp_data[harmonic_m]:.3f} | Phase = {phase_data[harmonic_m]:.4f}')

    return harmonic_amp, harmonic_phase


def sample_synthesis(f0, harmonic_amp, harmonic_phase, original_audio, sin_count=32):

    dn = original_audio.total_time / original_audio.N
    n = np.arange(0, original_audio.total_time, dn)

    #plt.figure(4)
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
    plt.figure(5)
    plt.plot(n, synth_signal)

    synth_signal = synth_signal.astype("int16")
    #print(synth_signal, original_audio.Fe)

    wavfile.write('./audio/out_audio.wav', original_audio.Fe, synth_signal)


def apply_envelope(envelope, data, total_time, N):
    dn = (total_time) / (N + 883)
    t = np.arange(0, total_time, dn)

    sample_padded = np.append(data, [0] * 883)

    final_synth_signal = envelope * sample_padded
    plt.figure(7)
    # plt.plot(t, envelope, 'b')
    # plt.plot(t, inv_fft_signal_padded, 'g')
    plt.plot(t, final_synth_signal, 'r')
    plt.title('final synth')

    write_audio('final_synth', 44100, final_synth_signal)
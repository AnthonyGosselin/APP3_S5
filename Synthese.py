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
    data_in16 = np.real(data).astype("int16")
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
    r = 1000
    max = -9999
    max_ind = 0
    left = int(r/2)
    right = r
    for i in range(center_val-left, center_val+right+1):
        val = data[i]
        if val > max:
            max = val
            max_ind = i

    if max_ind == center_val - left or max_ind == center_val + right:
        print("LIMIT CASE")

    return max_ind, max


# Return FULL amplitude and phase (not just for the specified intervals)
def fourier_spectra(audioSample, x_normalized=False, x_Freq = False, y_dB = False, showPhase=False, start_m=0, end_m=None, title="Spectre amplitude", figure_num=2):
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
    plt.figure(figure_num)
    if showPhase:
        plt.subplot(2, 1, 1)
    plt.plot(m, amp[start_m:end_m], 'g')#, use_line_collection=True)
    plt.title(title)

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

    return amp, phase

def get_harmonic_params(f0, num_harmonics, amp_data, phase_data, sample, printResults=True):
    harmonic_amp = []
    harmonic_phase = []
    harmonic_freq = []
    for i in range(1, num_harmonics+1):
        freq = f0 * i
        harmonic_m = round(freq * sample.N / sample.Fe)
        max_ind, max_amp = get_local_max(amp_data, harmonic_m)
        harmonic_amp.append(amp_data[max_ind])
        harmonic_phase.append(phase_data[max_ind])

        new_freq = max_ind * sample.Fe / sample.N
        harmonic_freq.append(new_freq)
        if printResults:
            #print(f' ogm={harmonic_m}, ogf={freq} m={max_ind} Harmonic #{i}: {new_freq} Hz --> Amp = {amp_data[max_ind]:.3f} | Phase = {phase_data[max_ind]:.4f}')
            print(f'{i}\t{new_freq:.0f}\t{amp_data[max_ind]:.3f}\t{phase_data[max_ind]:.4f}')

    return harmonic_amp, harmonic_phase, harmonic_freq


def sample_synthesis(f0, harmonic_amp, harmonic_phase, harmonic_freq, original_audio, sin_count=32):

    dn = original_audio.total_time / original_audio.N
    n = np.arange(0, original_audio.total_time, dn)

    # calculate diff
    freq_diff = []
    for i in range(0, sin_count):
        base_freq = 466 * (i+1)
        shifted_freq = f0 * (i+1)
        diff = base_freq - shifted_freq
        freq_diff.append(diff)

    synth_signal = 0
    all_sins = []
    for i in range(0, sin_count):
        amp = harmonic_amp[i]
        freq = harmonic_freq[i] - freq_diff[i]
        w = 2 * np.pi * freq #(f0 * (i + 1))
        phase = harmonic_phase[i]

        new_sin = amp * np.sin(w * n + phase)
        all_sins.append(new_sin)
        synth_signal = synth_signal + new_sin

    plt.figure(5)
    plt.plot(n, synth_signal)

    write_audio('sin_sum', original_audio.Fe, synth_signal)

    return synth_signal, all_sins


def apply_envelope(envelope, padding, data, total_time, N, save_as="final_synth"):
    dn = (total_time) / (N + padding)
    t = np.arange(0, total_time-dn, dn)

    padding = padding - 1

    sample_padded = np.append(data, [0] * padding)

    final_synth_signal = envelope * sample_padded / padding
    plt.figure(7)
    plt.plot(t, final_synth_signal, 'r')
    plt.title(save_as)

    write_audio(save_as, 44100, final_synth_signal)

    return final_synth_signal


def generate_synthesis(f0, envelope, padding, harm_amp, harm_phase, harm_freq, sample, save_as="final_synth"):
    sin_sum, all_sins = sample_synthesis(f0, harm_amp, harm_phase, harm_freq, sample, 32)
    final_signal = apply_envelope(envelope, padding, sin_sum, sample.total_time, sample.N, save_as=save_as)

    return final_signal

def create_symphony(envelope, padding, harm_amp, harm_phase, harm_freq, sample_down):
    la_d = generate_synthesis(466, envelope, padding, harm_amp, harm_phase, harm_freq, sample_down, save_as="final_LAd")
    sol = generate_synthesis(392, envelope, padding, harm_amp, harm_phase, harm_freq, sample_down, save_as="final_SOL")
    mi_b = generate_synthesis(311, envelope, padding, harm_amp, harm_phase, harm_freq, sample_down, save_as="final_MIb")
    fa = generate_synthesis(349, envelope, padding, harm_amp, harm_phase, harm_freq, sample_down, save_as="final_FA")
    re = generate_synthesis(294, envelope, padding, harm_amp, harm_phase, harm_freq, sample_down, save_as="final_RE")

    lad_sample = AudioSample(44100, la_d)
    fourier_spectra(lad_sample, x_Freq=True, y_dB = True, title="Spectre amplitude LA# synthèse", figure_num=10)

    st = 5000
    end = 15000
    sustain = 40000
    s = sol.tolist()[st:end]
    m = mi_b.tolist()[st:end + sustain]
    f = fa.tolist()[st:end]
    r = re.tolist()[st:end + sustain]

    symphony = s + s + s + m + f + f + f + r
    symphony_np = np.array(symphony)

    write_audio('symphony', 44100, symphony_np)


import matplotlib.pyplot as plt
import numpy as np
import Filtres
import Synthese

guitarFile = './audio/note_guitare_LAd.wav'
bassonFile = './audio/note_basson_plus_sinus_1000_Hz.wav'

def convFiltre(signal, filtre, y_dB=False, verbose=False):

    newN = len(signal.data) + len(filtre) - 1

    # Pad signals with 0 for Nh + Nx - 1
    paddedSignal = np.concatenate([signal.data, np.zeros(newN - len(signal.data))])
    paddedFiltre = np.concatenate([filtre, np.zeros(newN - len(filtre))])

    redressedSignal = np.abs(paddedSignal)

    # Convolution du filtre et du signal
    redressedFFTSignal = np.fft.fft(redressedSignal)
    filterFFT = np.fft.fft(paddedFiltre)

    if y_dB:
        redressedFFTSignal = 20 * np.log10(np.abs(redressedFFTSignal))
        filterFFT = 20 * np.log10(np.abs(paddedFiltre))


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
        plt.plot(paddedFiltre)
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

        # plt.show(np.abs(filteredSignalFrequentiel))

    return np.abs(filteredSignalTemporel)


###########################################################################################


def guitFunct():
    sample = Synthese.load_audio(guitarFile)
    sample_down = Synthese.down_sample(sample, plot=True, window=False)  # , start_time=0.17, end_time=0.18)
    amp, phase, inv_fft_signal = Synthese.fourier_spectra(sample_down, x_normalized=False, x_Freq=True, y_dB=False, showPhase=False)#, start_m=13500, end_m=13600)

    harm_amp, harm_phase = Synthese.get_harmonic_params(466, 32, amp, phase, sample_down, printResults=False)

    # ----

    wc = np.pi/1000
    dBGain = -3
    indB = False
    Npb, Hpb, hpb = Filtres.calcCoeffFIRPB(wc, dBGain)
    filtrePB = Filtres.filtrePasseBas(sample, forcedHVal=hpb, forcedNVal=Npb, y_dB=indB, normalized=True, verbose=True)
    envelope = convFiltre(sample, filtrePB, y_dB=indB, verbose=True)

    # Synthesis of notes
    Synthese.create_symphony(envelope, harm_amp, harm_phase, sample_down)

    plt.show()


def bassonFunct():
    sample = Synthese.load_audio(bassonFile)
    sample_down = Synthese.down_sample(sample, plot=False)  # , start_time=0.17, end_time=0.18)
    amp, phase = Synthese.fourier_spectra(sample, x_normalized=False, x_Freq=True, y_dB=False,
                                 showPhase=False)  # , start_m=0, end_m=1000)

    harm_amp, harm_phase = Synthese.get_harmonic_params(466, 32, amp, phase, sample, printResults=False)
    Synthese.sample_synthesis(466, harm_amp, harm_phase, sample)

    filtreCB = Filtres.filtreCoupeBande(sample, xFreq=False, normalized=False, verbose=True)

    plt.show()

guitFunct()
#bassonFunct()
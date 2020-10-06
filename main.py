import matplotlib.pyplot as plt
import numpy as np
import Filtres
import Synthese

guitarFile = './audio/note_guitare_LAd.wav'
bassonFile = './audio/note_basson_plus_sinus_1000_Hz.wav'

def convFiltre(signal, filtre, basson=False, y_dB=False, verbose=False, imprimerFiltre=False):

    newN = len(signal.data) + len(filtre) - 1

    nNorm = np.arange(0, 2 * np.pi * newN / signal.Fe, 2 * np.pi / signal.Fe)

    # Pad signals with 0 for Nh + Nx - 1
    paddedSignal = np.concatenate([signal.data, np.zeros(newN - len(signal.data))])
    paddedFiltre = np.concatenate([filtre, np.zeros(newN - len(filtre))])

    redressedSignal = np.abs(paddedSignal)

    # Convolution du filtre et du signal
    redressedFFTSignal = np.fft.fft(redressedSignal)
    filterFFT = np.fft.fft(paddedFiltre)

    if y_dB:
        redressedFFTSignal = 20 * np.log10(np.abs(redressedFFTSignal) / redressedSignal.size)
        filterFFT = 20 * np.log10(np.abs(paddedFiltre) / paddedFiltre.size)


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
        plt.plot(np.abs(redressedSignal) / redressedSignal.size)

        # Réponse du filtre
        plt.subplot(3,2,3)
        plt.title("Filtre")
        plt.plot(paddedFiltre)
        plt.subplot(3, 2, 4)
        plt.title("Filtre (freq)")
        plt.plot(np.abs(filterFFT))

        # Filtre * signal
        #filteredSignal = np.abs(redressedSignal * FIRpb)
        plt.subplot(3,2,5)
        plt.title("Signal conv. filtre passe-bas")
        plt.plot(filteredSignalTemporel)
        plt.subplot(3, 2, 6)
        plt.title("Signal conv. filtre passe-bas (freq)")
        plt.plot(np.abs(filteredSignalFrequentiel) / filteredSignalFrequentiel.size)

    if imprimerFiltre:
        # Réponse du filtre
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Filtre")
        plt.plot(paddedFiltre)
        plt.subplot(2, 1, 2)
        plt.title("Filtre RIF Passe-Bas (fréquentiel)")
        plt.plot(nNorm[0:int(nNorm.size/2)], 20 * np.log10(np.abs(filterFFT[0:int(filterFFT.size/2)])))
        plt.ylabel("Amplitude (dB)")
        plt.xlabel("Freq. [radians / échantillon]")

        # plt.show(np.abs(filteredSignalFrequentiel))

    return np.abs(filteredSignalTemporel)


###########################################################################################


def guitFunct():
    sample = Synthese.load_audio(guitarFile)
    sample_down = Synthese.down_sample(sample, plot=True, window=False)  # , start_time=0.17, end_time=0.18)
    amp, phase = Synthese.fourier_spectra(sample_down, x_normalized=True, x_Freq=False, y_dB=False, showPhase=False)#, start_m=13500, end_m=13600)

    harm_amp, harm_phase, harm_freq = Synthese.get_harmonic_params(466, 32, amp, phase, sample_down, printResults=True)

    # ----

    indB = False
    Ppb, Hpb, hpb = Filtres.calcCoeffFIRPB(np.pi/1000, -3)

    filtrePB = Filtres.filtreFIR(sample, forcedHVal=hpb, forcedPVal=Ppb, y_dB=indB, xFreq=True, normalized=False, verbose=True)
    envelope = convFiltre(sample, filtrePB, y_dB=False, verbose=True, imprimerFiltre=True)

    # Synthesis of notes
    Synthese.create_symphony(envelope, Ppb, harm_amp, harm_phase, harm_freq, sample_down)

    plt.show()


def bassonFunct():
    sample = Synthese.load_audio(bassonFile)
    sample_down = Synthese.down_sample(sample, plot=False, window=True)  # , start_time=0.17, end_time=0.18)
    amp, phase = Synthese.fourier_spectra(sample_down, x_normalized=False, x_Freq=True, y_dB=False,
                                 showPhase=False)  # , start_m=0, end_m=1000)

    harm_amp, harm_phase, harm_freq = Synthese.get_harmonic_params(240, 32, amp, phase, sample_down, printResults=False)
    Synthese.sample_synthesis(240, harm_amp, harm_phase, harm_freq, sample)

    filtreCB = Filtres.filtreCoupeBande(sample, y_dB=False, xFreq=True, normalized=False, verbose=True)

    # Unfiltered Signal
    nFreq = np.arange(0, sample.N, 1)
    nFreq = nFreq * sample.Fe / sample.N
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.title("Signal conv. filtre passe-bas")
    plt.plot(sample.data)
    plt.subplot(3, 2, 2)
    plt.title("Signal conv. filtre passe-bas (freq)")
    plt.plot(nFreq, np.abs(np.fft.fft(sample.data)) / sample.N)

    # Filtre * signal
    signalFiltered = np.convolve(sample.data, filtreCB)

    nFreq = np.arange(0, signalFiltered.size, 1)
    nFreq = nFreq * sample.Fe / signalFiltered.size
    plt.subplot(3, 2, 3)
    plt.title("Signal conv. filtre passe-bas")
    plt.plot(signalFiltered)
    plt.subplot(3, 2, 4)
    plt.title("Signal conv. filtre passe-bas (freq)")
    plt.plot(nFreq, np.abs(np.fft.fft(signalFiltered)) / signalFiltered.size)

    # Filtre * Filtre * signal
    signalFiltered2 = np.convolve(signalFiltered, filtreCB)

    nFreq = np.arange(0, signalFiltered2.size, 1)
    nFreq = nFreq * sample.Fe / signalFiltered2.size
    plt.subplot(3, 2, 5)
    plt.title("Signal conv. * 2 filtre passe-bas")
    plt.plot(signalFiltered2)
    plt.subplot(3, 2, 6)
    plt.title("Signal conv. * 2 filtre passe-bas (freq)")
    plt.plot(nFreq, np.abs(np.fft.fft(signalFiltered2)) / signalFiltered2.size)

    # Reponse filtres
    nFreq = np.arange(0, signalFiltered.size, 1)
    nFreq = nFreq * sample.Fe / signalFiltered.size
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("Amplitude signal conv. filtre coupe-bande")
    plt.plot(nFreq, np.abs(np.fft.fft(signalFiltered)) / signalFiltered.size)
    plt.ylabel("Amplitude")
    plt.axis([0, 1500, 0, 2000])
    plt.subplot(2, 1, 2)
    plt.title("Phase signal conv.  filtre coupe-bande")
    plt.plot(nFreq, np.angle(signalFiltered))
    plt.ylabel("Phase (rad)")
    plt.xlabel("Fréquence (Hz)")
    plt.axis([0, 1500, 0, 3])

    nFreq = np.arange(0, signalFiltered2.size, 1)
    nFreq = nFreq * sample.Fe / signalFiltered2.size
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("Amplitude signal conv. * 2 filtre coupe-bande")
    plt.plot(nFreq, np.abs(np.fft.fft(signalFiltered2)) / signalFiltered2.size)
    plt.ylabel("Amplitude")
    plt.axis([0, 1500, 0, 2000])
    plt.subplot(2, 1, 2)
    plt.title("Signal conv. * 2 filtre coupe-bande")
    plt.plot(nFreq, np.angle(signalFiltered2))
    plt.ylabel("Phase (rad)")
    plt.xlabel("Fréquence (Hz)")
    plt.axis([0, 1500, 0, 3])

    signalFilteredSample = Synthese.AudioSample(sample.Fe, signalFiltered2)

    sample_down = Synthese.down_sample(signalFilteredSample, plot=True, window=True)  # , start_time=0.17, end_time=0.18)
    amp, phase = Synthese.fourier_spectra(sample_down, x_normalized=False, x_Freq=True, y_dB=False,
                                 showPhase=False)  # , start_m=0, end_m=1000)

    harm_amp, harm_phase, harm_freq = Synthese.get_harmonic_params(240, 32, amp, phase, sample_down, printResults=False)
    Synthese.sample_synthesis(240, harm_amp, harm_phase, harm_freq, sample)

    Ppb, Hpb, hpb = Filtres.calcCoeffFIRPB(np.pi / 1000, -3)
    filtrePB = Filtres.filtreFIR(signalFilteredSample, forcedHVal=hpb, forcedPVal=Ppb, y_dB=False, xFreq=True, normalized=False,
                                 verbose=False)

    envelope = convFiltre(signalFilteredSample, filtrePB, y_dB=False, verbose=True, imprimerFiltre=True)

    basson_Note = Synthese.generate_synthesis(240, envelope, Ppb, harm_amp, harm_phase, harm_freq, sample_down, save_as="basson_Note")


    signalAudioFiltered = Synthese.write_audio("bassonFilteredSynthed", sample.Fe, signalFiltered)

    plt.show()

#guitFunct()
bassonFunct()
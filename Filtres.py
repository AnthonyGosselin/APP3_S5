import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

def filtrePasseBas(audioSample, y_dB=False, normalized=False, verbose=True):
    wc = np.pi / 1000
    Fe = audioSample.Fe
    N = audioSample.N
    w_norm = 2 * np.pi / N
    fc = wc * Fe / (2 * np.pi)

    m = Fe * N / Fe
    k = 2 * m + 1

    n = np.arange(0, w_norm * N, w_norm) if normalized else np.arange(0, N, 1)

    FIRpb = np.zeros(len(n))

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

    if verbose:
        # Réponse du filtre
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Filtre passe-bas")
        plt.plot(FIRpb)
        plt.subplot(2, 1, 2)
        plt.title("Filtre passe-bas (freq)")
        if y_dB:
            plt.plot(20*np.log10(np.abs(np.fft.fft(FIRpb))))
            plt.ylabel("Amplitude (dB)")
        else:
            plt.plot(np.abs(np.fft.fft(FIRpb)))
            plt.ylabel("Amplitude")
        plt.show()

    return FIRpb

def filtreCoupeBande(signal, normalized=False, verbose=False):

    N = signal.N
    #filtreCB = np.zeros(N)

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